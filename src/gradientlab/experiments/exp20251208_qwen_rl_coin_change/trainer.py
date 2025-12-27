"""
GRPO-style REINFORCE fine-tuning for a decoder-only LM on the Coin Change task.

Key PyTorch/HF fixes:
- Sample with model.generate() under torch.no_grad() + model.eval(),
  then recompute logits WITH grad (model.train(), use_cache=False).
- Enforce left padding and compute masks vectorially to isolate generated tokens.
- Make entropy/log-prob part of the graph (no @torch.no_grad around them).
- Safer training loop (AMP optional, gradient clipping, NaN guard).
- Save optimizer + RNG states for true resume.

Requirements:
  pip install torch transformers trackio

Notes:
  - This expects `coin_change_env.py` with:
      - generate_min_coins_instance(rng: random.Random) -> (coins, amount, _, _)
      - verify_solution(problem_text: str, predicted_text: str) -> float in [0,1]
"""

import math
import os
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoTokenizer, AutoModelForCausalLM
import trackio


# ---------------------------------------------------------
#  ENVIRONMENT
# ---------------------------------------------------------

from gradientlab.experiments.exp20251208_qwen_rl_coin_change.modeling import (
    init_qwen3_10m_bytelevel,
)
from gradientlab.experiments.exp20251208_qwen_rl_coin_change.rl_env import (
    generate_min_coins_instance,
    verify_solution,
    solve_min_coins,
)


# ---------------------------------------------------------
#  UTILITIES
# ---------------------------------------------------------


def set_global_seeds(seed: int = 42) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@dataclass
class TrainConfig:
    device: torch.device = field(
        default_factory=lambda: torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )

    # RL sampling
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50

    # GRPO
    batch_size: int = 32
    group_size: int = 8
    entropy_coef: float = 0.001
    max_grad_norm: float = 2.0

    # Optim
    lr: float = 1e-4
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0
    warmup_ratio: float = 0.1  # 10% warmup

    # AMP
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16  # use torch.float16 if you need GradScaler

    # Loop
    num_steps: int = 3_000
    log_every: int = 1
    save_every: int = 100
    sft_eval_every: int = 200
    rl_eval_every: int = 100
    save_dir: str = "./checkpoints_coinchange"

    # Evaluation
    num_eval_problems: int = 50
    num_sample_generations: int = 5  # how many to print/log

    # KL divergence regularization
    kl_coef: float = 0.5  # coefficient for KL penalty (higher = more regularization)

    # Resume
    resume_from: Optional[str] = None  # checkpoint directory to resume from

    # Experiment tracking
    project_name: str = "coin-change-rl"

    # SFT warmup phase (before RL)
    sft_steps: int = 3000  # number of supervised pretraining steps
    sft_batch_size: int = 32  # batch size for SFT
    sft_lr: float = 1e-4  # learning rate for SFT (typically higher than RL)

    # Optional speed/memory tweaks
    compile_forward: bool = True  # torch.compile for recompute forward


# ---------------------------------------------------------
#  PROMPTING
# ---------------------------------------------------------


def build_prompt(coins: List[int], amount: int) -> str:
    """
    Build the text the model sees as input.
    """
    coins_str = ",".join(str(c) for c in coins)
    prompt = (
        f"coins: {coins_str}\n"
        f"amount: {amount}\n\n"
    )
    return prompt


def generate_problem_batch(batch_size: int, rng: random.Random) -> List[Dict]:
    """
    Returns a list of problems:
      { "prompt": <input text>, "problem_text": <text for verify_solution>,
        "answer": <int>, "solution_coins": <List[int]> }
    """
    problems: List[Dict] = []
    for _ in range(batch_size):
        coins, amount, answer, solution_coins = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        problems.append({
            "prompt": prompt,
            "problem_text": prompt,
            "answer": answer,
            "solution_coins": solution_coins,
        })
    return problems


def build_completion(answer: int, solution_coins: List[int]) -> str:
    """
    Build the target completion text for SFT.
    """
    solution_str = "+".join(str(c) for c in solution_coins) if answer != -1 else ""
    return f"answer: {answer}\nsolution: {solution_str}\n"


def generate_sft_batch(
    batch_size: int, rng: random.Random
) -> List[Dict[str, str]]:
    """
    Generate batch of (prompt, completion) pairs for supervised fine-tuning.
    Returns list of {"prompt": ..., "completion": ...}
    """
    batch: List[Dict[str, str]] = []
    for _ in range(batch_size):
        coins, amount, answer, solution_coins = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        completion = build_completion(answer, solution_coins)
        batch.append({"prompt": prompt, "completion": completion})
    return batch


# ---------------------------------------------------------
#  MODEL & TOKENIZER
# ---------------------------------------------------------


def load_model_and_tokenizer(cfg: TrainConfig):
    model, tokenizer = init_qwen3_10m_bytelevel()
    # Tie pad to eos; use LEFT padding for decoder-only batching
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(cfg.device)

    if cfg.compile_forward and hasattr(torch, "compile"):
        # Compile helps the recompute forward path (use_cache=False)
        model = torch.compile(model)  # type: ignore

    return model, tokenizer


# ---------------------------------------------------------
#  ROLLOUT: sample completions and compute logprobs (WITH grad)
# ---------------------------------------------------------


@dataclass
class RolloutResult:
    """Results from rollout_and_compute_logprobs."""
    logprob_per_seq: torch.Tensor  # [B] sum of log-probs over generated tokens
    entropy_per_seq: torch.Tensor  # [B] mean token entropy over generated tokens
    decoded_completions: List[str]  # decoded completions (post-prompt)
    gen_ids: torch.Tensor  # [B, S] generated token ids
    gen_attn: torch.Tensor  # [B, S] attention mask for generated sequences
    gen_token_mask: torch.Tensor  # [B, S-1] mask for generated tokens in shifted space
    log_probs: torch.Tensor  # [B, S-1, V] log probabilities from policy


def rollout_and_compute_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: TrainConfig,
) -> RolloutResult:
    """
    For each prompt, sample one completion, then (with grads) recompute logits
    over the full sequence to extract log-probs/entropy on the generated tokens.

    Returns:
      RolloutResult containing log-probs, entropy, completions, and data needed for KL computation.
    """
    device = cfg.device

    # Encode batch WITHOUT EOS at end (only BOS at start)
    # This is critical: SFT trains on [BOS] prompt completion [EOS]
    # So generation must start from [BOS] prompt (no trailing EOS)
    #
    # With left-padding, we need: [PAD, ..., PAD, BOS, prompt_tokens]
    # So we insert BOS right before where content starts, not at position 0
    enc = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
    B = len(prompts)
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    # Find where content starts for each sequence (first non-pad position)
    # Then insert BOS there by shifting content and placing BOS
    old_ids = enc["input_ids"]
    old_mask = enc["attention_mask"]
    S = old_ids.size(1)

    # Create new tensors with one extra position
    new_ids = torch.full((B, S + 1), pad_id, dtype=old_ids.dtype)
    new_mask = torch.zeros((B, S + 1), dtype=old_mask.dtype)

    for i in range(B):
        # Find first non-pad position
        content_start = (old_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(content_start) > 0:
            start_idx = content_start[0].item()
        else:
            start_idx = S  # all padding (edge case)

        # Copy: [PAD...PAD, BOS, content...]
        # Padding stays at the beginning
        new_ids[i, :start_idx] = pad_id
        new_ids[i, start_idx] = bos_id
        new_ids[i, start_idx + 1:] = old_ids[i, start_idx:]

        new_mask[i, :start_idx] = 0
        new_mask[i, start_idx:] = 1

    input_ids = new_ids.to(device)
    attention_mask = new_mask.to(device)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # ---------- Sampling (no grad, eval mode) ----------
    model.eval()
    with torch.no_grad():
        # Disable torch.compile for generate() to avoid dynamic shape issues
        #with torch.compiler.disable():
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=True,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
        )
        gen_ids = gen_out.sequences  # [B, S]

    # Masks & lengths on full sequences
    gen_attn = (gen_ids != pad_id).long()  # [B, S]
    gen_lens = gen_attn.sum(dim=1)  # [B] number of non-pad tokens
    prefix_width = input_ids.size(1)  # batched prompt width (left-padded)

    # ---------- Recompute forward WITH grad for PG ----------
    model.train()
    # Autocast for memory/speed; GradScaler only needed for fp16
    # Support AMP on both CUDA and MPS
    use_amp = cfg.use_amp and (torch.cuda.is_available() or cfg.device.type == "mps")

    # We *do* want grads in this block:
    with torch.autocast(
        dtype=cfg.amp_dtype, device_type=cfg.device.type, enabled=use_amp
    ):
        outputs = model(gen_ids, attention_mask=gen_attn, use_cache=False)
        logits = outputs.logits  # [B, S, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]  # predicts token t+1
        shift_labels = gen_ids[:, 1:]  # actual token t+1
        S_minus_1 = shift_labels.size(1)

        # Build vectorized mask over label-space for generated tokens.
        # Label index t belongs to generation if t in [prefix_width-1, gen_len-2]
        # where gen_len is per-example (non-pad token count in full seq)
        t_index = torch.arange(S_minus_1, device=device).unsqueeze(0).expand(B, -1)
        start = prefix_width - 1
        end = (gen_lens - 1).unsqueeze(1)  # exclusive

        # Build mask for tokens after first EOS in the GENERATED portion only
        # (The prompt already contains an EOS from the tokenizer, so we must ignore it)
        # We only care about EOS tokens that appear at positions >= start (i.e., in generated part)
        is_eos_in_gen = (gen_ids[:, 1:] == eos_id) & (t_index >= start)  # [B, S-1]
        # Cumsum to mark all positions at or after first generated EOS
        eos_cumsum = is_eos_in_gen.cumsum(dim=1)  # [B, S-1]
        # Mask: True for tokens before first generated EOS, or the EOS token itself
        before_or_at_first_eos = (eos_cumsum == 0) | (is_eos_in_gen & (eos_cumsum == 1))

        gen_token_mask = (
            (t_index >= start)
            & (t_index < end)
            & (shift_labels != pad_id)
            & before_or_at_first_eos
        )

        # Log-probs and entropy
        log_probs = torch.log_softmax(shift_logits, dim=-1)  # [B, S-1, V]
        action_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(
            -1
        )  # [B, S-1]

        # Sum log-probs over generated tokens
        logprob_per_seq = (action_logp * gen_token_mask).sum(dim=1)  # [B]

        probs = log_probs.exp()
        entropy_tok = -(probs * log_probs).sum(dim=-1)  # [B, S-1]
        token_counts = gen_token_mask.sum(dim=1).clamp_min(1)  # [B]
        entropy_per_seq = (entropy_tok * gen_token_mask).sum(dim=1) / token_counts

    # ---------- Decode completions (post-prompt) ----------
    # Slice from prefix_width to end, let skip_special_tokens handle EOS/PAD cleanup
    decoded_completions: List[str] = []
    for i in range(B):
        comp_ids = gen_ids[i, prefix_width:]
        decoded_completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

    return RolloutResult(
        logprob_per_seq=logprob_per_seq,
        entropy_per_seq=entropy_per_seq,
        decoded_completions=decoded_completions,
        gen_ids=gen_ids,
        gen_attn=gen_attn,
        gen_token_mask=gen_token_mask,
        log_probs=log_probs,
    )


# ---------------------------------------------------------
#  REWARD
# ---------------------------------------------------------


def compute_rewards(
    problems: List[Dict[str, str]], completions: List[str]
) -> List[float]:
    rewards: List[float] = []
    for prob, comp in zip(problems, completions):
        r = float(verify_solution(prob["problem_text"], comp))
        rewards.append(r)
    return rewards


# ---------------------------------------------------------
#  SFT STEP (Supervised Fine-Tuning warmup)
# ---------------------------------------------------------


def sft_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    cfg: TrainConfig,
    rng: random.Random,
) -> Dict[str, float]:
    """
    One supervised fine-tuning step:
      - Generate batch of (prompt, completion) pairs
      - Compute cross-entropy loss on the completion tokens only
      - Backprop and update
    """
    device = cfg.device

    # Generate batch
    batch = generate_sft_batch(batch_size=cfg.sft_batch_size, rng=rng)

    # Tokenize prompts and full sequences separately to find completion boundaries
    prompts = [b["prompt"] for b in batch]
    full_texts = [b["prompt"] + b["completion"] for b in batch]

    # Tokenize with left padding for decoder-only
    # IMPORTANT: Get raw prompt lengths WITHOUT special tokens to correctly identify
    # where completion starts in the full sequence [BOS] prompt completion [EOS]
    prompt_enc_no_special = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
    full_enc = tokenizer(full_texts, return_tensors="pt", padding="longest")

    input_ids = full_enc["input_ids"].to(device)
    attention_mask = full_enc["attention_mask"].to(device)
    # Raw prompt length (without BOS/EOS), then add 1 for BOS to get position after prompt
    raw_prompt_lens = prompt_enc_no_special["attention_mask"].sum(dim=1).to(device)  # [B]
    # In full_enc: [BOS] prompt completion [EOS], completion starts at position (1 + raw_prompt_len)
    prompt_lens = raw_prompt_lens + 1  # +1 for BOS

    B, S = input_ids.shape
    pad_id = tokenizer.pad_token_id

    # Compute loss only on completion tokens (not prompt)
    # For left-padded sequences: prompt ends at position (S - (full_len - prompt_len))
    # But simpler: we know prompt_lens, and full sequence is left-padded
    # The prompt tokens are at positions [pad_len : pad_len + prompt_len]
    # Completion tokens are at positions [pad_len + prompt_len : ]

    full_lens = attention_mask.sum(dim=1)  # [B]
    pad_lens = S - full_lens  # [B] number of pad tokens at start

    # Build mask for completion tokens (in shifted label space)
    # shift_labels[i, t] corresponds to predicting token at position t+1
    # We want to mask positions where t+1 is in the completion part
    t_index = torch.arange(S - 1, device=device).unsqueeze(0).expand(B, -1)
    # Completion starts at pad_len + prompt_len, so in shifted space: pad_len + prompt_len - 1
    completion_start = (pad_lens + prompt_lens - 1).unsqueeze(1)  # [B, 1]
    completion_end = (S - 1)  # end of sequence in shifted space

    # Mask: True for completion tokens
    completion_mask = (t_index >= completion_start) & (t_index < completion_end)
    # Also exclude pad tokens
    shift_labels = input_ids[:, 1:]
    completion_mask = completion_mask & (shift_labels != pad_id)

    # Forward pass with AMP
    model.train()
    use_amp = cfg.use_amp and (torch.cuda.is_available() or cfg.device.type == "mps")

    with torch.autocast(
        dtype=cfg.amp_dtype, device_type=cfg.device.type, enabled=use_amp
    ):
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=False)
        logits = outputs.logits  # [B, S, V]

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]  # [B, S-1, V]

        # Cross-entropy loss on completion tokens only
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Masked mean
        masked_log_probs = token_log_probs * completion_mask
        num_tokens = completion_mask.sum().clamp_min(1)
        loss = -masked_log_probs.sum() / num_tokens

    # Backprop
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {"sft_loss": float("nan"), "sft_num_tokens": 0}

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return {
        "sft_loss": loss.item(),
        "sft_num_tokens": num_tokens.item(),
        "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
    }


# ---------------------------------------------------------
#  GRPO STEP
# ---------------------------------------------------------


def grpo_reinforce_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    cfg: TrainConfig,
    rng: random.Random,
    ref_model: Optional[AutoModelForCausalLM] = None,
) -> Dict[str, float]:
    """
    One GRPO step:
      - sample group_size completions per problem
      - baseline: mean reward per group
      - normalize advantages across all samples
      - loss = -E[adv * logprob] - entropy_coef * entropy_mean + kl_coef * KL(policy || ref)
    """
    device = cfg.device
    batch_size, group_size = cfg.batch_size, cfg.group_size
    # B = batch_size * group_size

    # 1) Generate problems
    problems = generate_problem_batch(batch_size=batch_size, rng=rng)
    prompts = [p["prompt"] for p in problems]

    # 2) Expand prompts (group_size per problem)
    prompts_expanded = [p for p in prompts for _ in range(group_size)]

    # 3) Rollout + logprobs (WITH grad through recompute)
    rollout = rollout_and_compute_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_expanded,
        cfg=cfg,
    )
    logprob_per_seq = rollout.logprob_per_seq
    entropy_per_seq = rollout.entropy_per_seq
    completions = rollout.decoded_completions

    # 4) Rewards
    problems_expanded = [p for p in problems for _ in range(group_size)]
    rewards = compute_rewards(problems_expanded, completions)
    rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)  # [B]

    # 5) Group baselines & advantages
    rewards_group = rewards_t.view(batch_size, group_size)  # [batch, group]
    baseline_group = rewards_group.mean(dim=1, keepdim=True)  # [batch, 1]
    advantages_group = rewards_group - baseline_group  # [batch, group]
    advantages = advantages_group.view(-1)  # [B]

    # Normalize advantages (stability)
    """ adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = (advantages - adv_mean) / adv_std """
    adv_mean = advantages_group.mean(dim=1, keepdim=True)
    adv_std = advantages_group.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    advantages = ((advantages_group - adv_mean) / adv_std).view(-1)

    advantages_detached = advantages.detach()

    # 6) Loss terms
    pg_loss = -(advantages_detached * logprob_per_seq).mean()
    entropy_mean = entropy_per_seq.mean()
    entropy_loss = -entropy_mean
    loss = pg_loss + cfg.entropy_coef * entropy_loss

    # 7) KL divergence penalty (if reference model provided)
    kl_mean = torch.tensor(0.0, device=device)
    if ref_model is not None and cfg.kl_coef > 0:
        # Compute reference model log-probs on the same generated sequences
        ref_model.eval()
        use_amp = cfg.use_amp and (torch.cuda.is_available() or device.type == "mps")
        with torch.no_grad():
            with torch.autocast(
                dtype=cfg.amp_dtype, device_type=device.type, enabled=use_amp
            ):
                ref_outputs = ref_model(
                    rollout.gen_ids, attention_mask=rollout.gen_attn, use_cache=False
                )
                ref_logits = ref_outputs.logits
                ref_shift_logits = ref_logits[:, :-1, :]
                ref_log_probs = torch.log_softmax(ref_shift_logits, dim=-1)

        # KL(policy || ref) = sum_x policy(x) * [log policy(x) - log ref(x)]
        # Per-token KL, then average over generated tokens
        # policy_log_probs: [B, S-1, V], ref_log_probs: [B, S-1, V]
        policy_probs = rollout.log_probs.exp()
        kl_per_token = (policy_probs * (rollout.log_probs - ref_log_probs)).sum(dim=-1)  # [B, S-1]

        # Mask to only count generated tokens
        token_counts = rollout.gen_token_mask.sum(dim=1).clamp_min(1)  # [B]
        kl_per_seq = (kl_per_token * rollout.gen_token_mask).sum(dim=1) / token_counts  # [B]
        kl_mean = kl_per_seq.mean()

        loss = loss + cfg.kl_coef * kl_mean

    # 8) Backprop & step (AMP aware)
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": float("nan"),
            "pg_loss": float("nan"),
            "kl": kl_mean.item(),
            "entropy": entropy_mean.item(),
            "reward_mean": rewards_t.mean().item(),
            "reward_std": rewards_t.std(unbiased=False).item(),
        }

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "kl": kl_mean.item(),
        "entropy": entropy_mean.item(),
        "reward_mean": rewards_t.mean().item(),
        "reward_std": rewards_t.std(unbiased=False).item(),
        "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
    }


# ---------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------


def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    cfg: TrainConfig,
    eval_rng: random.Random,
) -> Dict[str, float | List[Dict[str, str]]]:
    """
    Evaluate the model with greedy decoding on a fixed set of problems.

    Returns:
      - accuracy: fraction of problems with correct answer
      - reward_mean: mean reward across problems
      - samples: list of sample generations for logging
    """
    device = cfg.device
    model.eval()

    # Generate evaluation problems
    problems = generate_problem_batch(batch_size=cfg.num_eval_problems, rng=eval_rng)
    prompts = [p["prompt"] for p in problems]

    # Encode batch WITHOUT EOS at end (only BOS at start)
    # This matches SFT training: [BOS] prompt completion [EOS]
    # With left-padding: [PAD, ..., PAD, BOS, prompt_tokens]
    enc = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
    num_prompts = len(prompts)
    bos_id = tokenizer.bos_token_id
    pad_id = tokenizer.pad_token_id

    old_ids = enc["input_ids"]
    old_mask = enc["attention_mask"]
    S = old_ids.size(1)

    new_ids = torch.full((num_prompts, S + 1), pad_id, dtype=old_ids.dtype)
    new_mask = torch.zeros((num_prompts, S + 1), dtype=old_mask.dtype)

    for i in range(num_prompts):
        content_start = (old_mask[i] == 1).nonzero(as_tuple=True)[0]
        if len(content_start) > 0:
            start_idx = content_start[0].item()
        else:
            start_idx = S

        new_ids[i, :start_idx] = pad_id
        new_ids[i, start_idx] = bos_id
        new_ids[i, start_idx + 1:] = old_ids[i, start_idx:]

        new_mask[i, :start_idx] = 0
        new_mask[i, start_idx:] = 1

    input_ids = new_ids.to(device)
    attention_mask = new_mask.to(device)

    eos_id = tokenizer.eos_token_id

    # Greedy decoding
    with torch.no_grad():
        #with torch.compiler.disable():
        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=cfg.max_new_tokens,
            do_sample=False,  # greedy
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            return_dict_in_generate=True,
        )
        gen_ids = gen_out.sequences

    # Decode completions
    # prefix_width is the padded prompt length (input_ids is [BOS] + prompt, no trailing EOS)
    # model.generate() appends new tokens after input_ids, so completion starts at prefix_width
    prefix_width = input_ids.size(1)

    completions: List[str] = []
    for i in range(cfg.num_eval_problems):
        # Slice from end of prompt to end of sequence, decode with special tokens removed
        comp_ids = gen_ids[i, prefix_width:]
        completions.append(tokenizer.decode(comp_ids, skip_special_tokens=True))

    # Compute rewards
    rewards = compute_rewards(problems, completions)

    # Compute accuracy (reward == 1.0 means fully correct)
    accuracy = sum(1 for r in rewards if r == 1.0) / len(rewards)

    # Collect sample generations for logging
    samples: List[Dict] = []
    for i in range(min(cfg.num_sample_generations, len(problems))):
        expected = build_completion(problems[i]["answer"], problems[i]["solution_coins"])
        samples.append({
            "prompt": prompts[i],
            "completion": completions[i],
            "expected": expected,
            "reward": rewards[i],
        })

    model.train()

    return {
        "eval_accuracy": accuracy,
        "eval_reward_mean": sum(rewards) / len(rewards),
        "samples": samples,
    }


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
) -> LambdaLR:
    """
    Create a schedule with linear warmup and cosine decay.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def load_checkpoint(
    ckpt_dir: str,
    model: AutoModelForCausalLM,
    optimizer: torch.optim.Optimizer,
    scaler: torch.GradScaler,
    rng: random.Random,
    device: torch.device,
) -> int:
    """
    Load checkpoint and return the step to resume from.
    """
    # Load model weights using HF's from_pretrained would reload architecture,
    # so we load state dict directly
    from safetensors.torch import load_file

    state_dict = load_file(os.path.join(ckpt_dir, "model.safetensors"))
    model.load_state_dict(state_dict)

    # Load training state
    training_state = torch.load(
        os.path.join(ckpt_dir, "training_state.pt"),
        map_location=device,
        weights_only=False,
    )

    optimizer.load_state_dict(training_state["optimizer"])
    if "scaler" in training_state and training_state["scaler"] is not None:
        scaler.load_state_dict(training_state["scaler"])
    rng.setstate(training_state["py_rng"])
    torch.set_rng_state(training_state["torch_cpu_rng"])
    if training_state["torch_cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(training_state["torch_cuda_rng"])

    return training_state["step"]


# ---------------------------------------------------------
#  MAIN TRAINING LOOP
# ---------------------------------------------------------


def main():
    # Basic setup
    torch.set_float32_matmul_precision("high")
    set_global_seeds(42)

    cfg = TrainConfig()
    device = cfg.device
    ensure_dir(cfg.save_dir)

    # Initialize experiment tracking
    trackio.init(project=cfg.project_name)

    # Model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # GradScaler for AMP (persists across steps)
    use_amp = cfg.use_amp and (torch.cuda.is_available() or device.type == "mps")
    scaler_enabled = use_amp and (cfg.amp_dtype == torch.float16)
    scaler = torch.GradScaler(enabled=scaler_enabled)

    # RNG for problem generation
    rng = random.Random(42)

    # =================================================================
    # PHASE 1: SFT Warmup (learn the output format)
    # =================================================================
    if cfg.sft_steps > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 1: SFT Warmup ({cfg.sft_steps} steps)")
        print(f"{'='*60}\n")

        # SFT optimizer with higher LR
        sft_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.sft_lr,
            betas=cfg.adam_betas,
            weight_decay=cfg.weight_decay,
        )

        # SFT scheduler
        sft_warmup_steps = int(cfg.sft_steps * 0.1)
        sft_scheduler = get_cosine_schedule_with_warmup(
            sft_optimizer,
            num_warmup_steps=sft_warmup_steps,
            num_training_steps=cfg.sft_steps,
        )

        for sft_step_idx in range(1, cfg.sft_steps + 1):
            info = sft_step(
                model=model,
                tokenizer=tokenizer,
                optimizer=sft_optimizer,
                scaler=scaler,
                cfg=cfg,
                rng=rng,
            )

            sft_scheduler.step()
            current_lr = sft_scheduler.get_last_lr()[0]
            info["lr"] = current_lr

            # Log to trackio
            trackio.log(info, step=sft_step_idx)

            if sft_step_idx % cfg.log_every == 0:
                print(
                    f"[SFT] step {sft_step_idx:>6} | "
                    f"loss={info['sft_loss']:.4f} | "
                    f"tokens={info['sft_num_tokens']:.0f} | "
                    f"lr={current_lr:.2e}"
                )

            # Periodic evaluation during SFT
            if sft_step_idx % cfg.sft_eval_every == 0:
                eval_rng_copy = random.Random(123)
                eval_info = evaluate(model, tokenizer, cfg, eval_rng_copy)
                trackio.log({
                    "eval_accuracy": eval_info["eval_accuracy"],
                    "eval_reward_mean": eval_info["eval_reward_mean"],
                }, step=sft_step_idx)
                print(
                    f"  [EVAL] accuracy={eval_info['eval_accuracy']:.3f} | "
                    f"reward_mean={eval_info['eval_reward_mean']:.3f}"
                )
                # Print a few samples
                for i, sample in enumerate(eval_info["samples"][:3]):
                    print(f"  Sample {i+1}: {sample['completion'][:80]}...")

        print(f"\n{'='*60}")
        print("SFT Warmup Complete!")
        print(f"{'='*60}\n")

    # =================================================================
    # Create frozen reference model for KL regularization
    # =================================================================
    ref_model: Optional[AutoModelForCausalLM] = None
    if cfg.kl_coef > 0:
        import copy
        print("Creating frozen reference model for KL regularization...")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
        print(f"Reference model created (kl_coef={cfg.kl_coef})")

    # =================================================================
    # PHASE 2: RL Fine-tuning (GRPO)
    # =================================================================
    print(f"\n{'='*60}")
    print(f"PHASE 2: RL Fine-tuning ({cfg.num_steps} steps)")
    print(f"{'='*60}\n")

    # RL optimizer (fresh, with RL learning rate)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.adam_betas,
        weight_decay=cfg.weight_decay,
    )

    # LR scheduler with warmup and cosine decay for RL phase
    num_warmup_steps = int(cfg.num_steps * cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=cfg.num_steps,
    )

    # Resume from checkpoint if specified (skips SFT if resuming)
    start_step = 0
    if cfg.resume_from is not None:
        print(f"Resuming from checkpoint: {cfg.resume_from}")
        start_step = load_checkpoint(
            cfg.resume_from, model, optimizer, scaler, rng, device
        )
        # Step scheduler to correct position
        for _ in range(start_step):
            scheduler.step()
        print(f"Resumed at step {start_step}")

    # RL Training loop
    for step in range(start_step + 1, cfg.num_steps + 1):
        info = grpo_reinforce_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scaler=scaler,
            cfg=cfg,
            rng=rng,
            ref_model=ref_model,
        )

        # Step the scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        info["lr"] = current_lr

        # Log to trackio (offset step by sft_steps for continuous logging)
        trackio.log(info, step=cfg.sft_steps + step)

        if step % cfg.log_every == 0:
            print(
                f"[RL] step {step:>6} | "
                f"loss={info['loss']:.4f} | "
                f"pg={info['pg_loss']:.4f} | "
                f"kl={info['kl']:.4f} | "
                f"entropy={info['entropy']:.4f} | "
                f"R_mean={info['reward_mean']:.3f} | "
                f"R_std={info['reward_std']:.3f} | "
                f"lr={current_lr:.2e}"
            )

        # Evaluation
        if step % cfg.rl_eval_every == 0:
            eval_rng_copy = random.Random(123)  # Reset for consistent eval
            eval_info = evaluate(model, tokenizer, cfg, eval_rng_copy)

            # Log eval metrics (offset by sft_steps)
            trackio.log({
                "eval_accuracy": eval_info["eval_accuracy"],
                "eval_reward_mean": eval_info["eval_reward_mean"],
            }, step=cfg.sft_steps + step)

            print(
                f"  [EVAL] accuracy={eval_info['eval_accuracy']:.3f} | "
                f"reward_mean={eval_info['eval_reward_mean']:.3f}"
            )

            # Print sample generations
            print("  [SAMPLES]")
            for i, sample in enumerate(eval_info["samples"]):
                print(f"  --- Sample {i+1} (reward={sample['reward']:.3f}) ---")
                print(f"  Prompt: {sample['prompt']}")
                print(f"  Model: {sample['completion']}")
                print(f"  Expected: {sample['expected']}")

            # Log samples as text to trackio
            samples_text = "\n\n".join([
                f"**Sample {i+1}** (reward={s['reward']:.3f})\n"
                f"Prompt:\n```\n{s['prompt']}\n```\n"
                f"Model:\n```\n{s['completion']}\n```\n"
                f"Expected:\n```\n{s['expected']}\n```"
                for i, s in enumerate(eval_info["samples"])
            ])
            trackio.log({"eval_samples": samples_text}, step=cfg.sft_steps + step)

        # Save checkpoint
        if step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.save_dir, f"rl_step_{step:06d}")
            ensure_dir(ckpt_dir)
            # HF weights/tokenizer
            model.save_pretrained(ckpt_dir)
            # Optimizer + scaler + RNG states for full resume
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "py_rng": rng.getstate(),
                    "torch_cpu_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                os.path.join(ckpt_dir, "training_state.pt"),
            )
            print(f"  [SAVE] Checkpoint saved to {ckpt_dir}")

    # Final save
    ckpt_dir = os.path.join(cfg.save_dir, "final")
    ensure_dir(ckpt_dir)
    model.save_pretrained(ckpt_dir)
    print(f"Training complete! Final model saved to {ckpt_dir}")


if __name__ == "__main__":
    main()
