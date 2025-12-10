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
  pip install torch transformers

Notes:
  - This expects `coin_change_env.py` with:
      - generate_min_coins_instance(rng: random.Random) -> (coins, amount, _, _)
      - verify_solution(problem_text: str, predicted_text: str) -> float in [0,1]
"""

import os
import math
import json
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------
#  ENVIRONMENT
# ---------------------------------------------------------

from gradientlab.experiments.exp20251208_qwen_rl_coin_change.modeling import (
    init_qwen3_10m_bytelevel,
)
from gradientlab.experiments.exp20251208_qwen_rl_coin_change.rl_env import (
    generate_min_coins_instance,
    verify_solution,
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
    device: torch.device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # RL sampling
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int = 50

    # GRPO
    batch_size: int = 4
    group_size: int = 2
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # Optim
    lr: float = 8e-5
    adam_betas: Tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.0

    # AMP
    use_amp: bool = True
    amp_dtype: torch.dtype = torch.float16  # use torch.float16 if you need GradScaler

    # Loop
    num_steps: int = 10_000
    log_every: int = 10
    save_every: int = 100
    save_dir: str = "./checkpoints_coinchange"

    # Optional speed/memory tweaks
    enable_grad_checkpointing: bool = False
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
        "Please answer in the following format:\n"
        "answer: <min number of coins, or -1 if impossible>\n"
        "solution: <coin1+coin2+... or empty if impossible>\n"
    )
    return prompt


def generate_problem_batch(batch_size: int, rng: random.Random) -> List[Dict[str, str]]:
    """
    Returns a list of problems:
      { "prompt": <input text>, "problem_text": <text for verify_solution> }
    """
    problems: List[Dict[str, str]] = []
    for _ in range(batch_size):
        coins, amount, _, _ = generate_min_coins_instance(rng=rng)
        prompt = build_prompt(coins, amount)
        problems.append({"prompt": prompt, "problem_text": prompt})
    return problems


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


def rollout_and_compute_logprobs(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    cfg: TrainConfig,
    sample_generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """
    For each prompt, sample one completion, then (with grads) recompute logits
    over the full sequence to extract log-probs/entropy on the generated tokens.

    Returns:
      - logprob_per_seq: [B] sum of log-probs over generated tokens
      - entropy_per_seq: [B] mean token entropy over generated tokens
      - decoded_completions: List[str] decoded completions (post-prompt)
    """
    device = cfg.device

    # Encode batch
    enc = tokenizer(prompts, return_tensors="pt", padding="longest")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    B = input_ids.size(0)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    # ---------- Sampling (no grad, eval mode) ----------
    model.eval()
    with torch.no_grad():
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
    use_amp = cfg.use_amp and torch.cuda.is_available()

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
        gen_token_mask = (t_index >= start) & (t_index < end) & (shift_labels != pad_id)

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
    decoded_completions: List[str] = []
    for i in range(B):
        gl = gen_lens[i].item()
        comp_ids = gen_ids[i, prefix_width:gl]
        decoded_completions.append(tokenizer.decode(comp_ids))

    return logprob_per_seq, entropy_per_seq, decoded_completions


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
#  GRPO STEP
# ---------------------------------------------------------


def grpo_reinforce_step(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    optimizer: torch.optim.Optimizer,
    cfg: TrainConfig,
    rng: random.Random,
    torch_sampler: Optional[torch.Generator] = None,
) -> Dict[str, float]:
    """
    One GRPO step:
      - sample group_size completions per problem
      - baseline: mean reward per group
      - normalize advantages across all samples
      - loss = -E[adv * logprob] - entropy_coef * entropy_mean
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
    logprob_per_seq, entropy_per_seq, completions = rollout_and_compute_logprobs(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts_expanded,
        cfg=cfg,
        sample_generator=torch_sampler,
    )

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
    adv_mean = advantages.mean()
    adv_std = advantages.std(unbiased=False).clamp_min(1e-6)
    advantages = (advantages - adv_mean) / adv_std
    advantages_detached = advantages.detach()

    # 6) Loss terms
    pg_loss = -(advantages_detached * logprob_per_seq).mean()
    entropy_mean = entropy_per_seq.mean()
    entropy_loss = -entropy_mean
    loss = pg_loss + cfg.entropy_coef * entropy_loss

    # 7) Backprop & step (AMP aware)
    if not torch.isfinite(loss):
        optimizer.zero_grad(set_to_none=True)
        return {
            "loss": float("nan"),
            "pg_loss": float("nan"),
            "entropy": entropy_mean.item(),
            "reward_mean": rewards_t.mean().item(),
            "reward_std": rewards_t.std(unbiased=False).item(),
        }

    use_amp = cfg.use_amp and torch.cuda.is_available()
    scaler_needed = use_amp and (cfg.amp_dtype == torch.float16)
    scaler = torch.GradScaler(enabled=scaler_needed)

    optimizer.zero_grad(set_to_none=True)
    if scaler_needed:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        optimizer.step()

    return {
        "loss": loss.item(),
        "pg_loss": pg_loss.item(),
        "entropy": entropy_mean.item(),
        "reward_mean": rewards_t.mean().item(),
        "reward_std": rewards_t.std(unbiased=False).item(),
    }


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

    # Model & tokenizer
    model, tokenizer = load_model_and_tokenizer(cfg)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.adam_betas,
        weight_decay=cfg.weight_decay,
    )

    # Deterministic sampling generator (debug-friendly)
    torch_sampler = torch.Generator(
        device=device.type if device.type == "cuda" else "cpu"
    )
    torch_sampler.manual_seed(42)

    rng = random.Random(42)

    # Train
    for step in range(1, cfg.num_steps + 1):
        info = grpo_reinforce_step(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            cfg=cfg,
            rng=rng,
            torch_sampler=torch_sampler,
        )

        if step % cfg.log_every == 0:
            print(
                f"step {step:>6} | "
                f"loss={info['loss']:.4f} | "
                f"pg={info['pg_loss']:.4f} | "
                f"entropy={info['entropy']:.4f} | "
                f"R_mean={info['reward_mean']:.3f} | "
                f"R_std={info['reward_std']:.3f}"
            )

        if step % cfg.save_every == 0:
            ckpt_dir = os.path.join(cfg.save_dir, f"step_{step:06d}")
            ensure_dir(ckpt_dir)
            # HF weights/tokenizer
            model.save_pretrained(ckpt_dir)
            # Optimizer + RNG states for full resume
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "py_rng": rng.getstate(),
                    "torch_cpu_rng": torch.get_rng_state(),
                    "torch_cuda_rng": torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None,
                },
                os.path.join(ckpt_dir, "training_state.pt"),
            )


if __name__ == "__main__":
    main()
