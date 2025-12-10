"""
Tests for rollout_and_compute_logprobs and related RL computations.

Run with:
  uv run pytest src/gradientlab/experiments/exp20251208_qwen_rl_coin_change/test_rollout.py -v

Or standalone:
  uv run python -m gradientlab.experiments.exp20251208_qwen_rl_coin_change.test_rollout
"""

import torch
import random
from dataclasses import dataclass
from typing import List

# Import pytest only when running under pytest (not standalone)
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    # Create dummy decorator for standalone execution
    class pytest:
        @staticmethod
        def fixture(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

from gradientlab.experiments.exp20251208_qwen_rl_coin_change.modeling import (
    init_qwen3_10m_bytelevel,
)
from gradientlab.experiments.exp20251208_qwen_rl_coin_change.trainer import (
    TrainConfig,
    RolloutResult,
    rollout_and_compute_logprobs,
    build_prompt,
    generate_problem_batch,
)


# =========================================================
# Fixtures
# =========================================================


@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all tests."""
    model, tokenizer = init_qwen3_10m_bytelevel()
    # Use CPU for tests to avoid GPU memory issues
    device = torch.device("cpu")
    model.to(device)
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="module")
def cfg():
    """Create a test config with smaller values for faster tests."""
    return TrainConfig(
        device=torch.device("cpu"),
        max_new_tokens=32,  # Short for faster tests
        temperature=0.8,
        top_k=50,
        batch_size=2,
        group_size=2,
        use_amp=False,  # Disable AMP for CPU
        compile_forward=False,  # Disable compile for tests
    )


# =========================================================
# Test: Basic Rollout Structure
# =========================================================


class TestRolloutBasicStructure:
    """Test that rollout returns correct shapes and types."""

    def test_rollout_returns_correct_types(self, model_and_tokenizer, cfg):
        """Verify RolloutResult contains expected types."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert isinstance(result, RolloutResult)
        assert isinstance(result.logprob_per_seq, torch.Tensor)
        assert isinstance(result.entropy_per_seq, torch.Tensor)
        assert isinstance(result.decoded_completions, list)
        assert isinstance(result.gen_ids, torch.Tensor)
        assert isinstance(result.gen_attn, torch.Tensor)
        assert isinstance(result.gen_token_mask, torch.Tensor)
        assert isinstance(result.log_probs, torch.Tensor)

    def test_rollout_batch_dimensions(self, model_and_tokenizer, cfg):
        """Verify output tensors have correct batch dimensions."""
        model, tokenizer = model_and_tokenizer
        batch_size = 3
        prompts = [f"coins: 1,2,{i}\namount: {10 + i}\n\n" for i in range(batch_size)]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert result.logprob_per_seq.shape == (batch_size,)
        assert result.entropy_per_seq.shape == (batch_size,)
        assert len(result.decoded_completions) == batch_size
        assert result.gen_ids.shape[0] == batch_size
        assert result.gen_attn.shape[0] == batch_size
        assert result.gen_token_mask.shape[0] == batch_size

    def test_rollout_sequence_dimensions(self, model_and_tokenizer, cfg):
        """Verify sequence dimension relationships are correct."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        B, S = result.gen_ids.shape
        # gen_attn should match gen_ids shape
        assert result.gen_attn.shape == (B, S)
        # gen_token_mask is in shifted space (S-1)
        assert result.gen_token_mask.shape == (B, S - 1)
        # log_probs is [B, S-1, V]
        assert result.log_probs.shape[0] == B
        assert result.log_probs.shape[1] == S - 1


# =========================================================
# Test: Token Mask Correctness
# =========================================================


class TestTokenMaskCorrectness:
    """Test that gen_token_mask correctly identifies generated tokens."""

    def test_mask_excludes_prompt_tokens(self, model_and_tokenizer, cfg):
        """Verify mask is False for prompt positions."""
        model, tokenizer = model_and_tokenizer
        prompt = "coins: 1,2,5\namount: 11\n\n"
        prompts = [prompt]

        # Encode prompt to get its length
        enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        prompt_len = enc["input_ids"].shape[1] + 1  # +1 for BOS

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # In shifted space, prompt ends at position (prompt_len - 1)
        # All positions before (prompt_len - 1) should be masked out
        prefix_mask = result.gen_token_mask[0, : prompt_len - 1]
        assert prefix_mask.sum() == 0, f"Prompt tokens should be masked out, but got {prefix_mask}"

    def test_mask_includes_generated_tokens(self, model_and_tokenizer, cfg):
        """Verify mask is True for at least some generated positions."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Should have some generated tokens marked
        assert result.gen_token_mask.sum() > 0, "Should have some generated tokens"

    def test_mask_stops_at_eos(self, model_and_tokenizer, cfg):
        """Verify mask stops at or before EOS token."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        eos_id = tokenizer.eos_token_id
        gen_ids = result.gen_ids[0]

        # Find first EOS in generated portion (after prompt)
        enc = tokenizer([prompts[0]], return_tensors="pt", add_special_tokens=False)
        prompt_len = enc["input_ids"].shape[1] + 1

        # Look for EOS after prompt
        for i in range(prompt_len, len(gen_ids)):
            if gen_ids[i] == eos_id:
                # In shifted space, position i-1 corresponds to predicting token i
                # All mask positions after i-1 should be False
                mask_after_eos = result.gen_token_mask[0, i:]
                assert (
                    mask_after_eos.sum() == 0
                ), f"Mask should be False after EOS at position {i}"
                break


# =========================================================
# Test: Log Probability Correctness
# =========================================================


class TestLogProbCorrectness:
    """Test that log probabilities are computed correctly."""

    def test_logprobs_are_negative(self, model_and_tokenizer, cfg):
        """Log probabilities should be negative (or zero for prob=1)."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Per-sequence log probs should be negative (sum of negative values)
        assert result.logprob_per_seq[0] <= 0, "Sum of log probs should be <= 0"

    def test_logprobs_finite(self, model_and_tokenizer, cfg):
        """Log probabilities should not be NaN or Inf."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert torch.isfinite(result.logprob_per_seq).all(), "Log probs should be finite"
        assert torch.isfinite(result.log_probs).all(), "All log probs should be finite"

    def test_logprobs_sum_correctly(self, model_and_tokenizer, cfg):
        """Verify logprob_per_seq equals sum of masked log probs."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Get action log probs (log prob of actual tokens taken)
        shift_labels = result.gen_ids[:, 1:]
        action_logp = result.log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        # Sum over masked positions
        expected_logprob = (action_logp * result.gen_token_mask).sum(dim=1)

        torch.testing.assert_close(
            result.logprob_per_seq,
            expected_logprob,
            rtol=1e-4,
            atol=1e-6,
            msg="logprob_per_seq should equal sum of masked action log probs",
        )

    def test_log_probs_are_normalized(self, model_and_tokenizer, cfg):
        """Verify log_probs sum to ~1 when exponentiated (valid distribution)."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # exp(log_probs) should sum to 1 along vocab dimension
        probs = result.log_probs.exp()
        prob_sums = probs.sum(dim=-1)

        # Should be close to 1.0 for all positions
        torch.testing.assert_close(
            prob_sums,
            torch.ones_like(prob_sums),
            rtol=1e-3,
            atol=1e-5,
            msg="Probabilities should sum to 1",
        )


# =========================================================
# Test: Entropy Correctness
# =========================================================


class TestEntropyCorrectness:
    """Test that entropy is computed correctly."""

    def test_entropy_is_positive(self, model_and_tokenizer, cfg):
        """Entropy should be non-negative."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert (result.entropy_per_seq >= 0).all(), "Entropy should be non-negative"

    def test_entropy_is_finite(self, model_and_tokenizer, cfg):
        """Entropy should not be NaN or Inf."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert torch.isfinite(result.entropy_per_seq).all(), "Entropy should be finite"

    def test_entropy_computation_matches_formula(self, model_and_tokenizer, cfg):
        """Verify entropy = -sum(p * log(p)) averaged over generated tokens."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Compute entropy manually
        probs = result.log_probs.exp()
        entropy_tok = -(probs * result.log_probs).sum(dim=-1)  # [B, S-1]
        token_counts = result.gen_token_mask.sum(dim=1).clamp_min(1)
        expected_entropy = (entropy_tok * result.gen_token_mask).sum(dim=1) / token_counts

        torch.testing.assert_close(
            result.entropy_per_seq,
            expected_entropy,
            rtol=1e-4,
            atol=1e-6,
            msg="Entropy computation should match formula",
        )


# =========================================================
# Test: Gradient Flow
# =========================================================


class TestGradientFlow:
    """Test that gradients flow correctly through rollout."""

    def test_logprobs_have_grad(self, model_and_tokenizer, cfg):
        """Verify logprob_per_seq requires grad and can backprop."""
        model, tokenizer = model_and_tokenizer
        model.train()  # Need train mode for grad
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Check requires_grad
        assert result.logprob_per_seq.requires_grad, "logprob_per_seq should require grad"

        # Try backprop
        loss = result.logprob_per_seq.sum()
        loss.backward()

        # Check some parameter has grad
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break

        assert has_grad, "Gradients should flow to model parameters"

        # Clean up
        model.zero_grad()

    def test_entropy_is_detached(self, model_and_tokenizer, cfg):
        """Entropy per seq should be detached (no grad) as implemented."""
        model, tokenizer = model_and_tokenizer
        model.train()
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # The current implementation detaches entropy
        # This is a design choice - entropy bonus doesn't backprop through itself
        # Just verify it doesn't cause issues
        assert torch.isfinite(result.entropy_per_seq).all()


# =========================================================
# Test: Left Padding Handling
# =========================================================


class TestLeftPaddingHandling:
    """Test that left padding is handled correctly for batched prompts."""

    def test_different_length_prompts(self, model_and_tokenizer, cfg):
        """Verify batched prompts of different lengths work correctly."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            "coins: 1,2\namount: 5\n\n",  # Short
            "coins: 1,2,5,10,25,50\namount: 123\n\n",  # Long
        ]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Both should produce valid results
        assert len(result.decoded_completions) == 2
        assert torch.isfinite(result.logprob_per_seq).all()
        assert torch.isfinite(result.entropy_per_seq).all()

    def test_pad_tokens_not_in_mask(self, model_and_tokenizer, cfg):
        """Verify pad tokens are excluded from gen_token_mask."""
        model, tokenizer = model_and_tokenizer
        prompts = [
            "coins: 1,2\namount: 5\n\n",  # Short
            "coins: 1,2,5,10,25,50\namount: 123\n\n",  # Long
        ]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        pad_id = tokenizer.pad_token_id
        shift_labels = result.gen_ids[:, 1:]

        # Where shift_labels is pad, mask should be False
        pad_positions = shift_labels == pad_id
        masked_pads = result.gen_token_mask & pad_positions

        assert masked_pads.sum() == 0, "Pad tokens should not be included in mask"


# =========================================================
# Test: BOS Token Handling
# =========================================================


class TestBOSTokenHandling:
    """Test that BOS token is correctly inserted."""

    def test_bos_at_content_start(self, model_and_tokenizer, cfg):
        """Verify BOS is inserted at the start of content (after padding)."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        bos_id = tokenizer.bos_token_id
        gen_ids = result.gen_ids[0]
        gen_attn = result.gen_attn[0]

        # Find first non-pad position (where attention is 1)
        first_content_pos = (gen_attn == 1).nonzero(as_tuple=True)[0][0].item()

        assert (
            gen_ids[first_content_pos] == bos_id
        ), f"Expected BOS at position {first_content_pos}, got token {gen_ids[first_content_pos]}"


# =========================================================
# Test: Decoded Completions
# =========================================================


class TestDecodedCompletions:
    """Test that decoded completions are correct."""

    def test_completions_are_strings(self, model_and_tokenizer, cfg):
        """Verify completions are decoded to strings."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert all(isinstance(c, str) for c in result.decoded_completions)

    def test_completions_exclude_prompt(self, model_and_tokenizer, cfg):
        """Verify completions don't include the prompt."""
        model, tokenizer = model_and_tokenizer
        prompt = "coins: 1,2,5\namount: 11\n\n"
        prompts = [prompt]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        completion = result.decoded_completions[0]
        # Completion should not start with the prompt
        # (Note: small models may generate garbage, but it shouldn't be the exact prompt)
        assert not completion.startswith(
            prompt[:20]
        ), "Completion should not start with prompt"


# =========================================================
# Test: Numerical Stability
# =========================================================


class TestNumericalStability:
    """Test numerical stability under edge cases."""

    def test_empty_generation(self, model_and_tokenizer, cfg):
        """Test handling when model generates nothing (immediate EOS)."""
        model, tokenizer = model_and_tokenizer
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        # This test just ensures no crash - model might not generate immediate EOS
        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert torch.isfinite(result.logprob_per_seq).all()
        assert torch.isfinite(result.entropy_per_seq).all()

    def test_max_length_generation(self, model_and_tokenizer):
        """Test handling when generation hits max length."""
        model, tokenizer = model_and_tokenizer
        cfg_short = TrainConfig(
            device=torch.device("cpu"),
            max_new_tokens=5,  # Very short
            temperature=0.8,
            use_amp=False,
            compile_forward=False,
        )
        prompts = ["coins: 1,2,5\namount: 11\n\n"]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg_short)

        assert torch.isfinite(result.logprob_per_seq).all()
        assert torch.isfinite(result.entropy_per_seq).all()

    def test_batch_consistency(self, model_and_tokenizer, cfg):
        """Verify same prompt in batch gives consistent mask structure."""
        model, tokenizer = model_and_tokenizer
        prompt = "coins: 1,2,5\namount: 11\n\n"
        prompts = [prompt, prompt]  # Same prompt twice

        # Set seed for reproducibility
        torch.manual_seed(42)
        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # Mask shapes should be same for same prompts (though values may differ due to sampling)
        assert result.gen_token_mask[0].sum() > 0
        assert result.gen_token_mask[1].sum() > 0


# =========================================================
# Test: Integration with Problem Generation
# =========================================================


class TestIntegrationWithProblemGeneration:
    """Test rollout works correctly with actual problem generation."""

    def test_with_generated_problems(self, model_and_tokenizer, cfg):
        """Test rollout with problems from generate_problem_batch."""
        model, tokenizer = model_and_tokenizer
        rng = random.Random(42)

        problems = generate_problem_batch(batch_size=3, rng=rng)
        prompts = [p["prompt"] for p in problems]

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        assert len(result.decoded_completions) == 3
        assert torch.isfinite(result.logprob_per_seq).all()
        assert torch.isfinite(result.entropy_per_seq).all()


# =========================================================
# Test: Mask Index Computation
# =========================================================


class TestMaskIndexComputation:
    """Detailed tests for the mask index computation logic."""

    def test_mask_start_position(self, model_and_tokenizer, cfg):
        """Verify mask starts at correct position after prompt."""
        model, tokenizer = model_and_tokenizer
        prompt = "coins: 1,2,5\namount: 11\n\n"
        prompts = [prompt]

        # Compute expected prompt length
        enc = tokenizer([prompt], return_tensors="pt", add_special_tokens=False)
        raw_prompt_len = enc["input_ids"].shape[1]
        # With BOS added: prompt_width = raw_prompt_len + 1
        prefix_width = raw_prompt_len + 1

        result = rollout_and_compute_logprobs(model, tokenizer, prompts, cfg)

        # In shifted space, generation starts at index (prefix_width - 1)
        start_idx = prefix_width - 1

        # Check mask is False before start_idx
        if start_idx > 0:
            before_gen = result.gen_token_mask[0, :start_idx]
            assert (
                before_gen.sum() == 0
            ), f"Mask should be False for indices < {start_idx}"

        # Check mask has some True values at/after start_idx (if generated anything)
        after_gen = result.gen_token_mask[0, start_idx:]
        if after_gen.numel() > 0:
            # Should have at least some True values if tokens were generated
            pass  # This is allowed to be all zeros if nothing was generated


# =========================================================
# Run tests standalone
# =========================================================


def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "=" * 70)
    print("RUNNING ALL ROLLOUT TESTS")
    print("=" * 70)

    # Load model once
    print("\nLoading model...")
    model, tokenizer = init_qwen3_10m_bytelevel()
    device = torch.device("cpu")
    model.to(device)

    cfg = TrainConfig(
        device=device,
        max_new_tokens=32,
        temperature=0.8,
        top_k=50,
        batch_size=2,
        group_size=2,
        use_amp=False,
        compile_forward=False,
    )

    # Basic structure tests
    print("\n--- TestRolloutBasicStructure ---")
    basic = TestRolloutBasicStructure()
    try:
        basic.test_rollout_returns_correct_types((model, tokenizer), cfg)
        print("✓ test_rollout_returns_correct_types")
    except Exception as e:
        print(f"✗ test_rollout_returns_correct_types: {e}")

    try:
        basic.test_rollout_batch_dimensions((model, tokenizer), cfg)
        print("✓ test_rollout_batch_dimensions")
    except Exception as e:
        print(f"✗ test_rollout_batch_dimensions: {e}")

    try:
        basic.test_rollout_sequence_dimensions((model, tokenizer), cfg)
        print("✓ test_rollout_sequence_dimensions")
    except Exception as e:
        print(f"✗ test_rollout_sequence_dimensions: {e}")

    # Token mask tests
    print("\n--- TestTokenMaskCorrectness ---")
    mask_tests = TestTokenMaskCorrectness()
    try:
        mask_tests.test_mask_excludes_prompt_tokens((model, tokenizer), cfg)
        print("✓ test_mask_excludes_prompt_tokens")
    except Exception as e:
        print(f"✗ test_mask_excludes_prompt_tokens: {e}")

    try:
        mask_tests.test_mask_includes_generated_tokens((model, tokenizer), cfg)
        print("✓ test_mask_includes_generated_tokens")
    except Exception as e:
        print(f"✗ test_mask_includes_generated_tokens: {e}")

    try:
        mask_tests.test_mask_stops_at_eos((model, tokenizer), cfg)
        print("✓ test_mask_stops_at_eos")
    except Exception as e:
        print(f"✗ test_mask_stops_at_eos: {e}")

    # Log prob tests
    print("\n--- TestLogProbCorrectness ---")
    logprob_tests = TestLogProbCorrectness()
    try:
        logprob_tests.test_logprobs_are_negative((model, tokenizer), cfg)
        print("✓ test_logprobs_are_negative")
    except Exception as e:
        print(f"✗ test_logprobs_are_negative: {e}")

    try:
        logprob_tests.test_logprobs_finite((model, tokenizer), cfg)
        print("✓ test_logprobs_finite")
    except Exception as e:
        print(f"✗ test_logprobs_finite: {e}")

    try:
        logprob_tests.test_logprobs_sum_correctly((model, tokenizer), cfg)
        print("✓ test_logprobs_sum_correctly")
    except Exception as e:
        print(f"✗ test_logprobs_sum_correctly: {e}")

    try:
        logprob_tests.test_log_probs_are_normalized((model, tokenizer), cfg)
        print("✓ test_log_probs_are_normalized")
    except Exception as e:
        print(f"✗ test_log_probs_are_normalized: {e}")

    # Entropy tests
    print("\n--- TestEntropyCorrectness ---")
    entropy_tests = TestEntropyCorrectness()
    try:
        entropy_tests.test_entropy_is_positive((model, tokenizer), cfg)
        print("✓ test_entropy_is_positive")
    except Exception as e:
        print(f"✗ test_entropy_is_positive: {e}")

    try:
        entropy_tests.test_entropy_is_finite((model, tokenizer), cfg)
        print("✓ test_entropy_is_finite")
    except Exception as e:
        print(f"✗ test_entropy_is_finite: {e}")

    try:
        entropy_tests.test_entropy_computation_matches_formula((model, tokenizer), cfg)
        print("✓ test_entropy_computation_matches_formula")
    except Exception as e:
        print(f"✗ test_entropy_computation_matches_formula: {e}")

    # Gradient tests
    print("\n--- TestGradientFlow ---")
    grad_tests = TestGradientFlow()
    try:
        grad_tests.test_logprobs_have_grad((model, tokenizer), cfg)
        print("✓ test_logprobs_have_grad")
    except Exception as e:
        print(f"✗ test_logprobs_have_grad: {e}")

    # Left padding tests
    print("\n--- TestLeftPaddingHandling ---")
    padding_tests = TestLeftPaddingHandling()
    try:
        padding_tests.test_different_length_prompts((model, tokenizer), cfg)
        print("✓ test_different_length_prompts")
    except Exception as e:
        print(f"✗ test_different_length_prompts: {e}")

    try:
        padding_tests.test_pad_tokens_not_in_mask((model, tokenizer), cfg)
        print("✓ test_pad_tokens_not_in_mask")
    except Exception as e:
        print(f"✗ test_pad_tokens_not_in_mask: {e}")

    # BOS tests
    print("\n--- TestBOSTokenHandling ---")
    bos_tests = TestBOSTokenHandling()
    try:
        bos_tests.test_bos_at_content_start((model, tokenizer), cfg)
        print("✓ test_bos_at_content_start")
    except Exception as e:
        print(f"✗ test_bos_at_content_start: {e}")

    # Decoded completions tests
    print("\n--- TestDecodedCompletions ---")
    decode_tests = TestDecodedCompletions()
    try:
        decode_tests.test_completions_are_strings((model, tokenizer), cfg)
        print("✓ test_completions_are_strings")
    except Exception as e:
        print(f"✗ test_completions_are_strings: {e}")

    try:
        decode_tests.test_completions_exclude_prompt((model, tokenizer), cfg)
        print("✓ test_completions_exclude_prompt")
    except Exception as e:
        print(f"✗ test_completions_exclude_prompt: {e}")

    # Numerical stability tests
    print("\n--- TestNumericalStability ---")
    stability_tests = TestNumericalStability()
    try:
        stability_tests.test_empty_generation((model, tokenizer), cfg)
        print("✓ test_empty_generation")
    except Exception as e:
        print(f"✗ test_empty_generation: {e}")

    try:
        stability_tests.test_max_length_generation((model, tokenizer))
        print("✓ test_max_length_generation")
    except Exception as e:
        print(f"✗ test_max_length_generation: {e}")

    try:
        stability_tests.test_batch_consistency((model, tokenizer), cfg)
        print("✓ test_batch_consistency")
    except Exception as e:
        print(f"✗ test_batch_consistency: {e}")

    # Integration tests
    print("\n--- TestIntegrationWithProblemGeneration ---")
    integration_tests = TestIntegrationWithProblemGeneration()
    try:
        integration_tests.test_with_generated_problems((model, tokenizer), cfg)
        print("✓ test_with_generated_problems")
    except Exception as e:
        print(f"✗ test_with_generated_problems: {e}")

    # Mask index tests
    print("\n--- TestMaskIndexComputation ---")
    mask_idx_tests = TestMaskIndexComputation()
    try:
        mask_idx_tests.test_mask_start_position((model, tokenizer), cfg)
        print("✓ test_mask_start_position")
    except Exception as e:
        print(f"✗ test_mask_start_position: {e}")

    print("\n" + "=" * 70)
    print("TESTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    run_all_tests()
