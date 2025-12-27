import math
import random
import re
from typing import List, Tuple, Optional


# ---------- Core DP solver ----------


def solve_min_coins(coins: List[int], amount: int) -> Tuple[int, List[int]]:
    """
    Classic unbounded coin-change DP to get:
      - minimum number of coins
      - one optimal solution as a list of coin values

    Returns:
      (answer, solution_coins)

    If no solution exists:
      answer = -1, solution_coins = []
    """
    if amount < 0:
        return -1, []

    # dp[x] = min number of coins to make x
    INF = amount + 1
    dp = [0] + [INF] * amount
    prev_coin = [-1] * (amount + 1)

    for c in coins:
        if c <= 0:
            continue
        for x in range(c, amount + 1):
            if dp[x - c] + 1 < dp[x]:
                dp[x] = dp[x - c] + 1
                prev_coin[x] = c

    if dp[amount] == INF:
        return -1, []

    # Reconstruct one optimal solution
    solution = []
    curr = amount
    while curr > 0:
        c = prev_coin[curr]
        if c == -1:
            # Should not happen if dp[amount] != INF, but just in case
            return -1, []
        solution.append(c)
        curr -= c

    solution.reverse()  # optional, for nicer ordering
    return dp[amount], solution


# ---------- Sample generation ----------


def generate_min_coins_instance(
    min_coin_value: int = 1,
    max_coin_value: int = 50,
    min_num_coins: int = 1,
    max_num_coins: int = 6,
    min_amount: int = 1,
    max_amount: int = 200,
    ensure_solvable: bool = True,
    rng: Optional[random.Random] = None,
) -> Tuple[List[int], int, int, List[int]]:
    """
    Generates a single (coins, amount, answer, solution_coins) instance.

    If ensure_solvable=True, it resamples until there is at least one solution.
    """
    if rng is None:
        rng = random

    while True:
        k = rng.randint(min_num_coins, max_num_coins)
        # Sample distinct coin values
        coin_values = sorted(rng.sample(range(min_coin_value, max_coin_value + 1), k))
        amount = rng.randint(min_amount, max_amount)

        answer, solution_coins = solve_min_coins(coin_values, amount)

        if ensure_solvable and answer == -1:
            # resample
            continue

        return coin_values, amount, answer, solution_coins


def format_sample_text(
    coins: List[int], amount: int, answer: int, solution_coins: List[int]
) -> str:
    """
    Formats the instance as a text block, e.g.:

    coins: 1,2,5
    amount: 11

    answer: 3
    solution: 5+5+1
    """
    coins_str = ",".join(str(c) for c in coins)
    solution_str = "+".join(str(c) for c in solution_coins) if answer != -1 else ""

    sample = (
        f"coins: {coins_str}\n"
        f"amount: {amount}\n\n"
        f"answer: {answer}\n"
        f"solution: {solution_str}\n"
    )
    return sample


def generate_min_coins_sample(
    **kwargs,
) -> str:
    """
    Convenience wrapper: returns ONE fully formatted sample text.
    Accepts the same kwargs as generate_min_coins_instance().
    """
    coins, amount, answer, solution = generate_min_coins_instance(**kwargs)
    return format_sample_text(coins, amount, answer, solution)


def generate_min_coins_dataset(
    n_samples: int,
    seed: Optional[int] = None,
    **kwargs,
) -> List[str]:
    """
    Generate a list of n_samples formatted problem+solution strings.
    """
    rng = random.Random(seed) if seed is not None else random
    samples = []
    for _ in range(n_samples):
        coins, amount, answer, solution = generate_min_coins_instance(rng=rng, **kwargs)
        samples.append(format_sample_text(coins, amount, answer, solution))
    return samples


# ---------- Parsing utils for RL verification ----------

_COINS_RE = re.compile(r"^coins\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_AMOUNT_RE = re.compile(r"^amount\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_RE = re.compile(
    r"^answer\s*:\s*([\-]?\d+)\s*$", re.IGNORECASE | re.MULTILINE
)
_SOLUTION_RE = re.compile(r"^solution\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _parse_coins_and_amount(problem_text: str) -> Tuple[List[int], int]:
    """
    Extract coins and amount from the problem text.
    Expects lines like:
      coins: 1,2,5
      amount: 11
    """
    coins_match = _COINS_RE.search(problem_text)
    amount_match = _AMOUNT_RE.search(problem_text)

    if not coins_match or not amount_match:
        raise ValueError("Could not parse coins and/or amount from problem text.")

    coins_str = coins_match.group(1)
    coins = []
    for token in coins_str.split(","):
        token = token.strip()
        if token:
            coins.append(int(token))

    amount = int(amount_match.group(1))
    return coins, amount


def _parse_answer_and_solution(
    predicted_text: str,
) -> Tuple[Optional[int], Optional[List[int]]]:
    """
    Extract answer (int) and solution list from the model's predicted text.

    Expected pattern (case-insensitive):
      answer: 3
      solution: 5+5+1

    Returns:
      (answer, solution_coins) where any of them can be None if parsing fails.
    """
    answer_match = _ANSWER_RE.search(predicted_text)
    solution_match = _SOLUTION_RE.search(predicted_text)

    pred_answer = None
    pred_solution_coins: Optional[List[int]] = None

    if answer_match:
        try:
            pred_answer = int(answer_match.group(1).strip())
        except ValueError:
            pred_answer = None

    if solution_match:
        sol_str = solution_match.group(1).strip()
        # Allow empty solutions (e.g. for impossible cases if you want that)
        if sol_str:
            # Split on '+' and strip whitespace
            parts = sol_str.split("+")
            coins = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                # Be robust: ignore non-numeric tokens
                if re.fullmatch(r"\d+", p):
                    coins.append(int(p))
                else:
                    # If any token is malformed, treat entire solution as invalid
                    coins = []
                    break
            pred_solution_coins = coins if coins else None
        else:
            pred_solution_coins = []
    return pred_answer, pred_solution_coins


# ---------- Reward function for RL ----------


def score_fn(
    pred_answer: float, true_answer: float, eps: float = 1e-8, k: float = 1.0
) -> float:
    scale = max(abs(true_answer), eps)
    rel_error = abs(pred_answer - true_answer) / scale
    # true: 3, pred=2 =>
    # scale = 3
    # rel_error = abs(2-3) / 3 = 0.333
    # e^(-1*0.333) = 0.71
    # Exponential decay of reward with relative error
    return math.exp(-k * rel_error)


def trace_score(
    pred_solution: Optional[List[int]],
    true_solution: List[int],
    amount: int,
    coins: List[int],
) -> float:
    """
    Score the predicted solution trace in [0.0, 1.0] by checking values
    one by one against a true optimal solution.

    Components (all in [0, 1]):
      - amount_component: 1 if sum(pred_solution) == amount, else 0
      - length_component: smooth penalty based on |len(pred) - len(true)|
      - element_component: fraction of positions i where pred[i] == true[i]

    Final trace_score is a weighted combination of these components.

    Validation:
      - If any coin in pred_solution is not in the valid coins set, return 0.0
      - If len(pred_solution) != len(true_solution), apply 0.5x penalty
    """
    if pred_solution is None:
        return 0.0

    # Validate all predicted coins are in the valid set
    coins_set = set(coins)
    if any(c not in coins_set for c in pred_solution):
        return 0.0

    # Special case: amount == 0
    if amount == 0:
        # For zero amount, best solution is usually an empty list
        return 1.0 if sum(pred_solution) == 0 and len(pred_solution) == 0 else 0.0

    if not true_solution:
        # Should not normally happen; be conservative
        raise ValueError("true solution is empty, check data")

    len_true = len(true_solution)
    len_pred = len(pred_solution)

    # 1) Amount correctness - HARD GATE
    # If sum doesn't match target amount, trace is invalid
    if sum(pred_solution) != amount:
        return 0.0

    # 2) Length similarity (soft)
    length_diff = abs(len_pred - len_true)
    # Normalized penalty by true length; clipped in [0, 1]
    length_component = max(0.0, 1.0 - length_diff / max(1, len_true))

    # 3) Element-wise matches (position-by-position)
    min_len = min(len_true, len_pred)
    matches = sum(1 for i in range(min_len) if pred_solution[i] == true_solution[i])
    element_component = matches / max(1, len_true)

    # Weights for combining length and element components
    # (amount is now a hard gate, not weighted)
    w_length = 0.33
    w_element = 0.67

    score = w_length * length_component + w_element * element_component

    # Apply length mismatch penalty (0.5x) if lengths don't match
    if len_pred != len_true:
        score *= 0.5

    # Safety clip into [0, 1]
    return max(0.0, min(1.0, score))


def verify_solution(
    problem_text: str,
    predicted_text: str,
    answer_weight: float = 0.1,
    k: float = 1.0,
    eps: float = 1e-8,
) -> float:
    """
    Compute a scalar reward in [0.0, 1.0] for the model's predicted text.

    Reward structure:

      Let:
        a  = exponential numeric score in (0, 1]
        t  = trace score in [0, 1]
        w  = answer_weight in (0, 1)
        1-w = trace_weight

      - If the numeric answer is wrong:
            reward = w * a
        (purely distance-based shaping, ignores the trace)

      - If the numeric answer is exactly correct:
            reward = w * 1.0 + (1 - w) * t
                  ∈ [w, 1.0]
    """
    if not (0.0 < answer_weight < 1.0):
        raise ValueError("answer_weight must be in (0, 1).")

    try:
        coins, amount = _parse_coins_and_amount(problem_text)
    except ValueError:
        # Problem text is malformed; safest to give zero reward
        return 0.0

    # true_answer: minimal number of coins
    # true_solution: one optimal decomposition
    true_answer, true_solution = solve_min_coins(coins, amount)

    pred_answer, pred_solution = _parse_answer_and_solution(predicted_text)

    # If we couldn't parse the answer, zero reward
    if pred_answer is None:
        return 0.0

    # Numeric answer component
    answer_score = score_fn(pred_answer, true_answer, eps=eps, k=k)
    trace_weight = 1.0 - answer_weight

    # Wrong numeric answer → only distance-based signal, ignore trace
    if pred_answer != true_answer:
        reward = answer_weight * answer_score
        return max(0.0, min(1.0, reward))

    # Numeric answer is correct
    # Base reward for getting the answer exactly right
    base_reward = answer_weight * 1.0  # answer_score == 1.0 here

    # Trace score with element-wise comparison
    tscore = trace_score(pred_solution, true_solution, amount, coins)

    reward = base_reward + trace_weight * tscore
    # Safety clip (should already be in [0, 1])
    return max(0.0, min(1.0, reward))
