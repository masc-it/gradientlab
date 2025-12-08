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

_COINS_RE = re.compile(r"^\s*coins\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_AMOUNT_RE = re.compile(r"^\s*amount\s*:\s*(\d+)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_RE = re.compile(
    r"^\s*answer\s*:\s*([\-]?\d+)\s*$", re.IGNORECASE | re.MULTILINE
)
_SOLUTION_RE = re.compile(r"^\s*solution\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)


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


def verify_solution(problem_text: str, predicted_text: str) -> float:
    """
    Compute a scalar reward in [0.0, 1.0] for the model's predicted text.

    Logic:
      1. Parse coins & amount from problem_text.
      2. Compute the *true* optimal answer & one optimal solution via DP.
      3. Parse predicted answer & solution from predicted_text.
      4. Give reward = 1.0 iff:
         - predicted_answer == true_answer, and
         - if true_answer != -1:
             - predicted_solution is parseable,
             - sum(predicted_solution) == amount,
             - len(predicted_solution) == true_answer
         - if true_answer == -1:
             - we only require predicted_answer == -1

      Otherwise, reward = 0.0.

    You can easily change this to shaped rewards if you like.
    """
    try:
        coins, amount = _parse_coins_and_amount(problem_text)
    except ValueError:
        # Problem text is malformed; safest to give zero reward
        return 0.0

    true_answer, true_solution = solve_min_coins(coins, amount)
    pred_answer, pred_solution = _parse_answer_and_solution(predicted_text)

    # If we couldn't parse the answer, zero reward
    if pred_answer is None:
        return 0.0

    # If the problem has no solution, we only care that the model says -1.
    if true_answer == -1:
        return 1.0 if pred_answer == -1 else 0.0

    # Problem is solvable: check predicted answer
    if pred_answer != true_answer:
        return 0.0

    # Answer is correct; now check the solution string
    if pred_solution is None:
        # Could not parse solution → zero reward
        return 0.0

    # Check correct sum and count
    if sum(pred_solution) != amount:
        return 0.1
    if len(pred_solution) != true_answer:
        return 0.1

    # Everything matches (correct min count, correct decomposition)
    return 1.0
