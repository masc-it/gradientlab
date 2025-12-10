"""
Stress tests for the reward function in rl_env.py.
Also explores reward hacking vulnerabilities.

Run with: uv run python -m gradientlab.experiments.exp20251208_qwen_rl_coin_change.test_reward
"""

from gradientlab.experiments.exp20251208_qwen_rl_coin_change.rl_env import (
    verify_solution,
    trace_score,
    score_fn,
    solve_min_coins,
    _parse_answer_and_solution,
)


class TestVerifySolutionBasic:
    """Basic correctness tests."""

    def test_perfect_answer(self):
        """Perfect prediction should get reward ~1.0."""
        problem = "coins: 1,2,5\namount: 11"
        # Optimal: 5+5+1 = 3 coins
        prediction = "answer: 3\nsolution: 5+5+1"
        reward = verify_solution(problem, prediction)
        assert reward >= 0.95, f"Perfect answer should get high reward, got {reward}"

    def test_wrong_answer_gets_partial_reward(self):
        """Wrong numeric answer should still get some shaped reward."""
        problem = "coins: 1,2,5\namount: 11"
        # True answer is 3, predicting 4
        prediction = "answer: 4\nsolution: 5+5+1"
        reward = verify_solution(problem, prediction)
        assert 0 < reward < 0.7, f"Wrong answer should get partial reward, got {reward}"

    def test_no_answer_parsed_returns_zero(self):
        """If we can't parse the answer, return 0."""
        problem = "coins: 1,2,5\namount: 11"
        prediction = "I think the answer is three coins"
        reward = verify_solution(problem, prediction)
        assert reward == 0.0

    def test_malformed_problem_returns_zero(self):
        """If problem text is malformed, return 0."""
        problem = "this is not a valid problem"
        prediction = "answer: 3\nsolution: 5+5+1"
        reward = verify_solution(problem, prediction)
        assert reward == 0.0


class TestRewardHacking:
    """
    Test cases exploring potential reward hacking vulnerabilities.

    REWARD HACKING: Ways a model could exploit the reward function
    to get high rewards without actually solving the problem correctly.
    """

    def test_hack_correct_answer_wrong_solution(self):
        """
        HACK #1: Correct answer number, but solution doesn't use valid coins.

        If the model learns the numeric answer but provides a bogus solution
        that happens to sum to the correct amount, does it get high reward?
        """
        problem = "coins: 1,2,5\namount: 11"
        # True: answer=3, solution=5+5+1
        # Hack: correct answer, but solution uses coins NOT in the set (3+3+3+2)
        prediction = "answer: 3\nsolution: 3+3+3+2"  # 3 is not a valid coin!
        reward = verify_solution(problem, prediction)
        # This SHOULD be penalized, but trace_score doesn't check coin validity!
        print(f"Hack #1 (invalid coins in solution): reward={reward}")
        # This is a BUG if reward > 0.7 (answer_weight)

    def test_hack_correct_answer_wrong_sum(self):
        """
        HACK #2: Correct answer, but solution doesn't sum to amount.

        Model gives correct count but solution coins don't add up.
        """
        problem = "coins: 1,2,5\namount: 11"
        # True: answer=3, solution=5+5+1
        # Hack: answer=3 but solution=5+5+5 (sums to 15, not 11)
        prediction = "answer: 3\nsolution: 5+5+5"
        reward = verify_solution(problem, prediction)
        print(f"Hack #2 (wrong sum): reward={reward}")
        # Should be penalized by amount_component=0 in trace_score

    def test_hack_correct_answer_solution_count_mismatch(self):
        """
        HACK #3: Correct answer, solution has wrong number of coins.

        Model says answer=3 but solution has 4 coins.
        """
        problem = "coins: 1,2,5\namount: 11"
        # True: answer=3, solution=5+5+1
        # Hack: answer=3 but solution=5+2+2+2 (4 coins, sums to 11)
        prediction = "answer: 3\nsolution: 5+2+2+2"
        reward = verify_solution(problem, prediction)
        print(f"Hack #3 (count mismatch): reward={reward}")
        # The trace_score checks sum but doesn't verify len(solution) == answer

    def test_hack_empty_solution(self):
        """
        HACK #4: Correct answer with empty or missing solution.

        Model learns to output the number but never the trace.
        """
        problem = "coins: 1,2,5\namount: 11"
        prediction = "answer: 3\nsolution: "
        reward = verify_solution(problem, prediction)
        print(f"Hack #4 (empty solution): reward={reward}")
        # Gets answer_weight * 1.0 = 0.7 base reward

    def test_hack_negative_answer(self):
        """
        HACK #5: Can model exploit with negative answers?
        """
        problem = "coins: 1,2,5\namount: 11"
        prediction = "answer: -3\nsolution: 5+5+1"
        reward = verify_solution(problem, prediction)
        print(f"Hack #5 (negative answer): reward={reward}")

    def test_hack_zero_answer_nonzero_amount(self):
        """
        HACK #6: Answer=0 for a problem that requires coins.
        """
        problem = "coins: 1,2,5\namount: 11"
        prediction = "answer: 0\nsolution: "
        reward = verify_solution(problem, prediction)
        print(f"Hack #6 (zero answer): reward={reward}")

    def test_hack_very_large_answer(self):
        """
        HACK #7: Absurdly large answer.
        score_fn uses relative error, so this should be heavily penalized.
        """
        problem = "coins: 1,2,5\namount: 11"
        prediction = "answer: 1000000\nsolution: 1+1+1"
        reward = verify_solution(problem, prediction)
        print(f"Hack #7 (huge answer): reward={reward}")

    def test_hack_solution_order_matters(self):
        """
        HACK #8: Same coins different order - does element_component care?

        If true_solution=[5,5,1] and pred=[1,5,5], positions don't match.
        """
        problem = "coins: 1,2,5\namount: 11"
        # True solution is likely [1,5,5] or [5,5,1] depending on solve order
        prediction1 = "answer: 3\nsolution: 1+5+5"
        prediction2 = "answer: 3\nsolution: 5+5+1"
        reward1 = verify_solution(problem, prediction1)
        reward2 = verify_solution(problem, prediction2)
        print(f"Hack #8 (order matters): [1,5,5]={reward1}, [5,5,1]={reward2}")
        # This could create inconsistent rewards for equivalent solutions!


class TestTraceScoreVulnerabilities:
    """Deeper inspection of trace_score edge cases."""

    def test_trace_score_no_coin_validation(self):
        """
        trace_score now validates that predicted coins are in the valid set.
        Invalid coins should return 0.0.
        """
        # Problem has coins [1,2,5], amount=11, answer=3, solution=[5,5,1]
        # But we use coins not in set
        pred_solution = [4, 4, 3]  # 4 and 3 are NOT valid coins
        true_solution = [5, 5, 1]
        amount = 11
        coins = [1, 2, 5]

        score = trace_score(pred_solution, true_solution, amount, coins)
        print(f"Invalid coins trace_score: {score}")
        # With fix: Should be 0.0 because coins 4 and 3 are not in [1,2,5]
        assert score == 0.0, f"Invalid coins should get 0.0, got {score}"

    def test_trace_score_alternative_valid_solution(self):
        """
        What if there are multiple optimal solutions?
        The model finds a different valid one but gets penalized.
        """
        # coins=[1,2,5], amount=6
        # Possible optimal solutions with 2 coins: [5,1], [2,2,2]... wait [2,2,2] is 3
        # Actually [1,5] and [2,2,2] are both valid for amount=6
        # But [1,5] uses 2 coins, [2,2,2] uses 3. So [1,5] is optimal.
        # Let's use a case where there are multiple optimal paths

        # coins=[1,3,4], amount=6
        # Solutions: [3,3] = 2 coins, [4,1,1] = 3 coins
        # Optimal is [3,3]
        coins = [1, 3, 4]
        amount = 6
        true_ans, true_sol = solve_min_coins(coins, amount)
        print(f"Optimal for coins={coins}, amount={amount}: {true_ans}, {true_sol}")

        # If model finds [3,3] but true_sol is stored as [3,3] in different order
        # Actually solve_min_coins is deterministic, so not an issue here

    def test_trace_score_partial_match(self):
        """
        Partial element matches - how much reward?
        """
        true_solution = [5, 5, 1]
        coins = [1, 2, 5]

        # Exact match
        assert trace_score([5, 5, 1], true_solution, 11, coins) >= 0.95

        # One element wrong
        score1 = trace_score([5, 5, 2], true_solution, 11, coins)  # Sum=12, wrong
        score2 = trace_score([5, 2, 2], true_solution, 11, coins)  # Sum=9, wrong sum but valid coins
        print(f"One wrong (bad sum): {score1}")
        print(f"Two wrong (bad sum): {score2}")


class TestScoreFnEdgeCases:
    """Test the numeric score function."""

    def test_score_fn_exact(self):
        """Exact match should give 1.0."""
        assert score_fn(3, 3) == 1.0

    def test_score_fn_close(self):
        """Close values should give high score."""
        s = score_fn(4, 3)
        assert 0.5 < s < 1.0, f"Expected ~0.71, got {s}"

    def test_score_fn_true_is_zero(self):
        """
        Edge case: true_answer is 0.
        scale = max(0, eps) = eps
        If pred=0 too, score=1.0
        If pred=1, rel_error=1/eps which is huge -> score ~0
        """
        s_exact = score_fn(0, 0)
        s_off = score_fn(1, 0)
        print(f"score_fn(0,0)={s_exact}, score_fn(1,0)={s_off}")
        assert s_exact == 1.0
        # s_off should be very small due to huge relative error

    def test_score_fn_negative_true(self):
        """True answer is negative (e.g., -1 for impossible)."""
        # score_fn uses abs(true_answer) for scale
        s = score_fn(-1, -1)
        assert s == 1.0

        # pred=0, true=-1: |0-(-1)|/1 = 1, exp(-1) ~= 0.37
        s2 = score_fn(0, -1)
        print(f"score_fn(0, -1)={s2}")


class TestParsingRobustness:
    """Test how parsing handles edge cases."""

    def test_parse_multiline_answer(self):
        """Model outputs answer on multiple lines."""
        text = """
        Let me think...
        answer: 3
        The solution is:
        solution: 5+5+1
        """
        ans, sol = _parse_answer_and_solution(text)
        assert ans == 3
        assert sol == [5, 5, 1]

    def test_parse_extra_whitespace(self):
        """Extra spaces around values."""
        text = "answer:   3  \nsolution:  5 + 5 + 1  "
        ans, sol = _parse_answer_and_solution(text)
        assert ans == 3
        assert sol == [5, 5, 1]

    def test_parse_case_insensitive(self):
        """ANSWER vs answer."""
        text = "ANSWER: 3\nSOLUTION: 5+5+1"
        ans, sol = _parse_answer_and_solution(text)
        assert ans == 3
        assert sol == [5, 5, 1]

    def test_parse_malformed_solution(self):
        """Solution with non-numeric parts."""
        text = "answer: 3\nsolution: 5+five+1"
        ans, sol = _parse_answer_and_solution(text)
        assert ans == 3
        assert sol is None  # Parsing should fail for malformed

    def test_parse_float_answer(self):
        """What if model outputs float like 3.0?"""
        text = "answer: 3.0\nsolution: 5+5+1"
        ans, sol = _parse_answer_and_solution(text)
        # Regex expects integer pattern, so this should fail
        print(f"Float answer parse: ans={ans}, sol={sol}")


class TestRewardHackingSummary:
    """
    Summary of reward hacking vulnerabilities found.
    Run with: pytest -v -s test_reward.py::TestRewardHackingSummary
    """

    def test_summarize_hacks(self):
        """Print summary of all hacking vectors."""
        problem = "coins: 1,2,5\namount: 11"
        # True: answer=3, solution=[5,5,1] (or [1,5,5])

        hacks = {
            "Perfect solution": ("answer: 3\nsolution: 5+5+1", "Should be ~1.0"),
            "Invalid coins": ("answer: 3\nsolution: 4+4+3", "BUG: Should be <0.7"),
            "Wrong sum": ("answer: 3\nsolution: 5+5+5", "OK if <0.9"),
            "Length mismatch": ("answer: 3\nsolution: 5+2+2+2", "BUG if >0.8"),
            "Empty solution": ("answer: 3\nsolution: ", "Gets 0.7 (answer_weight)"),
            "No solution line": ("answer: 3", "Gets 0.7 (answer_weight)"),
            "Answer off by 1": ("answer: 4\nsolution: 5+5+1", "Shaped reward"),
            "Answer way off": ("answer: 100\nsolution: 5+5+1", "Should be ~0"),
        }

        print("\n" + "="*60)
        print("REWARD HACKING VULNERABILITY SUMMARY")
        print("="*60)

        for name, (pred, expected) in hacks.items():
            reward = verify_solution(problem, pred)
            print(f"\n{name}:")
            print(f"  Prediction: {pred!r}")
            print(f"  Reward: {reward:.4f}")
            print(f"  Expected: {expected}")


def run_all_tests():
    """Run all tests without pytest."""
    print("\n" + "="*70)
    print("RUNNING ALL REWARD FUNCTION TESTS")
    print("="*70)

    # Basic tests
    print("\n--- TestVerifySolutionBasic ---")
    basic = TestVerifySolutionBasic()
    try:
        basic.test_perfect_answer()
        print("✓ test_perfect_answer")
    except AssertionError as e:
        print(f"✗ test_perfect_answer: {e}")

    try:
        basic.test_wrong_answer_gets_partial_reward()
        print("✓ test_wrong_answer_gets_partial_reward")
    except AssertionError as e:
        print(f"✗ test_wrong_answer_gets_partial_reward: {e}")

    try:
        basic.test_no_answer_parsed_returns_zero()
        print("✓ test_no_answer_parsed_returns_zero")
    except AssertionError as e:
        print(f"✗ test_no_answer_parsed_returns_zero: {e}")

    try:
        basic.test_malformed_problem_returns_zero()
        print("✓ test_malformed_problem_returns_zero")
    except AssertionError as e:
        print(f"✗ test_malformed_problem_returns_zero: {e}")

    # Reward hacking tests
    print("\n--- TestRewardHacking ---")
    hacking = TestRewardHacking()
    hacking.test_hack_correct_answer_wrong_solution()
    hacking.test_hack_correct_answer_wrong_sum()
    hacking.test_hack_correct_answer_solution_count_mismatch()
    hacking.test_hack_empty_solution()
    hacking.test_hack_negative_answer()
    hacking.test_hack_zero_answer_nonzero_amount()
    hacking.test_hack_very_large_answer()
    hacking.test_hack_solution_order_matters()

    # Trace score tests
    print("\n--- TestTraceScoreVulnerabilities ---")
    trace = TestTraceScoreVulnerabilities()
    trace.test_trace_score_no_coin_validation()
    trace.test_trace_score_alternative_valid_solution()
    trace.test_trace_score_partial_match()

    # Score function tests
    print("\n--- TestScoreFnEdgeCases ---")
    scorefn = TestScoreFnEdgeCases()
    try:
        scorefn.test_score_fn_exact()
        print("✓ test_score_fn_exact")
    except AssertionError as e:
        print(f"✗ test_score_fn_exact: {e}")

    try:
        scorefn.test_score_fn_close()
        print("✓ test_score_fn_close")
    except AssertionError as e:
        print(f"✗ test_score_fn_close: {e}")

    scorefn.test_score_fn_true_is_zero()
    scorefn.test_score_fn_negative_true()

    # Parsing tests
    print("\n--- TestParsingRobustness ---")
    parsing = TestParsingRobustness()
    try:
        parsing.test_parse_multiline_answer()
        print("✓ test_parse_multiline_answer")
    except AssertionError as e:
        print(f"✗ test_parse_multiline_answer: {e}")

    try:
        parsing.test_parse_extra_whitespace()
        print("✓ test_parse_extra_whitespace")
    except AssertionError as e:
        print(f"✗ test_parse_extra_whitespace: {e}")

    try:
        parsing.test_parse_case_insensitive()
        print("✓ test_parse_case_insensitive")
    except AssertionError as e:
        print(f"✗ test_parse_case_insensitive: {e}")

    try:
        parsing.test_parse_malformed_solution()
        print("✓ test_parse_malformed_solution")
    except AssertionError as e:
        print(f"✗ test_parse_malformed_solution: {e}")

    parsing.test_parse_float_answer()

    # Summary
    print("\n")
    summary = TestRewardHackingSummary()
    summary.test_summarize_hacks()


if __name__ == "__main__":
    run_all_tests()
