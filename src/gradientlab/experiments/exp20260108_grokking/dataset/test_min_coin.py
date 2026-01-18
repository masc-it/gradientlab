"""Test the minimum coin change algorithm for correctness."""

from min_coin_change import solve_min_coin_change


def test_coin_change():
    """Test various coin change scenarios."""

    test_cases = [
        # (amount, coins, expected_min_count, description)
        (0, [1, 5, 10, 25], 0, "Zero amount"),
        (1, [1, 5, 10, 25], 1, "Single penny"),
        (5, [1, 5, 10, 25], 1, "Single nickel"),
        (10, [1, 5, 10, 25], 1, "Single dime"),
        (25, [1, 5, 10, 25], 1, "Single quarter"),
        (6, [1, 5, 10, 25], 2, "Nickel + penny"),
        (11, [1, 5, 10, 25], 2, "Dime + penny"),
        (30, [1, 5, 10, 25], 2, "Quarter + nickel"),
        (41, [1, 5, 10, 25], 4, "Quarter + dime + nickel + penny"),
        (99, [1, 5, 10, 25], 9, "3 quarters + 2 dimes + 4 pennies"),

        # Non-standard coins
        (6, [1, 3, 4], 2, "Non-standard: 3+3"),
        (10, [1, 3, 4], 3, "Non-standard: 4+3+3"),
        (15, [1, 3, 7, 15], 1, "Single 15"),
        (21, [1, 3, 7, 15], 3, "15+3+3"),

        # Edge case: cannot make exact change (no coin=1)
        (3, [5, 10], -1, "Impossible: 3 with coins [5,10]"),
        (7, [5, 10], -1, "Impossible: 7 with coins [5,10]"),

        # Large amounts
        (100, [1, 5, 10, 25, 50], 2, "Two half-dollars"),
        (99, [1, 5, 10, 25, 50], 8, "50+25+10+10+4pennies"),
        (250, [1, 5, 10, 25, 50], 5, "Five half-dollars"),
    ]

    print("Testing minimum coin change algorithm...\n")
    passed = 0
    failed = 0

    for amount, coins, expected_min, description in test_cases:
        min_coins, coins_used = solve_min_coin_change(amount, coins)

        # Check if the result matches expected
        if min_coins == expected_min:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1

        # Verify that coins sum to amount (if solution exists)
        if min_coins != -1:
            actual_sum = sum(coins_used)
            actual_count = len(coins_used)

            if actual_sum != amount:
                status = "✗ FAIL (sum mismatch)"
                failed += 1
                passed -= 1 if status == "✓ PASS" else 0

            if actual_count != min_coins:
                status = "✗ FAIL (count mismatch)"
                failed += 1
                passed -= 1 if status == "✓ PASS" else 0

            print(f"{status}: {description}")
            print(f"  Amount: {amount}, Coins: {coins}")
            print(f"  Expected: {expected_min}, Got: {min_coins}")
            print(f"  Coins used: {coins_used} (sum={actual_sum})")
        else:
            print(f"{status}: {description}")
            print(f"  Amount: {amount}, Coins: {coins}")
            print(f"  Expected: {expected_min}, Got: {min_coins} (impossible)")

        print()

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print(f"{'='*60}")

    if failed == 0:
        print("✓ All tests passed!")
        return True
    else:
        print(f"✗ {failed} test(s) failed")
        return False


if __name__ == "__main__":
    success = test_coin_change()
    exit(0 if success else 1)
