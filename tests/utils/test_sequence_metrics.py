from src.mslm.utils.sequence_metrics import (
    legacy_prefix_exact,
    levenshtein,
    strict_exact,
    token_edit_similarity,
    token_error_rate,
)


def test_strict_exact_requires_count_match():
    assert strict_exact([1, 2], [1, 2]) is True
    assert strict_exact([1, 2, 3], [1, 2]) is False  # overprediction
    assert strict_exact([1], [1, 2]) is False  # underprediction
    assert strict_exact([1, 3], [1, 2]) is False  # wrong token


def test_legacy_prefix_exact_ignores_extra_predicted_tokens():
    # Historical bug this plan fixes: a 3-token overprediction whose prefix
    # matches the 2-token target still counted as "exact".
    assert legacy_prefix_exact([1, 2, 9], [1, 2]) is True
    assert strict_exact([1, 2, 9], [1, 2]) is False


def test_token_error_rate_and_similarity():
    assert token_error_rate([1, 2], [1, 2]) == 0.0
    assert token_edit_similarity(0.0) == 1.0
    ter = token_error_rate([1, 3], [1, 2])
    assert ter == 0.5
    assert token_edit_similarity(ter) == 0.5


def test_token_error_rate_empty_target():
    assert token_error_rate([], []) == 0.0
    assert token_error_rate([1], []) == 1.0


def test_levenshtein_basic():
    assert levenshtein([1, 2, 3], [1, 2, 3]) == 0
    assert levenshtein([1, 2, 3], [1, 2]) == 1
    assert levenshtein([], [1, 2]) == 2


if __name__ == "__main__":
    test_strict_exact_requires_count_match()
    test_legacy_prefix_exact_ignores_extra_predicted_tokens()
    test_token_error_rate_and_similarity()
    test_token_error_rate_empty_target()
    test_levenshtein_basic()
    print("ok")
