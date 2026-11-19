"""Tests for multiple-testing correction."""

import pytest

from agent.tools.multiple_testing import PValue, benjamini_hochberg, format_p, holm, summarize_session_tests


def test_benjamini_hochberg_by_hand():
    # Sorted: .005, .01, .03, .04 -> p * m / rank = .02, .02, .04, .04 (already monotone)
    assert benjamini_hochberg([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.02, 0.04, 0.04, 0.02])


def test_holm_by_hand():
    # Sorted: .005, .01, .03, .04 -> p * (m - rank + 1) = .02, .03, .06, .06 (cumulative max)
    assert holm([0.01, 0.04, 0.03, 0.005]) == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_empty():
    assert benjamini_hochberg([]) == [] and holm([]) == []


def test_summary_flags_results_that_lose_significance():
    tests = [
        PValue("t", "strong effect", 0.001),
        PValue("t", "marginal effect", 0.04),
        PValue("t", "null 1", 0.5),
        PValue("t", "null 2", 0.7),
    ]
    summary = summarize_session_tests(tests)
    assert "Significant before correction: 2 | after correction: 1" in summary
    assert "marginal effect: p = 0.04, adjusted p = 0.08 -> not significant (lost significance after correction)" in summary


def test_summary_with_no_tests():
    assert "No hypothesis tests" in summarize_session_tests([])


def test_format_p_underflow():
    assert format_p(0.0) == "p < 1e-16"
    assert format_p(0.0123) == "p = 0.0123"
