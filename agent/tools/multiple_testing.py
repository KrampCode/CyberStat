"""
Multiple-testing support.

An agent that answers questions by running tests will, over a session, run many of
them - and at alpha = 0.05, one in twenty true null hypotheses comes back
"significant" by chance. Inferential tools therefore return the p-values they
produce alongside their text, the agent registers them with the session, and every
result is reported with a Benjamini-Hochberg adjusted p-value that controls the
false discovery rate across all tests run so far.
"""

from dataclasses import dataclass, field
from typing import Protocol

from statsmodels.stats.multitest import multipletests

ALPHA = 0.05


@dataclass
class PValue:
    """A single hypothesis test outcome to register with the session."""
    test: str
    hypothesis: str
    p_value: float


@dataclass
class InferenceResult:
    """Formatted output of an inferential tool plus the p-values it produced."""
    text: str
    p_values: list[PValue] = field(default_factory=list)


def format_p(p: float) -> str:
    """Format 'p = x', showing a bound instead of 0 when the p-value underflows."""
    return "p < 1e-16" if p < 1e-16 else f"p = {p:.4g}"


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """BH-adjusted p-values (controls the false discovery rate)."""
    if not p_values:
        return []
    return [float(p) for p in multipletests(p_values, method="fdr_bh")[1]]


class RecordedTest(Protocol):
    test: str
    hypothesis: str
    p_value: float


def summarize_session_tests(tests: list[RecordedTest], alpha: float = ALPHA) -> str:
    """
    Report every test run this session with raw and BH-adjusted p-values.
    
    Adjusted p-values depend on the full set of tests, so they can change as more
    tests are run - a result that was significant early in a session may not survive
    correction later.
    """
    if not tests:
        return "No hypothesis tests have been run this session."
    
    adjusted = benjamini_hochberg([t.p_value for t in tests])
    n_raw = sum(t.p_value < alpha for t in tests)
    n_adj = sum(p < alpha for p in adjusted)
    
    lines = [
        f"MULTIPLE TESTING SUMMARY: {len(tests)} tests this session, Benjamini-Hochberg FDR at {alpha}",
        f"Significant before correction: {n_raw} | after correction: {n_adj}",
        "",
    ]
    for i, (t, p_adj) in enumerate(zip(tests, adjusted), 1):
        status = "significant" if p_adj < alpha else "not significant"
        lost = " (lost significance after correction)" if t.p_value < alpha <= p_adj else ""
        lines.append(f"  {i}. [{t.test}] {t.hypothesis}: {format_p(t.p_value)}, adjusted {format_p(p_adj)} -> {status}{lost}")
    return "\n".join(lines)


def holm(p_values: list[float]) -> list[float]:
    """Holm-adjusted p-values (controls the family-wise error rate)."""
    if not p_values:
        return []
    return [float(p) for p in multipletests(p_values, method="holm")[1]]
