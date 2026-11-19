"""
Multiple linear regression (OLS) with assumption diagnostics.

Fits the model with statsmodels and checks the assumptions that make its
inference trustworthy:
- Homoscedasticity (Breusch-Pagan) - if violated, the model is refit with
  HC3 heteroscedasticity-robust standard errors
- Linearity / functional form (Ramsey RESET)
- Multicollinearity (variance inflation factors)
- Normality of residuals (Jarque-Bera)
- Influential observations (Cook's distance)
"""

import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResultsWrapper
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import jarque_bera

from .multiple_testing import ALPHA, InferenceResult, PValue, format_p

MAX_CATEGORY_LEVELS = 20
# Rule of thumb: at least 10 observations per estimated coefficient
MIN_OBS_PER_PARAM = 10
VIF_MODERATE = 5
VIF_SEVERE = 10


def _term(column: str, categorical: bool) -> str:
    quoted = f"Q({column!r})"
    return f"C({quoted})" if categorical else quoted


def _fmt(x: float) -> str:
    """Readable number: fixed-point with separators for typical magnitudes, scientific otherwise."""
    if x == 0 or 1e-2 <= abs(x) < 1e9:
        return f"{x:,.2f}"
    return f"{x:.3g}"


def clean_term_name(name: str) -> str:
    """Turn patsy names like C(Q('dept'))[T.Sales] into dept[T.Sales]."""
    name = re.sub(r"Q\('(.+?)'\)", r"\1", name)
    return re.sub(r"C\((.+?)\)", r"\1", name)


def fit_ols(
    df: pd.DataFrame,
    target: str,
    predictors: list[str]
) -> tuple[InferenceResult, RegressionResultsWrapper | None]:
    """
    Fit an OLS regression of target on predictors and run assumption diagnostics.

    Non-numeric predictors are dummy-coded, with the first level (alphabetically)
    as the reference category.

    Args:
        df: The dataframe
        target: Numeric outcome column
        predictors: Predictor columns (numeric or categorical)

    Returns:
        Tuple of (InferenceResult with coefficients, fit statistics and diagnostics,
        and one p-value per non-intercept coefficient; the fitted statsmodels
        results, or None if the model could not be fit)
    """
    if target not in df.columns:
        return InferenceResult(f"Error: Column '{target}' not found."), None
    if not pd.api.types.is_numeric_dtype(df[target]):
        return InferenceResult(f"Error: Target '{target}' is not numeric. Linear regression needs a numeric outcome."), None
    if not predictors:
        return InferenceResult("Error: Provide at least one predictor."), None
    missing = [p for p in predictors if p not in df.columns]
    if missing:
        return InferenceResult(f"Error: Columns not found: {missing}"), None
    if target in predictors:
        return InferenceResult("Error: The target cannot also be a predictor."), None

    categorical = {
        p for p in predictors
        if not pd.api.types.is_numeric_dtype(df[p]) or pd.api.types.is_bool_dtype(df[p])
    }
    for p in categorical:
        n_levels = df[p].nunique()
        if n_levels > MAX_CATEGORY_LEVELS:
            return InferenceResult(
                f"Error: Categorical predictor '{p}' has {n_levels} levels (max {MAX_CATEGORY_LEVELS}). "
                "Is it an ID column? Drop it or group rare levels."
            ), None

    data = df[[target] + predictors].dropna()
    n_dropped = len(df) - len(data)
    formula = f"{_term(target, False)} ~ " + " + ".join(_term(p, p in categorical) for p in predictors)

    model = smf.ols(formula, data=data)
    n, n_params = model.exog.shape
    if n <= n_params + 1:
        return InferenceResult(f"Error: {n} complete rows is too few to estimate {n_params} coefficients."), None

    fit = model.fit()

    # --- Diagnostics (computed on the classical fit) ---
    bp_pvalue = float(het_breuschpagan(fit.resid, model.exog)[1])
    reset_pvalue = float(linear_reset(fit, power=2, use_f=True).pvalue)
    jb_stat, jb_pvalue, resid_skew, resid_kurtosis = jarque_bera(fit.resid)
    cooks_d = fit.get_influence().cooks_distance[0]
    influential = np.where(cooks_d > 4 / n)[0]

    exog_names = model.exog_names
    vifs = {}
    if n_params > 2:  # VIF needs at least two predictors besides the intercept
        for i, name in enumerate(exog_names):
            if name != "Intercept":
                vifs[clean_term_name(name)] = float(variance_inflation_factor(model.exog, i))

    robust = bp_pvalue < ALPHA
    report = model.fit(cov_type="HC3") if robust else fit
    conf_int = report.conf_int(alpha=ALPHA)

    # --- Report ---
    lines = [
        f"OLS REGRESSION: {target} ~ {' + '.join(predictors)}",
        f"Observations: {n:,}" + (f" ({n_dropped:,} rows with missing values dropped)" if n_dropped else ""),
        f"R-squared: {fit.rsquared:.3f} | Adjusted R-squared: {fit.rsquared_adj:.3f}",
        f"Overall F-test: {format_p(report.f_pvalue)}",
        f"Standard errors: {'HC3 heteroscedasticity-robust' if robust else 'classical (homoscedasticity not rejected)'}",
    ]
    if categorical:
        refs = ", ".join(f"{p}='{sorted(data[p].astype(str).unique())[0]}'" for p in sorted(categorical))
        lines.append(f"Reference categories: {refs}")

    lines += ["", "COEFFICIENTS (95% CI):"]
    p_values = []
    for name in exog_names:
        label = clean_term_name(name)
        est, se, p = report.params[name], report.bse[name], report.pvalues[name]
        low, high = conf_int.loc[name]
        flag = " *" if p < ALPHA else ""
        lines.append(f"  {label}: {_fmt(est)} (SE {_fmt(se)}, CI [{_fmt(low)}, {_fmt(high)}], {format_p(p)}){flag}")
        if name != "Intercept":
            p_values.append(PValue("OLS coefficient", f"{label} predicts {target} (adjusted for other predictors)", float(p)))
    lines.append("  (* = significant at alpha=0.05 before multiple-testing correction)")

    lines += ["", "ASSUMPTION DIAGNOSTICS:"]
    lines.append(
        f"  Homoscedasticity (Breusch-Pagan): p={bp_pvalue:.3g} -> "
        + ("heteroscedastic; robust HC3 standard errors used" if robust else "OK")
    )
    lines.append(
        f"  Functional form (Ramsey RESET): p={reset_pvalue:.3g} -> "
        + ("possible nonlinearity or missing terms; check the residual plot" if reset_pvalue < ALPHA else "OK")
    )
    lines.append(
        f"  Residual normality (Jarque-Bera): p={jb_pvalue:.3g}, skew={resid_skew:.2f}, kurtosis={resid_kurtosis:.2f} -> "
        + ("non-normal residuals" + ("; with this sample size inference on coefficients is still approximately valid" if n >= 100 else "; treat p-values with caution at this sample size")
           if jb_pvalue < ALPHA else "OK")
    )
    if vifs:
        worst = max(vifs, key=vifs.get)
        level = "severe" if vifs[worst] > VIF_SEVERE else "moderate" if vifs[worst] > VIF_MODERATE else "OK"
        lines.append(f"  Multicollinearity: max VIF = {vifs[worst]:.2f} ({worst}) -> {level}")
        high = [f"{k} ({v:.1f})" for k, v in vifs.items() if v > VIF_MODERATE]
        if high:
            lines.append(f"    Inflated: {', '.join(high)} - their individual coefficients are unstable")
    lines.append(
        f"  Influential points (Cook's D > 4/n): {len(influential)}"
        + (f"; largest D = {cooks_d.max():.3f} at row {data.index[int(cooks_d.argmax())]}" if len(influential) else "")
    )
    if n < MIN_OBS_PER_PARAM * n_params:
        lines.append(
            f"  Sample size: {n} rows for {n_params} coefficients is below the ~{MIN_OBS_PER_PARAM} per coefficient "
            "rule of thumb - estimates may be unstable or overfit"
        )

    lines += [
        "",
        "Interpretation note: coefficients describe associations holding the other predictors constant. "
        "They are not causal effects unless the data come from a randomized design.",
    ]

    return InferenceResult("\n".join(lines), p_values), fit
