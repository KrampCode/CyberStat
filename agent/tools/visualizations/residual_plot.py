"""Regression diagnostic plots: residuals vs fitted and normal Q-Q."""

import io

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scipy_stats
from statsmodels.nonparametric.smoothers_lowess import lowess

from . import THEME, format_label


def create_residual_plots(
    fitted: np.ndarray,
    residuals: np.ndarray,
    target: str
) -> tuple[bytes, str, dict]:
    """
    Create side-by-side regression diagnostics.

    Left: residuals vs fitted values with a LOWESS smooth - a curved smooth suggests
    nonlinearity, a funnel shape suggests heteroscedasticity.
    Right: normal Q-Q plot of standardized residuals - points off the line in the
    tails suggest non-normal errors or outliers.

    Args:
        fitted: Fitted values from the model
        residuals: Residuals from the model
        target: Name of the outcome variable (for the title)

    Returns:
        Tuple of (image_bytes, description, statistics_dict)
    """
    fitted = np.asarray(fitted, dtype=float)
    residuals = np.asarray(residuals, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))

    # Residuals vs fitted
    ax1.scatter(fitted, residuals, s=14, alpha=0.6, color=THEME['neon_blue'], edgecolors='none')
    ax1.axhline(0, color=THEME['text_secondary'], linestyle='--', linewidth=1)
    smooth = lowess(residuals, fitted, frac=0.6)
    ax1.plot(smooth[:, 0], smooth[:, 1], color=THEME['neon_pink'], linewidth=2, label='LOWESS')
    ax1.set_xlabel('Fitted values', color='#ffffff')
    ax1.set_ylabel('Residuals', color='#ffffff')
    ax1.set_title('Residuals vs Fitted')
    ax1.legend(loc='upper right', framealpha=0.9)

    # Normal Q-Q of standardized residuals
    standardized = (residuals - residuals.mean()) / residuals.std(ddof=1)
    (theoretical, ordered), (slope, intercept, _) = scipy_stats.probplot(standardized, dist='norm')
    ax2.scatter(theoretical, ordered, s=14, alpha=0.6, color=THEME['neon_blue'], edgecolors='none')
    ax2.plot(theoretical, slope * theoretical + intercept, color=THEME['neon_pink'], linewidth=2)
    ax2.set_xlabel('Theoretical quantiles', color='#ffffff')
    ax2.set_ylabel('Standardized residuals', color='#ffffff')
    ax2.set_title('Normal Q-Q')

    fig.suptitle(f'Regression Diagnostics: {format_label(target)}', color=THEME['neon_blue'], fontweight='bold')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', facecolor=THEME['background'])
    buf.seek(0)
    plt.close(fig)

    stats = {
        "n_points": int(len(residuals)),
        "n_large_residuals": int((np.abs(standardized) > 3).sum()),
    }

    return buf.getvalue(), f"Regression diagnostics for {target}", stats
