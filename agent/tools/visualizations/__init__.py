"""
Visualization tools package.

Re-exports all visualization functions.
Applies a consistent cyberpunk dark theme to all plots.
"""

import re
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CYBERPUNK THEME - Dark mode with neon accents
# =============================================================================

# Color palette
CYBER_COLORS = {
    'background': '#0d1117',      # Dark background
    'plot_bg': '#161b22',         # Slightly lighter plot area
    'text': '#e6edf3',            # Light text
    'text_secondary': '#8b949e',  # Muted text
    'grid': '#30363d',            # Subtle grid
    'neon_blue': '#00d4ff',       # Primary accent
    'neon_pink': '#ff0080',       # Secondary accent
    'neon_purple': '#b026ff',     # Tertiary accent
    'neon_green': '#00ff9f',      # Success/positive
    'neon_orange': '#ff6b35',     # Warning
}

# Neon color cycle for data series
CYBER_PALETTE = [
    '#00d4ff',  # Neon blue
    '#ff0080',  # Neon pink
    '#00ff9f',  # Neon green
    '#b026ff',  # Neon purple
    '#ff6b35',  # Neon orange
    '#ffdd00',  # Neon yellow
    '#00ffff',  # Cyan
    '#ff4d6d',  # Coral pink
]

# Apply dark style
plt.style.use('dark_background')

# Custom rcParams for cyberpunk look
plt.rcParams.update({
    # Figure
    'figure.figsize': (6, 4),
    'figure.dpi': 100,
    'figure.facecolor': CYBER_COLORS['background'],
    'figure.edgecolor': CYBER_COLORS['background'],
    
    # Axes
    'axes.facecolor': CYBER_COLORS['plot_bg'],
    'axes.edgecolor': CYBER_COLORS['grid'],
    'axes.labelcolor': '#ffffff',  # Bright white for axis labels
    'axes.titlecolor': CYBER_COLORS['neon_blue'],
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'medium',
    'axes.grid': True,
    'axes.axisbelow': True,
    'axes.prop_cycle': plt.cycler(color=CYBER_PALETTE),
    
    # Grid
    'grid.color': CYBER_COLORS['grid'],
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
    'grid.alpha': 0.6,
    
    # Ticks
    'xtick.color': '#c9d1d9',  # Brighter tick labels
    'ytick.color': '#c9d1d9',  # Brighter tick labels
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    
    # Legend
    'legend.facecolor': CYBER_COLORS['plot_bg'],
    'legend.edgecolor': CYBER_COLORS['grid'],
    'legend.fontsize': 8,
    'legend.labelcolor': CYBER_COLORS['text'],
    
    # Text
    'text.color': CYBER_COLORS['text'],
    'font.family': 'sans-serif',
    'font.size': 9,
    
    # Lines
    'lines.linewidth': 2,
    'lines.markersize': 6,
    
    # Patch (bars, etc.)
    'patch.edgecolor': CYBER_COLORS['grid'],
    
    # Savefig
    'savefig.facecolor': CYBER_COLORS['background'],
    'savefig.edgecolor': CYBER_COLORS['background'],
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

# Set seaborn to use our dark palette
sns.set_palette(CYBER_PALETTE)
sns.set_style("darkgrid", {
    'axes.facecolor': CYBER_COLORS['plot_bg'],
    'axes.edgecolor': CYBER_COLORS['grid'],
    'grid.color': CYBER_COLORS['grid'],
    'text.color': CYBER_COLORS['text'],
    'xtick.color': '#c9d1d9',
    'ytick.color': '#c9d1d9',
})

# Export colors for use in individual plot functions
THEME = CYBER_COLORS


def format_label(text: str) -> str:
    """
    Format column names into clean, readable labels with proper title case.
    
    Transformations:
    - snake_case -> Title Case ("daily_steps" -> "Daily Steps")
    - camelCase -> Title Case ("dailySteps" -> "Daily Steps")  
    - Handles common abbreviations (id, clv, pca, etc.)
    - Removes common prefixes (col_, feat_, etc.)
    - Applies proper title case (small words like "of", "the" stay lowercase)
    
    Args:
        text: The raw column name or label
        
    Returns:
        Formatted label string
    """
    if not text:
        return ""
    
    # Common abbreviations to handle specially
    abbreviations = {
        'id': 'ID',
        'clv': 'CLV',
        'pca': 'PCA',
        'pc1': 'PC1',
        'pc2': 'PC2', 
        'pc3': 'PC3',
        'iqr': 'IQR',
        'std': 'Std Dev',
        'avg': 'Average',
        'num': 'Number',
        'qty': 'Quantity',
        'amt': 'Amount',
        'pct': 'Percent',
        'yrs': 'Years',
        'hrs': 'Hours',
        'mins': 'Minutes',
        'lbs': 'lbs',
        'kg': 'kg',
        'usd': 'USD',
        'url': 'URL',
        'api': 'API',
        'roi': 'ROI',
        'kpi': 'KPI',
        'nps': 'NPS',
        'cac': 'CAC',
    }
    
    # Small words that should not be capitalized (unless first word)
    small_words = {
        'a', 'an', 'the',                          # articles
        'and', 'but', 'or', 'nor', 'for', 'yet',   # conjunctions
        'of', 'in', 'on', 'at', 'to', 'by',        # prepositions
        'as', 'if', 'per', 'via', 'vs', 'from',
        'with', 'into', 'onto', 'upon',
    }
    
    # Remove common prefixes
    prefixes_to_remove = ['col_', 'feat_', 'feature_', 'var_', 'field_']
    for prefix in prefixes_to_remove:
        if text.lower().startswith(prefix):
            text = text[len(prefix):]
    
    # Handle camelCase by inserting spaces before capitals
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    
    # Replace underscores and hyphens with spaces
    text = text.replace('_', ' ').replace('-', ' ')
    
    # Split into words and process each
    words = text.split()
    formatted_words = []
    
    for i, word in enumerate(words):
        lower_word = word.lower()
        
        if lower_word in abbreviations:
            formatted_words.append(abbreviations[lower_word])
        elif i > 0 and lower_word in small_words:
            # Keep small words lowercase (except first word)
            formatted_words.append(lower_word)
        else:
            # Title case the word
            formatted_words.append(word.capitalize())
    
    return ' '.join(formatted_words)

from .histogram import create_histogram
from .scatter import create_scatter_plot
from .heatmap import create_correlation_heatmap
from .boxplot import create_box_plot
from .bar_chart import create_bar_chart
from .pca_plot import create_pca_scatter, create_scree_plot
from .cluster_plot import create_cluster_scatter, create_elbow_plot, create_cluster_profile_chart

__all__ = [
    "format_label",
    "create_histogram",
    "create_scatter_plot",
    "create_correlation_heatmap",
    "create_box_plot",
    "create_bar_chart",
    # PCA visualizations
    "create_pca_scatter",
    "create_scree_plot",
    # Clustering visualizations
    "create_cluster_scatter",
    "create_elbow_plot",
    "create_cluster_profile_chart",
]
