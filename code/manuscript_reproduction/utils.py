"""
Shared utilities for FlowGAN analysis scripts.
Contains common functions for statistics, plotting, and data processing.
"""

import os

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.multitest as smm
from skimage.metrics import structural_similarity as ssim
from typing import List, Dict, Tuple, Optional

# ============================================================================
# Plotting parameters
# ============================================================================
BOXPLOT_PARAMS = {'width': 0.4, 'linewidth': 2, 'palette': 'Set2'}
COLORS = ['#66c2a5', '#fc8d62']

# ============================================================================
# Region lists
# ============================================================================
# MCI-relevant regions for analysis (AD-related regions)
MCI_REGIONS = [
    'medialorbitofrontal',
    'posteriorcingulate',
    'transversetemporal',
    'CC_Posterior',
    'caudalmiddlefrontal',
    'CC_Anterior',
    'lingual',
    'CC_Central',
    'superiorparietal',
    'CC_Mid_Anterior',
    'parsorbitalis',
    'frontalpole',
    'temporalpole',
    'cuneus',
    'parahippocampal',
    'rostralanteriorcingulate',
    'pericalcarine',
    'rostralmiddlefrontal',
    'isthmuscingulate',
    'precuneus',
    'caudalanteriorcingulate',
    'parsopercularis',
    'CC_Mid_Posterior',
    'Caudate',
    'lateralorbitofrontal',
    'Putamen',
    'entorhinal',
    'superiortemporal',
    'parstriangularis',
    'insula',
    'Amygdala',
    'postcentral',
    'Pallidum',
    'superiorfrontal',
    'fusiform',
    'supramarginal',
    'inferiortemporal',
    'Thalamus',
    'inferiorparietal',
    'middletemporal',
    'Hippocampus',
    'precentral',
    'paracentral',
    'lateraloccipital'
]

# ============================================================================
# Statistical Functions
# ============================================================================

def mean_and_95ci(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate the mean and 95% confidence interval for a given array of data.

    Args:
        data: An array of numerical data points.

    Returns:
        Tuple containing (mean, ci_lower, ci_upper)
    """
    data = np.array(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    df = len(data) - 1
    ci = stats.t.interval(0.95, df, loc=mean, scale=sem)
    return mean, ci[0], ci[1]


def t_test_and_cohens_d(array1: np.ndarray, array2: np.ndarray, paired: bool = False) -> Dict:
    """
    Perform a t-test and calculate Cohen's d between two arrays.

    Args:
        array1: First array of numerical data.
        array2: Second array of numerical data.
        paired: If True, perform paired t-test.

    Returns:
        Dictionary with t-statistic, p-value, and Cohen's d.
    """
    array1 = np.array(array1)
    array2 = np.array(array2)

    if paired:
        t_stat, p_value = stats.ttest_rel(array1, array2)
    else:
        t_stat, p_value = stats.ttest_ind(array1, array2, equal_var=False)

    n1, n2 = len(array1), len(array2)
    s1, s2 = np.var(array1, ddof=1), np.var(array2, ddof=1)
    s = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))
    d = np.abs(np.mean(array1) - np.mean(array2)) / s if s > 0 else 0

    return {
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': d
    }


def apply_benjamini_hochberg(p_values: List[float]) -> List[float]:
    """
    Apply the Benjamini-Hochberg correction to a list of p-values.
    """
    rejected, corrected_p_values, _, _ = smm.multipletests(p_values, method='fdr_bh')
    return corrected_p_values.tolist()


def calculate_statistics(group1: np.ndarray, group2: np.ndarray,
                         group1_name: str = "Group 1",
                         group2_name: str = "Group 2",
                         metric_name: str = "Metric",
                         paired: bool = True) -> Dict:
    """
    Calculate comprehensive statistics comparing two groups.
    """
    g1 = np.array(group1)
    g2 = np.array(group2)

    if paired:
        valid_mask = ~(np.isnan(g1) | np.isnan(g2))
        g1 = g1[valid_mask]
        g2 = g2[valid_mask]
    else:
        g1 = g1[~np.isnan(g1)]
        g2 = g2[~np.isnan(g2)]

    mean1 = np.mean(g1)
    mean2 = np.mean(g2)
    std1 = np.std(g1, ddof=1)
    std2 = np.std(g2, ddof=1)
    n1 = len(g1)
    n2 = len(g2)
    se1 = std1 / np.sqrt(n1)
    se2 = std2 / np.sqrt(n2)

    ci1 = stats.t.interval(0.95, n1-1, loc=mean1, scale=se1)
    ci2 = stats.t.interval(0.95, n2-1, loc=mean2, scale=se2)

    if paired:
        t_stat, p_value = stats.ttest_rel(g1, g2)
        differences = g1 - g2
        cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    else:
        t_stat, p_value = stats.ttest_ind(g1, g2)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

    if abs(cohens_d) < 0.2:
        effect_size = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size = "small"
    elif abs(cohens_d) < 0.8:
        effect_size = "medium"
    else:
        effect_size = "large"

    return {
        'metric_name': metric_name,
        'group1_name': group1_name,
        'group2_name': group2_name,
        'group1_mean': mean1,
        'group2_mean': mean2,
        'group1_std': std1,
        'group2_std': std2,
        'group1_ci': ci1,
        'group2_ci': ci2,
        'group1_n': n1,
        'group2_n': n2,
        'paired': paired,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size': effect_size,
        't_statistic': t_stat
    }


def print_statistics(results: Dict, decimal_places: int = 4):
    """Print formatted statistics from calculate_statistics results."""
    print(f"\n{'='*70}")
    print(f"Statistical Analysis: {results['metric_name']}")
    print(f"Test Type: {'Paired' if results['paired'] else 'Independent'} t-test")
    print(f"{'='*70}")

    print(f"\n{results['group1_name']}:")
    print(f"  n = {results['group1_n']}")
    print(f"  Mean = {results['group1_mean']:.{decimal_places}f}")
    print(f"  SD = {results['group1_std']:.{decimal_places}f}")
    print(f"  95% CI = [{results['group1_ci'][0]:.{decimal_places}f}, {results['group1_ci'][1]:.{decimal_places}f}]")

    print(f"\n{results['group2_name']}:")
    print(f"  n = {results['group2_n']}")
    print(f"  Mean = {results['group2_mean']:.{decimal_places}f}")
    print(f"  SD = {results['group2_std']:.{decimal_places}f}")
    print(f"  95% CI = [{results['group2_ci'][0]:.{decimal_places}f}, {results['group2_ci'][1]:.{decimal_places}f}]")

    print(f"\nComparison:")
    print(f"  Mean difference = {results['group1_mean'] - results['group2_mean']:.{decimal_places}f}")
    print(f"  t-statistic = {results['t_statistic']:.{decimal_places}f}")
    sig = '***' if results['p_value'] < 0.001 else '**' if results['p_value'] < 0.01 else '*' if results['p_value'] < 0.05 else 'ns'
    print(f"  p-value = {results['p_value']:.{decimal_places}f} {sig}")
    print(f"  Cohen's d = {results['cohens_d']:.{decimal_places}f} ({results['effect_size']} effect)")


def analyze_and_print(group1, group2, group1_name="Group 1", group2_name="Group 2",
                      metric_name="Metric", decimal_places=4, paired=True) -> Dict:
    """Convenience function that calculates and prints statistics in one call."""
    results = calculate_statistics(group1, group2, group1_name, group2_name, metric_name, paired)
    print_statistics(results, decimal_places)
    return results


# ============================================================================
# Image Quality Metrics
# ============================================================================

def scale_image(image: np.ndarray) -> np.ndarray:
    """Scale image to [0, 1] range."""
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val == min_val:
        return np.zeros_like(image, dtype=np.float64)
    return (image - min_val) / (max_val - min_val)


def ssim_3d(img1: np.ndarray, img2: np.ndarray) -> float:
    """Compute 3D SSIM by averaging 2D SSIM over slices."""
    assert img1.shape == img2.shape, "Images must have the same shape"

    img1 = scale_image(img1.astype(np.float64))
    img2 = scale_image(img2.astype(np.float64))

    data_range = 1.0
    ssim_values = []
    for i in range(img1.shape[2]):
        ssim_values.append(ssim(img1[:, :, i], img2[:, :, i], data_range=data_range))
    return float(np.mean(ssim_values))


def rmse(img1: np.ndarray, img2: np.ndarray, scale: bool = True) -> float:
    """Root Mean Squared Error over the full 3D volume."""
    assert img1.shape == img2.shape, "Images must have the same shape"
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)
    if scale:
        x = scale_image(x)
        y = scale_image(y)
    mse = np.mean((x - y) ** 2)
    return float(np.sqrt(mse))


def psnr(img1: np.ndarray, img2: np.ndarray, data_range: Optional[float] = None,
         scale: bool = True, eps: float = 1e-12) -> float:
    """Peak Signal-to-Noise Ratio (dB) over the full 3D volume."""
    assert img1.shape == img2.shape, "Images must have the same shape"
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    if scale:
        x = scale_image(x)
        y = scale_image(y)
        dr = 1.0
    else:
        if data_range is None:
            xy_min = min(float(x.min()), float(y.min()))
            xy_max = max(float(x.max()), float(y.max()))
            dr = max(xy_max - xy_min, eps)
        else:
            dr = float(data_range)

    mse = np.mean((x - y) ** 2)
    if mse < eps:
        return float(100.0)
    return float(20.0 * np.log10(dr) - 10.0 * np.log10(mse))


def ncc(img1: np.ndarray, img2: np.ndarray, eps: float = 1e-12) -> float:
    """Normalized Cross-Correlation over the full 3D volume."""
    assert img1.shape == img2.shape, "Images must have the same shape"
    x = img1.astype(np.float64).ravel()
    y = img2.astype(np.float64).ravel()

    x = x - x.mean()
    y = y - y.mean()

    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    if denom < eps:
        return 0.0
    return float(np.dot(x, y) / denom)


def get_quality_metrics(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
    """Compute all quality metrics for two images."""
    return {
        'ssim': ssim_3d(img1, img2),
        'rmse': rmse(img1, img2),
        'psnr': psnr(img1, img2),
        'ncc': ncc(img1, img2)
    }


# ============================================================================
# Bland-Altman and Correlation Plots
# ============================================================================

def bland_altman_and_corr_plot(x: np.ndarray, y: np.ndarray,
                                title: Optional[str] = None,
                                unit: Optional[str] = None,
                                point_size: int = 25,
                                alpha: float = 0.8,
                                stats_only: bool = False,
                                figsize: Tuple[int, int] = (10, 5),
                                color: str = '#66c2a5') -> Dict:
    """
    Side-by-side Bland-Altman (left) and correlation (right) plots.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        raise ValueError("No finite data after masking NaNs/inf.")

    means = (x + y) / 2.0
    diffs = x - y

    bias = np.mean(diffs)
    sd = np.std(diffs, ddof=1)
    loa_lower = bias - 1.96 * sd
    loa_upper = bias + 1.96 * sd

    n = diffs.size
    n_outside = int(np.sum((diffs < loa_lower) | (diffs > loa_upper)))
    pct_outside = 100.0 * n_outside / n

    rho, pval = spearmanr(x, y)

    if stats_only:
        return {
            "bias": bias, "sd": sd,
            "loa_lower": loa_lower, "loa_upper": loa_upper,
            "n": n, "n_outside": n_outside, "pct_outside": pct_outside,
            "spearman_r": rho, "p_value": pval
        }

    fig, (ax_ba, ax_corr) = plt.subplots(1, 2, figsize=figsize)

    # Bland-Altman
    ax_ba.scatter(means, diffs, s=point_size, alpha=alpha, color=color)
    ax_ba.axhline(bias, linestyle='--', linewidth=1.5, color='k')
    ax_ba.axhline(loa_lower, linestyle=':', linewidth=1.5, color='k')
    ax_ba.axhline(loa_upper, linestyle=':', linewidth=1.5, color='k')

    xlab = "Mean of methods"
    ax_ba.set_xlabel(f"{xlab} ({unit})" if unit else xlab)
    ax_ba.set_ylabel(f"Difference (A − B) ({unit})" if unit else "Difference (A − B)")
    ax_ba.set_title("Bland–Altman" if not title else f"{title} — Bland–Altman")

    ba_txt = (f"bias = {bias:.3f}\n"
              f"SD = {sd:.3f}\n"
              f"LoA = [{loa_lower:.3f}, {loa_upper:.3f}]\n"
              f"outside LoA: {n_outside}/{n} ({pct_outside:.1f}%)")
    ax_ba.text(0.02, 0.98, ba_txt, transform=ax_ba.transAxes,
               ha='left', va='top', fontsize=10,
               bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Correlation
    ax_corr.scatter(x, y, s=point_size, alpha=alpha, color=color)
    lim_min = min(np.min(x), np.min(y))
    lim_max = max(np.max(x), np.max(y))
    ax_corr.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1.5)
    ax_corr.set_xlim(lim_min, lim_max)
    ax_corr.set_ylim(lim_min, lim_max)

    ax_corr.set_xlabel(f"Method A ({unit})" if unit else "Method A")
    ax_corr.set_ylabel(f"Method B ({unit})" if unit else "Method B")
    ax_corr.set_title("Correlation" if not title else f"{title} — Correlation")

    ax_corr.text(0.02, 0.98, f"Spearman ρ = {rho:.3f}\np = {pval:.3g}",
                 transform=ax_corr.transAxes, ha='left', va='top',
                 fontsize=10, bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    sns.despine(fig=fig)
    fig.tight_layout()

    return {
        "bias": bias, "sd": sd,
        "loa_lower": loa_lower, "loa_upper": loa_upper,
        "n": n, "n_outside": n_outside, "pct_outside": pct_outside,
        "spearman_r": rho, "p_value": pval,
        "fig": fig, "ax_ba": ax_ba, "ax_corr": ax_corr
    }


# ============================================================================
# Congruency Analysis
# ============================================================================

def get_congruency(df_ai: pd.DataFrame, x_var: str, y_var: str, region: str) -> float:
    """Calculate congruency rate (proportion of same sign) between two variables for a region."""
    df_region = df_ai[df_ai['Region'] == region]
    x_vec = df_region[x_var].values
    y_vec = df_region[y_var].values

    same_sign_count = np.sum(((x_vec >= 0) & (y_vec >= 0)) | ((x_vec < 0) & (y_vec < 0)))
    return same_sign_count / len(x_vec) if len(x_vec) > 0 else 0


def plot_congruency_scatterplot(df_ai: pd.DataFrame, x_var: str, y_var: str,
                                 region: str, ax=None, title: Optional[str] = None,
                                 show_stats: bool = True) -> Dict:
    """
    Enhanced congruency scatterplot with statistical annotations.
    """
    if ax is None:
        ax = plt.gca()

    df_region = df_ai[df_ai['Region'] == region].copy()
    x_vec = df_region[x_var].values
    y_vec = df_region[y_var].values

    # Filter out NaN and infinite values
    valid_mask = np.isfinite(x_vec) & np.isfinite(y_vec)
    x_vec = x_vec[valid_mask]
    y_vec = y_vec[valid_mask]

    same_sign = ((x_vec >= 0) & (y_vec >= 0)) | ((x_vec < 0) & (y_vec < 0))

    ax.scatter(x_vec[same_sign], y_vec[same_sign], color='#2166ac', marker='o',
               s=60, alpha=0.7, label='Congruent', edgecolors='white', linewidth=0.5)
    ax.scatter(x_vec[~same_sign], y_vec[~same_sign], color='#b2182b', marker='o',
               s=60, alpha=0.7, label='Incongruent', edgecolors='white', linewidth=0.5)

    same_sign_count = np.sum(same_sign)
    diff_sign_count = np.sum(~same_sign)
    total = len(x_vec)
    congruency_rate = same_sign_count / total if total > 0 else 0

    lim = np.max([np.max(np.abs(x_vec)), np.max(np.abs(y_vec))]) * 1.15

    # Quadrant shading
    ax.axhspan(0, lim, 0.5, 1, alpha=0.05, color='blue', zorder=0)
    ax.axhspan(-lim, 0, 0, 0.5, alpha=0.05, color='blue', zorder=0)
    ax.axhspan(0, lim, 0, 0.5, alpha=0.05, color='red', zorder=0)
    ax.axhspan(-lim, 0, 0.5, 1, alpha=0.05, color='red', zorder=0)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(x=0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)

    # Calculate Pearson correlation only on valid data
    if len(x_vec) > 1:
        r, p = stats.pearsonr(x_vec, y_vec)
    else:
        r, p = np.nan, np.nan

    if show_stats:
        textstr = f'Congruent: {same_sign_count}/{total} ({congruency_rate:.1%})\nr = {r:.2f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_xlabel(x_var, fontweight='bold', fontsize=11)
    ax.set_ylabel(y_var, fontweight='bold', fontsize=11)
    ax.set_title(title if title else region, fontweight='bold', fontsize=12)
    ax.set_aspect('equal')
    sns.despine(ax=ax)

    return {
        'congruency_rate': congruency_rate,
        'same_sign': same_sign_count,
        'diff_sign': diff_sign_count,
        'total': total,
        'r': r,
        'p': p
    }


# ============================================================================
# Utility for saving figures
# ============================================================================

_PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def _display_path(filepath: str) -> str:
    """Path relative to the package root, so printed output (and the saved
    notebook, which keeps it) does not carry the absolute path of whichever
    machine happened to run the analysis."""
    try:
        return os.path.relpath(filepath, _PACKAGE_ROOT)
    except ValueError:          # different drive on Windows
        return filepath


def save_figure(fig, filename: str, figures_dir: str, formats: List[str] = ['pdf', 'png']):
    """Save figure in multiple formats."""
    for fmt in formats:
        filepath = os.path.join(figures_dir, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=300, bbox_inches='tight', format=fmt)
        print(f"Saved: {_display_path(filepath)}")


def save_table(df: pd.DataFrame, filename: str, tables_dir: str,
               formats: List[str] = ['csv', 'xlsx']):
    """Save DataFrame in multiple formats."""
    for fmt in formats:
        filepath = os.path.join(tables_dir, f"{filename}.{fmt}")
        if fmt == 'csv':
            df.to_csv(filepath, index=False)
        elif fmt == 'xlsx':
            df.to_excel(filepath, index=False)
        print(f"Saved: {_display_path(filepath)}")
