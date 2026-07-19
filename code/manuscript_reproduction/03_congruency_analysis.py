"""
Script 03: Part 3 - Congruency Analysis

This script performs congruency analysis:
- Sign congruency between real FDG-PET asymmetry and FlowGAN/ASL asymmetry
- McNemar's test for comparing congruency rates
- Magnitude attenuation analysis
- Region-by-region congruency comparison

Works for both TLE and MCI datasets.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar

from utils import (
    plot_congruency_scatterplot, get_congruency, save_figure, save_table,
    BOXPLOT_PARAMS
)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures', '03_congruency_analysis')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'tables', '03_congruency_analysis')

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']


# ============================================================================
# Congruency Computation Functions
# ============================================================================

def compute_all_congruencies(df_ai):
    """Compute congruency rates for all regions."""
    regions = df_ai['Region'].unique()
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    c_asl_list = []
    c_recon_list = []

    for region in regions:
        c_asl = get_congruency(df_ai, 'PET AI Original', 'ASL AI', region)
        c_recon = get_congruency(df_ai, 'PET AI Original', 'PET AI Recon', region)
        c_asl_list.append(c_asl)
        c_recon_list.append(c_recon)

    df_congruency = pd.DataFrame({
        'Region': regions,
        'Congruency_ASL': c_asl_list,
        'Congruency_FlowGAN': c_recon_list,
        'Congruency_Diff': np.array(c_recon_list) - np.array(c_asl_list)
    })

    return df_congruency


def mcnemar_congruency_test(df_ai, region, reference_var='PET AI Original',
                             var1='ASL AI', var2='PET AI Recon'):
    """
    McNemar's test comparing congruency rates of two modalities with a reference.
    """
    df_region = df_ai[df_ai['Region'] == region].copy()

    ref = df_region[reference_var].values
    m1 = df_region[var1].values  # ASL
    m2 = df_region[var2].values  # Synthetic

    # Congruency for each modality
    cong_m1 = ((ref >= 0) & (m1 >= 0)) | ((ref < 0) & (m1 < 0))
    cong_m2 = ((ref >= 0) & (m2 >= 0)) | ((ref < 0) & (m2 < 0))

    # Build contingency table for McNemar's test
    a = np.sum(cong_m1 & cong_m2)      # Both congruent
    b = np.sum(cong_m1 & ~cong_m2)     # ASL congruent, Synthetic not
    c = np.sum(~cong_m1 & cong_m2)     # Synthetic congruent, ASL not
    d = np.sum(~cong_m1 & ~cong_m2)    # Neither congruent

    contingency_table = np.array([[a, b], [c, d]])

    try:
        result = mcnemar(contingency_table, exact=True)
        p_value = result.pvalue
    except Exception:
        p_value = np.nan

    return {
        'contingency_table': contingency_table,
        'p_value': p_value,
        'ASL_congruency': np.mean(cong_m1),
        'Synthetic_congruency': np.mean(cong_m2),
        'n_ASL_only': b,
        'n_Synthetic_only': c
    }


def magnitude_attenuation_analysis(df_ai, region):
    """Analyze magnitude attenuation in synthetic vs real FDG."""
    df_region = df_ai[df_ai['Region'] == region].copy()

    real_ai = df_region['PET AI Original'].values
    synth_ai = df_region['PET AI Recon'].values

    # Only for congruent cases (same sign)
    congruent = ((real_ai >= 0) & (synth_ai >= 0)) | ((real_ai < 0) & (synth_ai < 0))

    # Magnitude ratio (absolute values)
    valid = np.abs(real_ai[congruent]) > 0.001
    ratio = np.abs(synth_ai[congruent][valid]) / np.abs(real_ai[congruent][valid])
    ratio = ratio[np.isfinite(ratio)]

    if len(ratio) == 0:
        return {
            'mean_ratio': np.nan,
            'std_ratio': np.nan,
            'median_ratio': np.nan,
            'n_congruent': np.sum(congruent),
            'n_amplified': 0,
            'n_preserved': 0,
            'n_attenuated': 0
        }

    n_amplified = np.sum(ratio > 1.2)
    n_preserved = np.sum((ratio >= 0.8) & (ratio <= 1.2))
    n_attenuated = np.sum(ratio < 0.8)

    return {
        'mean_ratio': np.mean(ratio),
        'std_ratio': np.std(ratio),
        'median_ratio': np.median(ratio),
        'n_congruent': np.sum(congruent),
        'n_amplified': n_amplified,
        'n_preserved': n_preserved,
        'n_attenuated': n_attenuated,
        'pct_amplified': n_amplified / len(ratio) * 100,
        'pct_preserved': n_preserved / len(ratio) * 100,
        'pct_attenuated': n_attenuated / len(ratio) * 100
    }


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_congruency_bar_chart(df_congruency, dataset_name):
    """Plot bar chart showing congruency difference per region."""
    df_sorted = df_congruency.sort_values('Congruency_Diff')

    colors = ['lightblue' if diff >= 0 else 'lightcoral' for diff in df_sorted['Congruency_Diff']]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(df_sorted)), df_sorted['Congruency_Diff'], color=colors)
    ax.set_xlabel('Regions', fontweight='bold')
    ax.set_ylabel('Congruency Difference (Synthetic PET - ASL)', fontweight='bold')
    ax.set_title(f'{dataset_name} - Congruency Difference by Region', fontweight='bold')
    ax.set_xticks(np.arange(len(df_sorted)))
    ax.set_xticklabels(df_sorted['Region'], rotation=90)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)

    # Add annotations
    n_improved = np.sum(df_sorted['Congruency_Diff'] > 0)
    n_total = len(df_sorted)
    ax.annotate(f'FlowGAN > ASL: {n_improved}/{n_total}', xy=(n_total - 5, df_sorted['Congruency_Diff'].max() + 0.02),
                fontsize=10, color='lightblue')

    sns.despine(ax=ax)
    plt.tight_layout()

    return fig


def plot_congruency_scatterplots_grid(df_ai, regions, dataset_name):
    """Plot congruency scatterplots for selected regions."""
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 2, figsize=(12, 5 * n_regions))

    if n_regions == 1:
        axes = axes.reshape(1, -1)

    for i, region in enumerate(regions):
        if region not in df_ai['Region'].unique():
            continue

        # Real PET vs ASL
        plot_congruency_scatterplot(df_ai, 'PET AI Original', 'ASL AI', region, axes[i, 0],
                                    title=f'{region}: Real PET vs ASL')

        # Real PET vs Synthetic PET
        plot_congruency_scatterplot(df_ai, 'PET AI Original', 'PET AI Recon', region, axes[i, 1],
                                    title=f'{region}: Real PET vs Synthetic PET')

    fig.suptitle(f'{dataset_name} - Congruency Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_congruency_comparison_summary(df_ai, dataset_name):
    """Create summary figure with overall congruency comparison."""
    regions = df_ai['Region'].unique()
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    # Calculate overall congruency
    all_cong_asl = []
    all_cong_synth = []

    for region in regions:
        df_region = df_ai[df_ai['Region'] == region]
        ref = df_region['PET AI Original'].values
        asl = df_region['ASL AI'].values
        synth = df_region['PET AI Recon'].values

        cong_asl = ((ref >= 0) & (asl >= 0)) | ((ref < 0) & (asl < 0))
        cong_synth = ((ref >= 0) & (synth >= 0)) | ((ref < 0) & (synth < 0))

        all_cong_asl.extend(cong_asl)
        all_cong_synth.extend(cong_synth)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel A: Overall congruency rates
    rates = [np.mean(all_cong_synth), np.mean(all_cong_asl)]
    ax = axes[0]
    bars = ax.bar(['Synthetic PET', 'ASL'], rates, color=['#2166ac', '#7fbf7f'], edgecolor='black')
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Congruency Rate', fontweight='bold')
    ax.set_title('A. Overall Sign Congruency', fontweight='bold')
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontsize=12, fontweight='bold')
    sns.despine(ax=ax)

    # Panel B: Region-level comparison
    df_cong = compute_all_congruencies(df_ai)
    ax = axes[1]
    ax.scatter(df_cong['Congruency_ASL'], df_cong['Congruency_FlowGAN'],
               alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlabel('ASL Congruency', fontweight='bold')
    ax.set_ylabel('Synthetic PET Congruency', fontweight='bold')
    ax.set_title('B. Region-Level Comparison', fontweight='bold')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    n_above = np.sum(df_cong['Congruency_FlowGAN'] > df_cong['Congruency_ASL'])
    ax.text(0.05, 0.95, f'Synthetic PET > ASL: {n_above}/{len(df_cong)}',
            transform=ax.transAxes, fontsize=10)
    sns.despine(ax=ax)

    # Panel C: Distribution of differences
    ax = axes[2]
    ax.hist(df_cong['Congruency_Diff'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(df_cong['Congruency_Diff']), color='green', linestyle='-', linewidth=2)
    ax.set_xlabel('Congruency Difference (Synthetic PET - ASL)', fontweight='bold')
    ax.set_ylabel('Number of Regions', fontweight='bold')
    ax.set_title('C. Distribution of Differences', fontweight='bold')
    ax.text(0.05, 0.95, f'Mean diff: {np.mean(df_cong["Congruency_Diff"]):.3f}',
            transform=ax.transAxes, fontsize=10, color='green')
    sns.despine(ax=ax)

    fig.suptitle(f'{dataset_name} - Congruency Summary', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig, df_cong


def plot_magnitude_analysis(df_ai, regions, dataset_name):
    """Plot magnitude attenuation analysis for key regions."""
    fig, axes = plt.subplots(1, len(regions), figsize=(5 * len(regions), 5))
    if len(regions) == 1:
        axes = [axes]

    for ax, region in zip(axes, regions):
        df_region = df_ai[df_ai['Region'] == region]

        if len(df_region) == 0:
            continue

        real = np.abs(df_region['PET AI Original'].values)
        synth = np.abs(df_region['PET AI Recon'].values)

        ax.scatter(real, synth, color='#2166ac', alpha=0.6, s=50, edgecolors='white')
        max_val = max(np.max(real), np.max(synth)) * 1.1
        ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)

        r = np.corrcoef(real, synth)[0, 1]
        ax.text(0.05, 0.95, f'r = {r:.2f}', transform=ax.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel('|Real PET AI|', fontweight='bold')
        ax.set_ylabel('|Synthetic PET AI|', fontweight='bold')
        ax.set_title(f'{region}', fontweight='bold')
        ax.set_xlim([0, max_val])
        ax.set_ylim([0, max_val])
        sns.despine(ax=ax)

    fig.suptitle(f'{dataset_name} - Magnitude Preservation', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# ============================================================================
# Main Analysis
# ============================================================================

def build_asymmetry_dataframe(df_merged):
    """Build asymmetry index DataFrame for all regions."""
    regions = sorted(set(df_merged['region_name']))
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    df_ai = pd.DataFrame()
    for region in regions:
        try:
            if df_merged[df_merged['region_name'] == region]['side'].values[0] == 'Mid':
                continue

            left_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Left')]
            right_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Right')]

            if len(left_df) != len(right_df):
                continue

            ai_pet = (left_df['value_pet_original'].values - right_df['value_pet_original'].values) / \
                     (left_df['value_pet_original'].values + right_df['value_pet_original'].values)
            ai_recon = (left_df['value_pet_recon'].values - right_df['value_pet_recon'].values) / \
                       (left_df['value_pet_recon'].values + right_df['value_pet_recon'].values)
            ai_asl = (left_df['value_asl'].values - right_df['value_asl'].values) / \
                     (left_df['value_asl'].values + right_df['value_asl'].values)

            region_df = pd.DataFrame({
                'Subject': left_df['subject'].values,
                'Region': region,
                'PET AI Original': ai_pet,
                'PET AI Recon': ai_recon,
                'ASL AI': ai_asl
            })
            df_ai = pd.concat([df_ai, region_df])
        except Exception:
            continue

    return df_ai.reset_index(drop=True)


def run_analysis(dataset='TLE', atlas='DKT'):
    """Run congruency analysis for specified dataset and atlas."""
    print(f"\n{'='*60}")
    print(f"Running Congruency Analysis - {dataset} ({atlas} Atlas)")
    print(f"{'='*60}\n")

    # Load data
    if dataset == 'TLE':
        if atlas == 'HarvardOxford':
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_ho.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_ho.pkl')
        else:  # DKT
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')
    else:  # MCI
        if atlas == 'HarvardOxford':
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci_ho.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci_ho.pkl')
        else:  # DKT
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')

    with open(pkl_path, 'rb') as f:
        df_merged = pickle.load(f)

    print(f"Loaded {len(df_merged['subject'].unique())} subjects")

    # Build asymmetry DataFrame
    df_ai = build_asymmetry_dataframe(df_merged)
    print(f"Built asymmetry DataFrame with {len(df_ai)} rows")

    # Check if we have sufficient data for analysis
    if len(df_ai) == 0:
        print(f"Warning: No asymmetry data available for {dataset} dataset")
        print(f"Skipping congruency analysis for {dataset}")
        return None, None, None, None

    # Compute all congruencies
    df_congruency = compute_all_congruencies(df_ai)
    print(f"\nCongruency Summary:")
    print(f"  Mean ASL congruency: {df_congruency['Congruency_ASL'].mean():.3f}")
    print(f"  Mean FlowGAN congruency: {df_congruency['Congruency_FlowGAN'].mean():.3f}")
    print(f"  Regions where FlowGAN > ASL: {np.sum(df_congruency['Congruency_Diff'] > 0)}/{len(df_congruency)}")

    save_table(df_congruency, f'congruency_by_region_{dataset}_{atlas}', TABLES_DIR)

    # Summary figure
    fig_summary, _ = plot_congruency_comparison_summary(df_ai, dataset)
    save_figure(fig_summary, f'congruency_summary_{dataset}_{atlas}', FIGURES_DIR)
    plt.close(fig_summary)

    # Bar chart
    fig_bar = plot_congruency_bar_chart(df_congruency, dataset)
    save_figure(fig_bar, f'congruency_bar_chart_{dataset}_{atlas}', FIGURES_DIR)
    plt.close(fig_bar)

    # Scatterplots for key regions
    key_regions = ['Hippocampus', 'parahippocampal', 'Amygdala', 'insula']
    available_regions = [r for r in key_regions if r in df_ai['Region'].unique()]

    if available_regions:
        fig_scatter = plot_congruency_scatterplots_grid(df_ai, available_regions, dataset)
        save_figure(fig_scatter, f'congruency_scatterplots_{dataset}_{atlas}', FIGURES_DIR)
        plt.close(fig_scatter)

        # Magnitude analysis
        fig_mag = plot_magnitude_analysis(df_ai, available_regions, dataset)
        save_figure(fig_mag, f'magnitude_preservation_{dataset}_{atlas}', FIGURES_DIR)
        plt.close(fig_mag)

    # McNemar's test for key regions
    mcnemar_results = []
    for region in available_regions:
        result = mcnemar_congruency_test(df_ai, region)
        result['Region'] = region
        mcnemar_results.append(result)

    df_mcnemar = pd.DataFrame(mcnemar_results)
    df_mcnemar = df_mcnemar[['Region', 'ASL_congruency', 'Synthetic_congruency', 'p_value', 'n_ASL_only', 'n_Synthetic_only']]
    save_table(df_mcnemar, f'mcnemar_test_results_{dataset}_{atlas}', TABLES_DIR)
    print("\nMcNemar's Test Results:")
    print(df_mcnemar.to_string(index=False))

    # Magnitude attenuation for key regions
    mag_results = []
    for region in available_regions:
        result = magnitude_attenuation_analysis(df_ai, region)
        result['Region'] = region
        mag_results.append(result)

    df_magnitude = pd.DataFrame(mag_results)
    save_table(df_magnitude, f'magnitude_attenuation_{dataset}_{atlas}', TABLES_DIR)

    plt.close('all')

    return df_ai, df_congruency, df_mcnemar, df_magnitude


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Run for TLE with DKT
    print("\n" + "=" * 70)
    print("TLE - DKT ATLAS")
    print("=" * 70)
    df_ai_tle, df_cong_tle, df_mcnemar_tle, df_mag_tle = run_analysis('TLE', 'DKT')

    # Run for TLE with Harvard-Oxford
    print("\n" + "=" * 70)
    print("TLE - HARVARD-OXFORD ATLAS")
    print("=" * 70)
    df_ai_tle_ho, df_cong_tle_ho, df_mcnemar_tle_ho, df_mag_tle_ho = run_analysis('TLE', 'HarvardOxford')

    # Run for MCI with DKT
    print("\n" + "=" * 70)
    print("MCI - DKT ATLAS")
    print("=" * 70)
    df_ai_mci, df_cong_mci, df_mcnemar_mci, df_mag_mci = run_analysis('MCI', 'DKT')

    # Run for MCI with Harvard-Oxford
    print("\n" + "=" * 70)
    print("MCI - HARVARD-OXFORD ATLAS")
    print("=" * 70)
    df_ai_mci_ho, df_cong_mci_ho, df_mcnemar_mci_ho, df_mag_mci_ho = run_analysis('MCI', 'HarvardOxford')

    print("\n" + "=" * 70)
    print("Congruency Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 70)
