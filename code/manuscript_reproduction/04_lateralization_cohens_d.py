"""
Script 04: Lateralization Capacity - Cohen's d Scatterplot

This script analyzes lateralization capacity for TLE by computing Cohen's d
for distinguishing L-TLE from R-TLE using asymmetry indices, and creates
multi-colored quadrant scatterplots.

For MCI, it computes Cohen's d for distinguishing MCI from HC using SUVR values.

Works for both TLE and MCI datasets with DKT and Harvard-Oxford atlases.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu

from utils import save_figure, save_table, BOXPLOT_PARAMS, MCI_REGIONS

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures', '04_lateralization_cohens_d')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'tables', '04_lateralization_cohens_d')

# Clinical data paths
CLINICAL_DATA_TLE = os.path.join(SCRIPT_DIR, 'data', 'clinical_metadata.xlsx')
MCI_CONTROL_LIST = os.path.join(SCRIPT_DIR, 'data', 'list_of_control_subjects.txt')
MCI_PATIENT_LIST = os.path.join(SCRIPT_DIR, 'data', 'list_of_MCI_subjects.txt')

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']


# ============================================================================
# Load Clinical Metadata
# ============================================================================

def load_clinical_metadata(pet_subject_ids=None):
    """Load and process clinical metadata for TLE lateralization.

    Filters for well-lateralized TLE subjects:
    1. Must have PET data (if pet_subject_ids provided)
    2. Must have temporal onset (localization contains 'temporal')
    3. Must have clear lateralization (left or right, excluding bilateral/unknown)

    Args:
        pet_subject_ids: Optional list of subject IDs (e.g., 'sub-RID0681') that have PET data.
                        If provided, only subjects in this list will be included.

    Returns:
        DataFrame with columns ['Subject', 'isLeft'] for well-lateralized TLE subjects.
    """
    df_metadata = pd.read_excel(CLINICAL_DATA_TLE)
    df_metadata['record_id'] = ['sub-RID' + str(x).zfill(4) for x in df_metadata['record_id'].values]

    # Step 1: Filter to subjects with PET data (if provided)
    if pet_subject_ids is not None:
        df_metadata = df_metadata[df_metadata['record_id'].isin(pet_subject_ids)].copy()
        print(f"  Subjects with PET data: {len(df_metadata)}")

    # Step 2: Filter for well-lateralized subjects (left or right, excluding bilateral/unknown)
    df_well_lat = df_metadata[df_metadata['clinicalHypothesis1_Lateralization'].isin(['left', 'Left', 'right', 'Right'])].copy()
    print(f"  Well-lateralized subjects (L/R only): {len(df_well_lat)}")

    # Step 3: Filter for temporal onset (localization contains 'temporal')
    df_well_lat['loc_clean'] = df_well_lat['clinicalHypothesis1_Localization'].str.strip().str.lower()
    df_temporal = df_well_lat[df_well_lat['loc_clean'].str.contains('temporal', na=False)].copy()
    print(f"  With temporal onset: {len(df_temporal)}")

    # Create lateralization columns
    df_temporal['isLeft'] = 0
    df_temporal['isRight'] = 0
    df_temporal.loc[df_temporal['clinicalHypothesis1_Lateralization'].isin(['left', 'Left']), 'isLeft'] = 1
    df_temporal.loc[df_temporal['clinicalHypothesis1_Lateralization'].isin(['right', 'Right']), 'isRight'] = 1

    # Final output
    df_metadata_lr = df_temporal[['record_id', 'isLeft']].copy()
    df_metadata_lr.columns = ['Subject', 'isLeft']

    return df_metadata_lr


def load_mci_metadata():
    """Load MCI diagnosis labels."""
    control_subjects = np.loadtxt(MCI_CONTROL_LIST, dtype='str')
    mci_subjects = np.loadtxt(MCI_PATIENT_LIST, dtype='str')

    subjects = np.concatenate([control_subjects, mci_subjects])
    labels = [0] * len(control_subjects) + [1] * len(mci_subjects)

    return pd.DataFrame({'Subject': subjects, 'is_mci': labels})


# ============================================================================
# Compute Cohen's d for Lateralization
# ============================================================================

def compute_cohens_d(left_vals, right_vals):
    """Compute Cohen's d for separating L-TLE from R-TLE."""
    n1, n2 = len(left_vals), len(right_vals)
    if n1 < 2 or n2 < 2:
        return np.nan

    s1, s2 = np.var(left_vals, ddof=1), np.var(right_vals, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return np.nan

    d = (np.mean(left_vals) - np.mean(right_vals)) / pooled_std
    return d


def compute_lateralization_capacity(df_ai, df_metadata):
    """Compute Cohen's d for each modality and region."""
    # Merge with metadata
    df_ai_merged = df_ai.merge(df_metadata, on='Subject', how='inner')

    regions = df_ai_merged['Region'].unique()
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    results = []

    for region in regions:
        df_region = df_ai_merged[df_ai_merged['Region'] == region]

        left_tle = df_region[df_region['isLeft'] == 1]
        right_tle = df_region[df_region['isLeft'] == 0]

        if len(left_tle) < 2 or len(right_tle) < 2:
            continue

        # Cohen's d for Real FDG
        d_real = compute_cohens_d(
            left_tle['PET AI Original'].values,
            right_tle['PET AI Original'].values
        )

        # Cohen's d for Synthetic FDG
        d_synth = compute_cohens_d(
            left_tle['PET AI Recon'].values,
            right_tle['PET AI Recon'].values
        )

        # Cohen's d for ASL
        d_asl = compute_cohens_d(
            left_tle['ASL AI'].values,
            right_tle['ASL AI'].values
        )

        results.append({
            'Region': region,
            'Cohens_d_Real_FDG': d_real,
            'Cohens_d_Synthetic_FDG': d_synth,
            'Cohens_d_ASL': d_asl,
            'n_left': len(left_tle),
            'n_right': len(right_tle)
        })

    return pd.DataFrame(results)


def build_suvr_dataframe_mci(df_merged, metadata, regions_filter=None, atlas='DKT'):
    """Build SUVR DataFrame for MCI analysis (bilateral values, not asymmetry).

    Args:
        df_merged: DataFrame with regional values
        metadata: DataFrame with Subject and is_mci columns
        regions_filter: Optional list of regions to include (default: MCI_REGIONS for DKT atlas)
        atlas: Atlas type ('DKT' or 'HarvardOxford'). MCI_REGIONS filtering only applied for DKT.
    """
    # Get putamen for normalization
    left_put_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_put_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]
    left_put_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_put_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]
    left_put_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_put_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]

    subjects = metadata['Subject'].unique()
    subjects_in_data = df_merged['subject'].unique()
    subjects = [s for s in subjects if s in subjects_in_data]

    # Use MCI_REGIONS for DKT atlas only, use all available regions for HarvardOxford
    available_regions = sorted(set(df_merged['region_name']))
    if regions_filter is not None:
        regions = [r for r in regions_filter if r in available_regions and r != 'Putamen']
    elif atlas == 'DKT':
        regions = [r for r in MCI_REGIONS if r in available_regions and r != 'Putamen']
    else:  # HarvardOxford - use all available regions
        regions = [r for r in available_regions if r != 'Putamen']

    print(f"  Using {len(regions)} MCI-relevant regions for analysis ({atlas} atlas)")

    data = []
    for sub in subjects:
        try:
            # Get putamen normalization
            put_og_left = left_put_og[left_put_og['subject'] == sub]['value_pet_original'].values
            put_og_right = right_put_og[right_put_og['subject'] == sub]['value_pet_original'].values
            put_recon_left = left_put_recon[left_put_recon['subject'] == sub]['value_pet_recon'].values
            put_recon_right = right_put_recon[right_put_recon['subject'] == sub]['value_pet_recon'].values
            put_asl_left = left_put_asl[left_put_asl['subject'] == sub]['value_asl'].values
            put_asl_right = right_put_asl[right_put_asl['subject'] == sub]['value_asl'].values

            # Check if putamen values exist
            if len(put_og_left) == 0 or len(put_og_right) == 0:
                continue
            if len(put_recon_left) == 0 or len(put_recon_right) == 0:
                continue
            if len(put_asl_left) == 0 or len(put_asl_right) == 0:
                continue

            put_og = put_og_left[0] + put_og_right[0]
            put_recon = put_recon_left[0] + put_recon_right[0]
            put_asl = put_asl_left[0] + put_asl_right[0]

            # Skip if any normalization value is zero or NaN
            if put_og == 0 or put_recon == 0 or put_asl == 0:
                continue
            if np.isnan(put_og) or np.isnan(put_recon) or np.isnan(put_asl):
                continue

            is_mci = metadata[metadata['Subject'] == sub]['is_mci'].values[0]
            row = {'Subject': sub, 'is_mci': is_mci}

            for region in regions:
                df_region = df_merged[(df_merged['subject'] == sub) & (df_merged['region_name'] == region)]
                if len(df_region) == 0:
                    row[f'{region}_real'] = np.nan
                    row[f'{region}_synth'] = np.nan
                    row[f'{region}_asl'] = np.nan
                    continue

                # Average left and right
                val_og = df_region['value_pet_original'].mean() / put_og
                val_recon = df_region['value_pet_recon'].mean() / put_recon
                val_asl = df_region['value_asl'].mean() / put_asl

                row[f'{region}_real'] = val_og
                row[f'{region}_synth'] = val_recon
                row[f'{region}_asl'] = val_asl

            data.append(row)
        except Exception:
            continue

    return pd.DataFrame(data)


def compute_mci_discriminability(df_suvr, metadata, atlas='DKT'):
    """Compute Cohen's d for distinguishing MCI from HC for each modality and region.
    
    Args:
        df_suvr: DataFrame with SUVR values
        metadata: DataFrame with Subject and is_mci columns
        atlas: Atlas type ('DKT' or 'HarvardOxford'). MCI_REGIONS filtering only applied for DKT.
    """
    # Get region names from columns
    regions = sorted(set([c.rsplit('_', 1)[0] for c in df_suvr.columns
                        if c.endswith('_real') or c.endswith('_synth') or c.endswith('_asl')]))
    # Filter to only MCI_REGIONS for DKT atlas only
    if atlas == 'DKT':
        regions = [r for r in regions if r in MCI_REGIONS and r not in EXCLUDE_REGIONS]
    else:  # HarvardOxford - use all available regions
        regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    results = []

    for region in regions:
        real_col = f'{region}_real'
        synth_col = f'{region}_synth'
        asl_col = f'{region}_asl'

        if real_col not in df_suvr.columns or synth_col not in df_suvr.columns or asl_col not in df_suvr.columns:
            continue

        # Get values for HC and MCI
        hc_real = df_suvr[df_suvr['is_mci'] == 0][real_col].values
        mci_real = df_suvr[df_suvr['is_mci'] == 1][real_col].values

        hc_synth = df_suvr[df_suvr['is_mci'] == 0][synth_col].values
        mci_synth = df_suvr[df_suvr['is_mci'] == 1][synth_col].values

        hc_asl = df_suvr[df_suvr['is_mci'] == 0][asl_col].values
        mci_asl = df_suvr[df_suvr['is_mci'] == 1][asl_col].values

        # Filter out NaN values
        hc_real = hc_real[~np.isnan(hc_real)]
        mci_real = mci_real[~np.isnan(mci_real)]
        hc_synth = hc_synth[~np.isnan(hc_synth)]
        mci_synth = mci_synth[~np.isnan(mci_synth)]
        hc_asl = hc_asl[~np.isnan(hc_asl)]
        mci_asl = mci_asl[~np.isnan(mci_asl)]

        if len(hc_real) < 2 or len(mci_real) < 2:
            continue

        # Cohen's d for Real FDG
        d_real = compute_cohens_d(mci_real, hc_real)

        # Cohen's d for Synthetic FDG
        d_synth = compute_cohens_d(mci_synth, hc_synth)

        # Cohen's d for ASL
        d_asl = compute_cohens_d(mci_asl, hc_asl)

        results.append({
            'Region': region,
            'Cohens_d_Real_FDG': d_real,
            'Cohens_d_Synthetic_FDG': d_synth,
            'Cohens_d_ASL': d_asl,
            'n_hc': len(hc_real),
            'n_mci': len(mci_real)
        })

    return pd.DataFrame(results)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_cohens_d_scatterplot_quadrant(df_lateralization, dataset_name='TLE', atlas='DKT'):
    """
    Create Cohen's d scatterplot with colored quadrants.
    X-axis: ASL Cohen's d
    Y-axis: Synthetic FDG Cohen's d
    Colors indicate whether each modality improves over the other.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    x = df_lateralization['Cohens_d_ASL'].values
    y = df_lateralization['Cohens_d_Synthetic_FDG'].values
    regions = df_lateralization['Region'].values

    # Define threshold (1 SD of difference)
    threshold = np.nanstd(np.abs(y - x))

    # Color coding based on magnitude comparison (using absolute values)
    colors = []
    for xi, yi in zip(x, y):
        if np.abs(yi - xi) <= threshold:
            colors.append('gray')  # Similar performance
        elif np.abs(yi) > np.abs(xi):
            colors.append('#2166ac')  # FlowGAN better (blue)
        else:
            colors.append('#b2182b')  # ASL better (red)

    # Create scatter
    scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add region labels for outliers
    for xi, yi, region in zip(x, y, regions):
        if np.abs(yi - xi) > threshold * 2:
            ax.annotate(region, (xi, yi), fontsize=8, alpha=0.7)

    # Unity line
    lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1.5, alpha=0.5)

    # Quadrant shading
    ax.axhspan(0, lim, 0.5, 1, alpha=0.03, color='green', zorder=0)  # Q1: Both positive
    ax.axhspan(-lim, 0, 0, 0.5, alpha=0.03, color='green', zorder=0)  # Q3: Both negative
    ax.axhspan(0, lim, 0, 0.5, alpha=0.03, color='red', zorder=0)    # Q2: Discordant
    ax.axhspan(-lim, 0, 0.5, 1, alpha=0.03, color='red', zorder=0)   # Q4: Discordant

    # Reference lines
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Cohen's d (ASL)", fontweight='bold', fontsize=14)
    ax.set_ylabel("Cohen's d (Synthetic PET)", fontweight='bold', fontsize=14)
    ax.set_title(f'{dataset_name} Lateralization Capacity ({atlas} Atlas)', fontweight='bold', fontsize=16)
    ax.set_aspect('equal')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2166ac', label='Synthetic PET > ASL', alpha=0.7),
        Patch(facecolor='#b2182b', label='ASL > Synthetic PET', alpha=0.7),
        Patch(facecolor='gray', label='Similar', alpha=0.7)
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    # Add summary stats (using absolute values for magnitude comparison)
    n_flowgan_better = np.sum(np.abs(np.array(y)) > np.abs(np.array(x)) + threshold)
    n_asl_better = np.sum(np.abs(np.array(x)) > np.abs(np.array(y)) + threshold)
    n_similar = len(x) - n_flowgan_better - n_asl_better
    summary = f'Synthetic PET better: {n_flowgan_better}\nASL better: {n_asl_better}\nSimilar: {n_similar}'
    ax.text(0.95, 0.05, summary, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    sns.despine(ax=ax)
    plt.tight_layout()

    return fig


def plot_cohens_d_three_way_comparison(df_lateralization, dataset_name='TLE', atlas='DKT'):
    """Create three-panel comparison of Cohen's d across modalities."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Collect data for plotting
    modalities = ['Real PET', 'Synthetic PET', 'ASL']
    cols = ['Cohens_d_Real_FDG', 'Cohens_d_Synthetic_FDG', 'Cohens_d_ASL']
    colors_map = {'Real PET': '#1f77b4', 'Synthetic PET': '#2ca02c', 'ASL': '#ff7f0e'}

    # Panel A: Real PET vs ASL
    x = df_lateralization['Cohens_d_ASL'].values
    y = df_lateralization['Cohens_d_Real_FDG'].values
    axes[0].scatter(x, y, c=colors_map['Real PET'], s=80, alpha=0.7, edgecolors='white')
    lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.2
    axes[0].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    axes[0].set_xlim(-lim, lim)
    axes[0].set_ylim(-lim, lim)
    axes[0].set_xlabel("Cohen's d (ASL)", fontweight='bold')
    axes[0].set_ylabel("Cohen's d (Real PET)", fontweight='bold')
    axes[0].set_title('A. Real PET vs ASL', fontweight='bold')
    axes[0].set_aspect('equal')

    # Panel B: Synthetic PET vs ASL
    y2 = df_lateralization['Cohens_d_Synthetic_FDG'].values
    axes[1].scatter(x, y2, c=colors_map['Synthetic PET'], s=80, alpha=0.7, edgecolors='white')
    axes[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].set_xlabel("Cohen's d (ASL)", fontweight='bold')
    axes[1].set_ylabel("Cohen's d (Synthetic PET)", fontweight='bold')
    axes[1].set_title('B. Synthetic PET vs ASL', fontweight='bold')
    axes[1].set_aspect('equal')

    # Panel C: Synthetic PET vs Real PET
    axes[2].scatter(y, y2, c='purple', s=80, alpha=0.7, edgecolors='white')
    axes[2].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    axes[2].set_xlim(-lim, lim)
    axes[2].set_ylim(-lim, lim)
    axes[2].set_xlabel("Cohen's d (Real PET)", fontweight='bold')
    axes[2].set_ylabel("Cohen's d (Synthetic PET)", fontweight='bold')
    axes[2].set_title('C. Synthetic PET vs Real PET', fontweight='bold')
    axes[2].set_aspect('equal')

    for ax in axes:
        ax.axhline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.axvline(0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
        sns.despine(ax=ax)

    fig.suptitle(f'{dataset_name} - Lateralization Capacity Comparison ({atlas})', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_cohens_d_bar_comparison(df_lateralization, dataset_name='TLE', atlas='DKT'):
    """Create bar chart comparing Cohen's d across modalities for key regions."""
    key_regions = ['Hippocampus', 'parahippocampal', 'Amygdala', 'insula', 'entorhinal', 'inferiortemporal']
    df_plot = df_lateralization[df_lateralization['Region'].isin(key_regions)].copy()

    if len(df_plot) == 0:
        return None

    # Melt for plotting
    df_melt = df_plot.melt(id_vars='Region',
                            value_vars=['Cohens_d_Real_FDG', 'Cohens_d_Synthetic_FDG', 'Cohens_d_ASL'],
                            var_name='Modality', value_name="Cohen's d")
    df_melt['Modality'] = df_melt['Modality'].map({
        'Cohens_d_Real_FDG': 'Real PET',
        'Cohens_d_Synthetic_FDG': 'Synthetic PET',
        'Cohens_d_ASL': 'ASL'
    })

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df_melt, x='Region', y="Cohen's d", hue='Modality', ax=ax,
                palette={'Real PET': '#1f77b4', 'Synthetic PET': '#2ca02c', 'ASL': '#ff7f0e'})
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Region', fontweight='bold')
    ax.set_ylabel("Cohen's d", fontweight='bold')
    ax.set_title(f'{dataset_name} - Lateralization Capacity by Region ({atlas})', fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    ax.legend(title='Modality')
    sns.despine(ax=ax)
    plt.tight_layout()

    return fig


def plot_improvement_over_asl(df_lateralization, dataset_name='TLE', atlas='DKT'):
    """Plot showing where Synthetic FDG improves over ASL."""
    df = df_lateralization.copy()
    df['Synth_vs_ASL'] = df['Cohens_d_Synthetic_FDG'] - df['Cohens_d_ASL']
    df['Real_vs_ASL'] = df['Cohens_d_Real_FDG'] - df['Cohens_d_ASL']

    df_sorted = df.sort_values('Synth_vs_ASL')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Improvement distribution
    colors = ['#2ca02c' if x > 0 else '#d62728' for x in df_sorted['Synth_vs_ASL']]
    axes[0].bar(np.arange(len(df_sorted)), df_sorted['Synth_vs_ASL'], color=colors, alpha=0.7)
    axes[0].axhline(0, color='k', linestyle='--')
    axes[0].set_xlabel('Regions (sorted)', fontweight='bold')
    axes[0].set_ylabel("Cohen's d Improvement (Synth - ASL)", fontweight='bold')
    axes[0].set_title('A. Synthetic FDG Improvement Over ASL', fontweight='bold')
    axes[0].set_xticks([])
    n_improved = np.sum(df_sorted['Synth_vs_ASL'] > 0)
    axes[0].text(0.95, 0.95, f'Improved: {n_improved}/{len(df_sorted)}',
                 transform=axes[0].transAxes, fontsize=12, ha='right', va='top')
    sns.despine(ax=axes[0])

    # Panel B: Comparison of improvements (using absolute values for magnitude comparison)
    comparisons = ['Synthetic > ASL', 'Real FDG > ASL', 'Synthetic > Real']
    percentages = [
        np.sum(df['Synth_vs_ASL'] > 0) / len(df) * 100,
        np.sum(df['Real_vs_ASL'] > 0) / len(df) * 100,
        np.sum(np.abs(df['Cohens_d_Synthetic_FDG']) > np.abs(df['Cohens_d_Real_FDG'])) / len(df) * 100
    ]
    colors = ['#2ca02c', '#1f77b4', '#9467bd']

    bars = axes[1].bar(comparisons, percentages, color=colors, edgecolor='black', alpha=0.7)
    axes[1].axhline(50, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_ylabel('Percentage of Regions (%)', fontweight='bold')
    axes[1].set_title('B. Pairwise Comparisons', fontweight='bold')
    axes[1].set_ylim([0, 100])

    for bar, pct in zip(bars, percentages):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                     f'{pct:.0f}%', ha='center', fontsize=12, fontweight='bold')
    sns.despine(ax=axes[1])

    fig.suptitle(f'{dataset_name} - ASL Improvement Analysis ({atlas})', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_cohens_d_scatter_comparison(df_lateralization, dataset_name='TLE', atlas='DKT'):
    """
    Create a scatter plot comparing Cohen's d differences:
    X-axis: Synthetic FDG - ASL (positive = Synthetic better than ASL)
    Y-axis: Real FDG - ASL (positive = Real better than ASL)
    
    Interpretation:
    - Points near diagonal: Synthetic FDG performs similarly to Real FDG (both better/worse than ASL)
    - Points in upper-right: Both Real and Synthetic better than ASL
    - Points in lower-left: Both Real and Synthetic worse than ASL
    - Points above diagonal: Real FDG outperforms Synthetic FDG (relative to ASL)
    - Points below diagonal: Synthetic FDG outperforms Real FDG (relative to ASL)
    """
    # Extract Cohen's d values
    regions = df_lateralization['Region'].values
    real_d = df_lateralization['Cohens_d_Real_FDG'].values.astype(float)
    synth_d = df_lateralization['Cohens_d_Synthetic_FDG'].values.astype(float)
    asl_d = df_lateralization['Cohens_d_ASL'].values.astype(float)
    
    # Filter out NaN values
    valid_mask = ~(np.isnan(real_d) | np.isnan(synth_d) | np.isnan(asl_d))
    regions = regions[valid_mask]
    real_d = real_d[valid_mask]
    synth_d = synth_d[valid_mask]
    asl_d = asl_d[valid_mask]
    
    # Compute differences from ASL
    x = np.abs(synth_d) - np.abs(asl_d)  # Synthetic - ASL
    y = np.abs(real_d) - np.abs(asl_d)   # Real - ASL
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define quadrant colors
    ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    # Add diagonal (where Synthetic = Real relative to ASL)
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.2
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1.5, label='Synthetic = Real (vs ASL)')
    
    # Color points by quadrant
    colors = []
    for xi, yi in zip(x, y):
        if xi > 0 and yi > 0:
            colors.append('#2ca02c')  # Green: Both better than ASL
        elif xi < 0 and yi < 0:
            colors.append('#d62728')  # Red: Both worse than ASL
        elif xi > 0 and yi < 0:
            colors.append('#ff7f0e')  # Orange: Synthetic better, Real worse
        else:
            colors.append('#1f77b4')  # Blue: Real better, Synthetic worse
    
    # Scatter plot
    scatter = ax.scatter(x, y, c=colors, s=100, alpha=0.7, edgecolors='black', linewidths=1)
    
    # Add region labels for extreme points
    for i, region in enumerate(regions):
        # Label points that are far from origin or diagonal
        dist_from_origin = np.sqrt(x[i]**2 + y[i]**2)
        dist_from_diagonal = np.abs(x[i] - y[i]) / np.sqrt(2)
        
        if dist_from_origin > np.percentile(np.sqrt(x**2 + y**2), 75) or dist_from_diagonal > np.percentile(np.abs(x - y) / np.sqrt(2), 75):
            ax.annotate(region, (x[i], y[i]), fontsize=8, alpha=0.8,
                       xytext=(5, 5), textcoords='offset points')
    
    # Add quadrant labels
    ax.text(lim*0.7, lim*0.7, 'Both > ASL', fontsize=12, color='#2ca02c',
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-lim*0.7, -lim*0.7, 'Both < ASL', fontsize=12, color='#d62728',
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(lim*0.7, -lim*0.5, 'Synthetic > ASL\nReal < ASL', fontsize=10, color='#ff7f0e',
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-lim*0.7, lim*0.5, 'Real > ASL\nSynthetic < ASL', fontsize=10, color='#1f77b4',
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    ax.set_xlabel("Cohen's d: Synthetic FDG - ASL", fontweight='bold', fontsize=12)
    ax.set_ylabel("Cohen's d: Real FDG - ASL", fontweight='bold', fontsize=12)
    ax.set_title(f"Effect Size Comparison for TLE Lateralization\n({atlas} Atlas, {len(regions)} regions)",
                 fontweight='bold', fontsize=14)
    
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_aspect('equal')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=10, label='Both > ASL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#d62728', markersize=10, label='Both < ASL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Real > ASL, Synthetic < ASL'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Synthetic > ASL, Real < ASL'),
        Line2D([0], [0], linestyle='--', color='black', alpha=0.5, label='Synthetic = Real (vs ASL)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
    
    # Add correlation
    r, p = stats.pearsonr(x, y)
    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3f}', transform=ax.transAxes,
            fontsize=11, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Count quadrants
    n_both_better = np.sum((x > 0) & (y > 0))
    n_both_worse = np.sum((x < 0) & (y < 0))
    n_real_better = np.sum((x < 0) & (y > 0))
    n_synth_better = np.sum((x > 0) & (y < 0))
    
    summary_text = f"Quadrant counts:\n  Both > ASL: {n_both_better}\n  Both < ASL: {n_both_worse}\n  Real only > ASL: {n_real_better}\n  Synth only > ASL: {n_synth_better}"
    ax.text(0.05, 0.15, summary_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    sns.despine(ax=ax)
    plt.tight_layout()
    
    # Print interpretation
    print("="*70)
    print("INTERPRETATION")
    print("="*70)
    print(f"\nCorrelation between Real-ASL and Synthetic-ASL differences: r = {r:.2f}")
    print(f"\nQuadrant Analysis:")
    print(f"  Upper-Right (Both > ASL): {n_both_better} regions ({n_both_better/len(regions)*100:.0f}%)")
    print(f"  Lower-Left (Both < ASL): {n_both_worse} regions ({n_both_worse/len(regions)*100:.0f}%)")
    print(f"  Upper-Left (Real > ASL, Synthetic < ASL): {n_real_better} regions ({n_real_better/len(regions)*100:.0f}%)")
    print(f"  Lower-Right (Synthetic > ASL, Real < ASL): {n_synth_better} regions ({n_synth_better/len(regions)*100:.0f}%)")
    
    print(f"\nKey Insight:")
    if r > 0.5:
        print(f"  Strong positive correlation (r={r:.2f}) indicates that Synthetic FDG")
        print(f"  captures similar regional patterns as Real FDG relative to ASL.")
    if n_real_better > n_synth_better:
        print(f"  However, {n_real_better} regions show Real FDG > ASL but Synthetic FDG < ASL,")
        print(f"  indicating Synthetic FDG loses discriminative power in these regions.")
    
    # Points above diagonal = Real better than Synthetic
    above_diagonal = np.sum(y > x)
    below_diagonal = np.sum(y < x)
    print(f"\n  Regions where Real > Synthetic (above diagonal): {above_diagonal} ({above_diagonal/len(regions)*100:.0f}%)")
    print(f"  Regions where Synthetic > Real (below diagonal): {below_diagonal} ({below_diagonal/len(regions)*100:.0f}%)")
    
    return fig, x, y, regions


# ============================================================================
# Build Asymmetry DataFrame
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


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(dataset='TLE', atlas='DKT'):
    """Run lateralization capacity analysis for specified dataset."""
    print(f"\n{'='*60}")
    print(f"Running Lateralization Capacity Analysis - {dataset} ({atlas} Atlas)")
    print(f"{'='*60}\n")

    if dataset == 'TLE':
        # Load TLE data - select pickle file based on atlas
        if atlas == 'HarvardOxford':
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_ho.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_ho.pkl')
        else:
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')

        if not os.path.exists(pkl_path):
            print(f"Error: Data file not found at {pkl_path}")
            print(f"Run 00_prepare_data.py first to generate the data files.")
            return None, None

        with open(pkl_path, 'rb') as f:
            df_merged = pickle.load(f)

        pet_subjects = df_merged['subject'].unique()
        print(f"Loaded {len(pet_subjects)} TLE subjects with PET data")

        # Load clinical metadata for well-lateralized TLE subjects only
        print("\nFiltering for well-lateralized TLE subjects with temporal onset:")
        df_metadata = load_clinical_metadata(pet_subject_ids=list(pet_subjects))
        print(f"\nFinal cohort: {len(df_metadata)} well-lateralized TLE patients")
        print(f"  L-TLE: {np.sum(df_metadata['isLeft'] == 1)}")
        print(f"  R-TLE: {np.sum(df_metadata['isLeft'] == 0)}")

        # Build asymmetry DataFrame
        df_ai = build_asymmetry_dataframe(df_merged)

        # Compute lateralization capacity
        df_lateralization = compute_lateralization_capacity(df_ai, df_metadata)
        print(f"\nComputed Cohen's d for {len(df_lateralization)} regions")

        # Summary statistics
        print("\nSummary Statistics:")
        print(f"  Mean |d| Real FDG: {np.mean(np.abs(df_lateralization['Cohens_d_Real_FDG'])):.3f}")
        print(f"  Mean |d| Synthetic FDG: {np.mean(np.abs(df_lateralization['Cohens_d_Synthetic_FDG'])):.3f}")
        print(f"  Mean |d| ASL: {np.mean(np.abs(df_lateralization['Cohens_d_ASL'])):.3f}")

        # Save table
        save_table(df_lateralization, f'lateralization_cohens_d_TLE_{atlas}', TABLES_DIR)

        # Create figures
        # Main Cohen's d quadrant scatterplot
        fig_quad = plot_cohens_d_scatterplot_quadrant(df_lateralization, 'TLE', atlas)
        save_figure(fig_quad, f'cohens_d_scatterplot_quadrant_TLE_{atlas}', FIGURES_DIR)
        plt.close(fig_quad)

        # Three-way comparison
        fig_three = plot_cohens_d_three_way_comparison(df_lateralization, 'TLE', atlas)
        save_figure(fig_three, f'cohens_d_three_way_TLE_{atlas}', FIGURES_DIR)
        plt.close(fig_three)

        # Bar comparison for key regions
        fig_bar = plot_cohens_d_bar_comparison(df_lateralization, 'TLE', atlas)
        if fig_bar is not None:
            save_figure(fig_bar, f'cohens_d_bar_comparison_TLE_{atlas}', FIGURES_DIR)
            plt.close(fig_bar)

        # Improvement over ASL
        fig_improve = plot_improvement_over_asl(df_lateralization, 'TLE', atlas)
        save_figure(fig_improve, f'improvement_over_asl_TLE_{atlas}', FIGURES_DIR)
        plt.close(fig_improve)

        # Scatter comparison plot
        fig_scatter_comp, _, _, _ = plot_cohens_d_scatter_comparison(df_lateralization, 'TLE', atlas)
        save_figure(fig_scatter_comp, f'cohens_d_scatter_comparison_TLE_{atlas}', FIGURES_DIR)
        plt.close(fig_scatter_comp)

        plt.close('all')

        return df_lateralization, df_ai

    else:  # MCI
        # Load MCI data - select pickle file based on atlas
        if atlas == 'HarvardOxford':
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci_ho.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci_ho.pkl')
        else:
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')
            if not os.path.exists(pkl_path):
                pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')

        if not os.path.exists(pkl_path):
            print(f"Error: Data file not found at {pkl_path}")
            print(f"Run 00_prepare_data.py first to generate the data files.")
            return None, None

        with open(pkl_path, 'rb') as f:
            df_merged = pickle.load(f)

        print(f"Loaded {len(df_merged['subject'].unique())} MCI subjects")

        # Load MCI metadata
        df_metadata = load_mci_metadata()
        print(df_metadata.head())
        print(f"MCI metadata: {len(df_metadata)} subjects")
        print(f"  HC: {np.sum(df_metadata['is_mci'] == 0)}")
        print(f"  MCI: {np.sum(df_metadata['is_mci'] == 1)}")

        # Build SUVR DataFrame
        df_suvr = build_suvr_dataframe_mci(df_merged, df_metadata, atlas=atlas)
        print(f"Built SUVR DataFrame with {len(df_suvr)} subjects")

        if len(df_suvr) == 0:
            print(f"Warning: No SUVR data available for MCI dataset")
            print(f"Skipping MCI analysis")
            return None, None

        # Compute discriminability (Cohen's d for MCI vs HC)
        df_discriminability = compute_mci_discriminability(df_suvr, df_metadata, atlas=atlas)
        print(f"\nComputed Cohen's d for {len(df_discriminability)} regions")

        # Summary statistics
        print("\nSummary Statistics:")
        print(f"  Mean |d| Real FDG: {np.mean(np.abs(df_discriminability['Cohens_d_Real_FDG'])):.3f}")
        print(f"  Mean |d| Synthetic FDG: {np.mean(np.abs(df_discriminability['Cohens_d_Synthetic_FDG'])):.3f}")
        print(f"  Mean |d| ASL: {np.mean(np.abs(df_discriminability['Cohens_d_ASL'])):.3f}")

        # Save table
        save_table(df_discriminability, f'lateralization_cohens_d_MCI_{atlas}', TABLES_DIR)

        # Create figures (adapted for MCI)
        # Main Cohen's d quadrant scatterplot
        fig_quad = plot_cohens_d_scatterplot_quadrant(df_discriminability, 'MCI', atlas)
        save_figure(fig_quad, f'cohens_d_scatterplot_quadrant_MCI_{atlas}', FIGURES_DIR)
        plt.close(fig_quad)

        # Three-way comparison
        fig_three = plot_cohens_d_three_way_comparison(df_discriminability, 'MCI', atlas)
        save_figure(fig_three, f'cohens_d_three_way_MCI_{atlas}', FIGURES_DIR)
        plt.close(fig_three)

        # Bar comparison for key regions
        fig_bar = plot_cohens_d_bar_comparison(df_discriminability, 'MCI', atlas)
        if fig_bar is not None:
            save_figure(fig_bar, f'cohens_d_bar_comparison_MCI_{atlas}', FIGURES_DIR)
            plt.close(fig_bar)

        # Improvement over ASL
        fig_improve = plot_improvement_over_asl(df_discriminability, 'MCI', atlas)
        save_figure(fig_improve, f'improvement_over_asl_MCI_{atlas}', FIGURES_DIR)
        plt.close(fig_improve)

        # Scatter comparison plot
        fig_scatter_comp, _, _, _ = plot_cohens_d_scatter_comparison(df_discriminability, 'MCI', atlas)
        save_figure(fig_scatter_comp, f'cohens_d_scatter_comparison_MCI_{atlas}', FIGURES_DIR)
        plt.close(fig_scatter_comp)

        plt.close('all')

        return df_discriminability, df_suvr


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Lateralization Capacity Analysis (Cohen\'s d)')
    parser.add_argument('--atlas', type=str, default='DKT', choices=['DKT', 'HarvardOxford'],
                        help='Atlas to use (default: DKT). HarvardOxford support is experimental.')
    parser.add_argument('--include-ho', action='store_true',
                        help='Also run analysis with Harvard-Oxford atlas in addition to DKT')
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    atlases_to_run = [args.atlas]
    if args.include_ho and 'HarvardOxford' not in atlases_to_run:
        atlases_to_run.append('HarvardOxford')

    for atlas in atlases_to_run:
        # Run for TLE
        print("\n" + "=" * 70)
        print(f"TLE - {atlas} ATLAS")
        print("=" * 70)
        df_lat_tle, df_ai_tle = run_analysis('TLE', atlas)

        # Run for MCI
        print("\n" + "=" * 70)
        print(f"MCI - {atlas} ATLAS")
        print("=" * 70)
        df_lat_mci, df_suvr_mci = run_analysis('MCI', atlas)

    print("\n" + "=" * 70)
    print("Lateralization Capacity Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 70)
