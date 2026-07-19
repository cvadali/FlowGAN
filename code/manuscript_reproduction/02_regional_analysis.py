"""
Script 02: Part 2 - Regional Analysis

This script performs regional analysis including:
- Within-subject correlation across brain regions
- Across-subject correlation for each region
- Asymmetry Index (AI) analysis
- Bland-Altman and correlation plots

Works for both TLE and MCI datasets, with DKT and Harvard-Oxford atlases.
"""

import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from statannotations.Annotator import Annotator

from utils import (
    bland_altman_and_corr_plot, analyze_and_print, apply_benjamini_hochberg,
    save_figure, save_table, BOXPLOT_PARAMS, COLORS
)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures', '02_regional_analysis')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'tables', '02_regional_analysis')
DKT_CSV = os.path.join(SCRIPT_DIR, 'data', 'dkt.csv')

# Harvard-Oxford atlas paths (only needed to rebuild pickles from raw imaging; not shipped)
HO_ATLAS_PATH = os.path.join(SCRIPT_DIR, 'data', 'HO_MNI_resliced.nii.gz')
SOURCE_MNI = None  # not included in reviewer package (requires original imaging data)

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']


# ============================================================================
# Data Loading
# ============================================================================

def load_dkt_atlas():
    """Load DKT atlas."""
    dkt = pd.read_csv(DKT_CSV)
    region_names = []
    sides = []
    for region in dkt['atlas_region']:
        if len(region.split('-')) == 3:
            side = 'Left' if region.split('-')[1] == 'lh' else 'Right'
            region_name = region.split('-')[2]
        elif (len(region.split('-')) == 2) and (('Left' in region) or ('Right' in region)):
            side = region.split('-')[0]
            region_name = region.split('-')[1]
        else:
            region_name = region
            side = 'Mid'
        region_names.append(region_name)
        sides.append(side)
    dkt['side'] = sides
    dkt['region_name'] = region_names
    return dkt


def get_putamen_normalization_values(df_merged):
    """Extract putamen values for normalization."""
    left_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')][['value_pet_original', 'subject']]
    right_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')][['value_pet_original', 'subject']]
    left_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')][['value_pet_recon', 'subject']]
    right_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')][['value_pet_recon', 'subject']]
    left_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')][['value_asl', 'subject']]
    right_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')][['value_asl', 'subject']]
    return left_og, right_og, left_recon, right_recon, left_asl, right_asl


# ============================================================================
# Within-Subject Analysis (across regions)
# ============================================================================

def within_subject_analysis(df_merged, putamen_vals, analysis_type='SUVR'):
    """
    For a given subject, compute correlation of SUVR/AI between modalities across all regions.
    """
    left_og, right_og, left_recon, right_recon, left_asl, right_asl = putamen_vals
    subjects = df_merged['subject'].unique()

    recon_r = []
    asl_r = []
    recon_bias = []
    asl_bias = []

    for sub in subjects:
        df_subject = df_merged[df_merged['subject'] == sub].copy()

        try:
            norm_og = (left_og[left_og['subject'] == sub]['value_pet_original'].values[0] +
                       right_og[right_og['subject'] == sub]['value_pet_original'].values[0])
            norm_recon = (left_recon[left_recon['subject'] == sub]['value_pet_recon'].values[0] +
                          right_recon[right_recon['subject'] == sub]['value_pet_recon'].values[0])
            norm_asl = (left_asl[left_asl['subject'] == sub]['value_asl'].values[0] +
                        right_asl[right_asl['subject'] == sub]['value_asl'].values[0])

            if norm_og == 0 or norm_recon == 0 or norm_asl == 0:
                continue

            df_subject['value_pet_original'] = df_subject['value_pet_original'] / norm_og
            df_subject['value_pet_recon'] = df_subject['value_pet_recon'] / norm_recon
            df_subject['value_asl'] = df_subject['value_asl'] / norm_asl

            if analysis_type == 'SUVR':
                x, y = df_subject.dropna()['value_pet_original'], df_subject.dropna()['value_pet_recon']
                ba = bland_altman_and_corr_plot(x, y, stats_only=True)
                recon_r.append(ba['spearman_r'])
                recon_bias.append(ba['bias'])

                x, y = df_subject.dropna()['value_pet_original'], df_subject.dropna()['value_asl']
                ba = bland_altman_and_corr_plot(x, y, stats_only=True)
                asl_r.append(ba['spearman_r'])
                asl_bias.append(ba['bias'])

            else:  # Asymmetry
                left_vals = df_subject[df_subject['side'] == 'Left']
                right_vals = df_subject[df_subject['side'] == 'Right']

                # Get matching regions
                regions = [r for r in left_vals.region_name.values
                           if r in right_vals.region_name.values and r not in EXCLUDE_REGIONS]

                left_vals = left_vals.set_index('region_name').loc[regions].reset_index()
                right_vals = right_vals.set_index('region_name').loc[regions].reset_index()

                ai_og = (left_vals['value_pet_original'].values - right_vals['value_pet_original'].values) / \
                        (left_vals['value_pet_original'].values + right_vals['value_pet_original'].values)
                ai_recon = (left_vals['value_pet_recon'].values - right_vals['value_pet_recon'].values) / \
                           (left_vals['value_pet_recon'].values + right_vals['value_pet_recon'].values)
                ai_asl = (left_vals['value_asl'].values - right_vals['value_asl'].values) / \
                         (left_vals['value_asl'].values + right_vals['value_asl'].values)

                ba = bland_altman_and_corr_plot(ai_og, ai_recon, stats_only=True)
                recon_r.append(ba['spearman_r'])
                recon_bias.append(ba['bias'])

                ba = bland_altman_and_corr_plot(ai_og, ai_asl, stats_only=True)
                asl_r.append(ba['spearman_r'])
                asl_bias.append(ba['bias'])

        except Exception as e:
            continue

    return {
        'recon_r': recon_r, 'asl_r': asl_r,
        'recon_bias': recon_bias, 'asl_bias': asl_bias
    }


# ============================================================================
# Across-Subject Analysis (for each region)
# ============================================================================

def across_subject_analysis(df_merged, putamen_vals, analysis_type='SUVR'):
    """
    For a given region, compute correlation between modalities across subjects.
    """
    left_og, right_og, left_recon, right_recon, left_asl, right_asl = putamen_vals

    regions = sorted(set(df_merged['region_name']))
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    rows = []

    if analysis_type == 'SUVR':
        # Build SUVR contrast DataFrame
        df_contrast = pd.DataFrame()
        for region in regions:
            if df_merged[df_merged['region_name'] == region]['side'].values[0] == 'Mid':
                continue

            left_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Left')]
            right_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Right')]

            if len(left_df) != len(right_df):
                continue

            suvr_pet = (left_df['value_pet_original'].values + right_df['value_pet_original'].values) / \
                       (left_og['value_pet_original'].values + right_og['value_pet_original'].values)
            suvr_recon = (left_df['value_pet_recon'].values + right_df['value_pet_recon'].values) / \
                         (left_recon['value_pet_recon'].values + right_recon['value_pet_recon'].values)
            rcbf_asl = (left_df['value_asl'].values + right_df['value_asl'].values) / \
                       (left_asl['value_asl'].values + right_asl['value_asl'].values)

            region_df = pd.DataFrame({
                'Subject': left_df['subject'].values,
                'Region': region,
                'PET SUVR Original': suvr_pet,
                'PET SUVR FlowGAN': suvr_recon,
                'ASL rCBF': rcbf_asl
            })
            df_contrast = pd.concat([df_contrast, region_df])

        df_contrast = df_contrast.reset_index(drop=True)

        for region in df_contrast['Region'].unique():
            try:
                x = df_contrast.loc[df_contrast['Region'] == region, 'PET SUVR Original']
                y = df_contrast.loc[df_contrast['Region'] == region, 'PET SUVR FlowGAN']
                ba_recon = bland_altman_and_corr_plot(x, y, stats_only=True)

                x = df_contrast.loc[df_contrast['Region'] == region, 'PET SUVR Original']
                y = df_contrast.loc[df_contrast['Region'] == region, 'ASL rCBF']
                ba_asl = bland_altman_and_corr_plot(x, y, stats_only=True)

                rows.append({
                    'Region': region,
                    'Spearman_r_FlowGAN': ba_recon['spearman_r'],
                    'Bias_FlowGAN': ba_recon['bias'],
                    'Spearman_r_ASL': ba_asl['spearman_r'],
                    'Bias_ASL': ba_asl['bias']
                })
            except Exception:
                continue

    else:  # Asymmetry
        df_ai = pd.DataFrame()
        for region in regions:
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

        df_ai = df_ai.reset_index(drop=True)

        for region in df_ai['Region'].unique():
            try:
                x = df_ai.loc[df_ai['Region'] == region, 'PET AI Original']
                y = df_ai.loc[df_ai['Region'] == region, 'PET AI Recon']
                ba_recon = bland_altman_and_corr_plot(x, y, stats_only=True)

                x = df_ai.loc[df_ai['Region'] == region, 'PET AI Original']
                y = df_ai.loc[df_ai['Region'] == region, 'ASL AI']
                ba_asl = bland_altman_and_corr_plot(x, y, stats_only=True)

                rows.append({
                    'Region': region,
                    'Spearman_r_FlowGAN': ba_recon['spearman_r'],
                    'Bias_FlowGAN': ba_recon['bias'],
                    'Spearman_r_ASL': ba_asl['spearman_r'],
                    'Bias_ASL': ba_asl['bias']
                })
            except Exception:
                continue

    return pd.DataFrame(rows)


# ============================================================================
# Asymmetry Index DataFrame Builder
# ============================================================================

def build_asymmetry_dataframe(df_merged):
    """Build asymmetry index DataFrame for all regions."""
    regions = sorted(set(df_merged['region_name']))
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    df_ai = pd.DataFrame()
    for region in regions:
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

    return df_ai.reset_index(drop=True)


# ============================================================================
# SUVR DataFrame Builder
# ============================================================================

def build_suvr_dataframe(df_merged, putamen_vals):
    """Build SUVR DataFrame for all regions (bilateral sum normalized by putamen)."""
    left_og, right_og, left_recon, right_recon, left_asl, right_asl = putamen_vals

    regions = sorted(set(df_merged['region_name']))
    regions = [r for r in regions if r not in EXCLUDE_REGIONS]

    df_suvr = pd.DataFrame()
    for region in regions:
        if df_merged[df_merged['region_name'] == region]['side'].values[0] == 'Mid':
            continue

        left_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Left')]
        right_df = df_merged[(df_merged['region_name'] == region) & (df_merged['side'] == 'Right')]

        if len(left_df) != len(right_df):
            continue

        # Bilateral SUVR normalized by putamen
        suvr_pet = (left_df['value_pet_original'].values + right_df['value_pet_original'].values) / \
                   (left_og['value_pet_original'].values + right_og['value_pet_original'].values)
        suvr_recon = (left_df['value_pet_recon'].values + right_df['value_pet_recon'].values) / \
                     (left_recon['value_pet_recon'].values + right_recon['value_pet_recon'].values)
        suvr_asl = (left_df['value_asl'].values + right_df['value_asl'].values) / \
                   (left_asl['value_asl'].values + right_asl['value_asl'].values)

        region_df = pd.DataFrame({
            'Subject': left_df['subject'].values,
            'Region': region,
            'PET SUVR Original': suvr_pet,
            'PET SUVR FlowGAN': suvr_recon,
            'ASL rCBF': suvr_asl
        })
        df_suvr = pd.concat([df_suvr, region_df])

    return df_suvr.reset_index(drop=True)


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_within_subject_comparison(results, analysis_type, dataset_name):
    """Plot within-subject correlation comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # Spearman r
    df_plot = pd.DataFrame({
        'Value': results['recon_r'] + results['asl_r'],
        'Comparison': ['Synthetic PET'] * len(results['recon_r']) + ['ASL'] * len(results['asl_r'])
    })
    sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=axes[0], **BOXPLOT_PARAMS)
    axes[0].set_title('Spearman\'s r', fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('Spearman\'s r')
    sns.despine(ax=axes[0])

    # Add statistical annotation
    annotator = Annotator(axes[0], x='Comparison', y='Value', data=df_plot,
                      pairs=[('Synthetic PET', 'ASL')])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # Bias
    df_plot = pd.DataFrame({
        'Value': results['recon_bias'] + results['asl_bias'],
        'Comparison': ['Synthetic PET'] * len(results['recon_bias']) + ['ASL'] * len(results['asl_bias'])
    })
    sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=axes[1], **BOXPLOT_PARAMS)
    axes[1].set_title('Bias', fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('Bias')
    sns.despine(ax=axes[1])

    # Add statistical annotation
    annotator = Annotator(axes[1], x='Comparison', y='Value', data=df_plot,
                      pairs=[('Synthetic PET', 'ASL')])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    fig.suptitle(f'{dataset_name} - Within-Subject {analysis_type} Comparison', fontweight='bold')
    plt.tight_layout()

    return fig


def plot_correlation_difference_bar(results_df, analysis_type, dataset_name):
    """Plot bar chart showing correlation difference per region."""
    results_df = results_df.copy()
    results_df['Corr_Diff'] = results_df['Spearman_r_FlowGAN'] - results_df['Spearman_r_ASL']
    results_df = results_df.sort_values('Corr_Diff')

    threshold = np.nanstd(np.abs(results_df['Corr_Diff']))

    colors = []
    for diff in results_df['Corr_Diff']:
        if diff > threshold:
            colors.append('lightblue')
        elif diff < -threshold:
            colors.append('lightcoral')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(np.arange(len(results_df)), results_df['Corr_Diff'], color=colors)
    ax.set_xlabel('Regions', fontweight='bold')
    ax.set_ylabel('Correlation Difference (Synthetic PET - ASL)', fontweight='bold')
    ax.set_title(f'{dataset_name} - {analysis_type} Correlation Difference by Region', fontweight='bold')
    ax.set_xticks(np.arange(len(results_df)))
    ax.set_xticklabels(results_df['Region'], rotation=90)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    sns.despine(ax=ax)
    plt.tight_layout()

    return fig


def plot_within_region_correlation_bias(results_df, analysis_type, dataset_name):
    """Plot within-region correlation and bias analysis for all regions.

    This creates a multi-panel figure showing:
    - Panel A: Spearman correlation comparison (Synthetic PET vs ASL)
    - Panel B: Bias comparison (Synthetic PET vs ASL)
    """
    results_df = results_df.copy()

    # Sort regions by Synthetic PET correlation for better visualization
    results_df = results_df.sort_values('Spearman_r_FlowGAN')

    fig, axes = plt.subplots(1, 2, figsize=(8, 5))

    # Panel A: Correlation comparison
    ax = axes[0]
    df_plot = pd.DataFrame({
        'Value': np.concatenate([results_df['Spearman_r_FlowGAN'].values, results_df['Spearman_r_ASL'].values]),
        'Comparison': ['Synthetic PET'] * len(results_df) + ['ASL'] * len(results_df)
    })
    sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=ax, **BOXPLOT_PARAMS)
    ax.set_title("Spearman's r", fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel("Spearman's r")
    sns.despine(ax=ax)

    # Add statistical annotation
    annotator = Annotator(ax, x='Comparison', y='Value', data=df_plot,
                      pairs=[('Synthetic PET', 'ASL')])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    # Panel B: Bias comparison
    ax = axes[1]
    df_plot = pd.DataFrame({
        'Value': np.concatenate([results_df['Bias_FlowGAN'].values, results_df['Bias_ASL'].values]),
        'Comparison': ['Synthetic PET'] * len(results_df) + ['ASL'] * len(results_df)
    })
    sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=ax, **BOXPLOT_PARAMS)
    ax.set_title('Bias', fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel('Bias')
    sns.despine(ax=ax)

    # Add statistical annotation
    annotator = Annotator(ax, x='Comparison', y='Value', data=df_plot,
                      pairs=[('Synthetic PET', 'ASL')])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    fig.suptitle(f'{dataset_name} - Within-Region {analysis_type} Analysis', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_bland_altman_selected_regions(df_ai, regions, dataset_name):
    """Plot Bland-Altman and correlation for selected regions."""
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 4, figsize=(16, 4 * n_regions))

    for i, region in enumerate(regions):
        df_region = df_ai[df_ai['Region'] == region]

        if len(df_region) == 0:
            continue

        # Real PET vs Synthetic PET - BA
        x = df_region['PET AI Original'].values
        y = df_region['PET AI Recon'].values
        diff = x - y
        axes[i, 0].scatter((x + y) / 2, diff, color=COLORS[0], alpha=0.7)
        axes[i, 0].axhline(np.mean(diff), color='k', linestyle='--')
        axes[i, 0].axhline(np.mean(diff) + 1.96 * np.std(diff), color='k', linestyle=':')
        axes[i, 0].axhline(np.mean(diff) - 1.96 * np.std(diff), color='k', linestyle=':')
        axes[i, 0].set_title(f'{region} - Real PET vs Synthetic PET (BA)')
        axes[i, 0].set_xlabel('Mean')
        axes[i, 0].set_ylabel('Difference')

        # Real PET vs Synthetic PET - Corr
        axes[i, 1].scatter(x, y, color=COLORS[0], alpha=0.7)
        lim = max(np.max(np.abs(x)), np.max(np.abs(y))) * 1.1
        axes[i, 1].plot([-lim, lim], [-lim, lim], 'k--')
        axes[i, 1].set_xlim(-lim, lim)
        axes[i, 1].set_ylim(-lim, lim)
        rho, _ = stats.spearmanr(x, y)
        axes[i, 1].set_title(f'Real PET vs Synthetic PET (r={rho:.2f})')
        axes[i, 1].set_xlabel('Real PET AI')
        axes[i, 1].set_ylabel('Synthetic PET AI')

        # Real PET vs ASL - BA
        y_asl = df_region['ASL AI'].values
        diff_asl = x - y_asl
        axes[i, 2].scatter((x + y_asl) / 2, diff_asl, color=COLORS[1], alpha=0.7)
        axes[i, 2].axhline(np.mean(diff_asl), color='k', linestyle='--')
        axes[i, 2].axhline(np.mean(diff_asl) + 1.96 * np.std(diff_asl), color='k', linestyle=':')
        axes[i, 2].axhline(np.mean(diff_asl) - 1.96 * np.std(diff_asl), color='k', linestyle=':')
        axes[i, 2].set_title(f'{region} - Real PET vs ASL (BA)')
        axes[i, 2].set_xlabel('Mean')
        axes[i, 2].set_ylabel('Difference')

        # Real PET vs ASL - Corr
        axes[i, 3].scatter(x, y_asl, color=COLORS[1], alpha=0.7)
        lim = max(np.max(np.abs(x)), np.max(np.abs(y_asl))) * 1.1
        axes[i, 3].plot([-lim, lim], [-lim, lim], 'k--')
        axes[i, 3].set_xlim(-lim, lim)
        axes[i, 3].set_ylim(-lim, lim)
        rho, _ = stats.spearmanr(x, y_asl)
        axes[i, 3].set_title(f'Real PET vs ASL (r={rho:.2f})')
        axes[i, 3].set_xlabel('Real PET AI')
        axes[i, 3].set_ylabel('ASL AI')

    for ax in axes.flatten():
        sns.despine(ax=ax)

    fig.suptitle(f'{dataset_name} - Bland-Altman & Correlation (Asymmetry)', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


def plot_bland_altman_suvr_selected_regions(df_suvr, regions, dataset_name):
    """Plot Bland-Altman and correlation for SUVR values in selected regions."""
    n_regions = len(regions)
    fig, axes = plt.subplots(n_regions, 4, figsize=(16, 4 * n_regions))

    if n_regions == 1:
        axes = axes.reshape(1, -1)

    for i, region in enumerate(regions):
        df_region = df_suvr[df_suvr['Region'] == region]

        if len(df_region) == 0:
            continue

        # Real PET vs Synthetic PET - BA
        x = df_region['PET SUVR Original'].values
        y = df_region['PET SUVR FlowGAN'].values
        valid = np.isfinite(x) & np.isfinite(y)
        x_valid, y_valid = x[valid], y[valid]

        axes[i, 0].scatter((x_valid + y_valid) / 2, x_valid - y_valid, color=COLORS[0], alpha=0.7)
        axes[i, 0].axhline(np.mean(x_valid - y_valid), color='k', linestyle='--')
        axes[i, 0].axhline(np.mean(x_valid - y_valid) + 1.96 * np.std(x_valid - y_valid), color='k', linestyle=':')
        axes[i, 0].axhline(np.mean(x_valid - y_valid) - 1.96 * np.std(x_valid - y_valid), color='k', linestyle=':')
        axes[i, 0].set_title(f'{region} - Real PET vs Synthetic PET (BA)')
        axes[i, 0].set_xlabel('Mean SUVR')
        axes[i, 0].set_ylabel('Difference')

        # Real PET vs Synthetic PET - Corr
        axes[i, 1].scatter(x_valid, y_valid, color=COLORS[0], alpha=0.7)
        lim_min = min(np.min(x_valid), np.min(y_valid)) * 0.9
        lim_max = max(np.max(x_valid), np.max(y_valid)) * 1.1
        axes[i, 1].plot([lim_min, lim_max], [lim_min, lim_max], 'k--')
        axes[i, 1].set_xlim(lim_min, lim_max)
        axes[i, 1].set_ylim(lim_min, lim_max)
        rho, _ = stats.spearmanr(x_valid, y_valid)
        axes[i, 1].set_title(f'Real PET vs Synthetic PET (r={rho:.2f})')
        axes[i, 1].set_xlabel('Real PET SUVR')
        axes[i, 1].set_ylabel('Synthetic PET SUVR')

        # Real PET vs ASL - BA
        y_asl = df_region['ASL rCBF'].values
        valid_asl = np.isfinite(x) & np.isfinite(y_asl)
        x_asl_valid, y_asl_valid = x[valid_asl], y_asl[valid_asl]

        axes[i, 2].scatter((x_asl_valid + y_asl_valid) / 2, x_asl_valid - y_asl_valid, color=COLORS[1], alpha=0.7)
        axes[i, 2].axhline(np.mean(x_asl_valid - y_asl_valid), color='k', linestyle='--')
        axes[i, 2].axhline(np.mean(x_asl_valid - y_asl_valid) + 1.96 * np.std(x_asl_valid - y_asl_valid), color='k', linestyle=':')
        axes[i, 2].axhline(np.mean(x_asl_valid - y_asl_valid) - 1.96 * np.std(x_asl_valid - y_asl_valid), color='k', linestyle=':')
        axes[i, 2].set_title(f'{region} - Real PET vs ASL (BA)')
        axes[i, 2].set_xlabel('Mean')
        axes[i, 2].set_ylabel('Difference')

        # Real PET vs ASL - Corr
        axes[i, 3].scatter(x_asl_valid, y_asl_valid, color=COLORS[1], alpha=0.7)
        lim_min = min(np.min(x_asl_valid), np.min(y_asl_valid)) * 0.9
        lim_max = max(np.max(x_asl_valid), np.max(y_asl_valid)) * 1.1
        axes[i, 3].plot([lim_min, lim_max], [lim_min, lim_max], 'k--')
        axes[i, 3].set_xlim(lim_min, lim_max)
        axes[i, 3].set_ylim(lim_min, lim_max)
        rho, _ = stats.spearmanr(x_asl_valid, y_asl_valid)
        axes[i, 3].set_title(f'Real PET vs ASL (r={rho:.2f})')
        axes[i, 3].set_xlabel('Real PET SUVR')
        axes[i, 3].set_ylabel('ASL rCBF')

    for ax in axes.flatten():
        sns.despine(ax=ax)

    fig.suptitle(f'{dataset_name} - Bland-Altman & Correlation (SUVR)', fontweight='bold', y=1.02)
    plt.tight_layout()

    return fig


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(dataset='TLE', atlas='DKT'):
    """Run regional analysis for specified dataset and atlas."""
    print(f"\n{'='*60}")
    print(f"Running Regional Analysis - {dataset} ({atlas} Atlas)")
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

    # Get putamen normalization values
    putamen_vals = get_putamen_normalization_values(df_merged)

    # Build asymmetry DataFrame
    df_ai = build_asymmetry_dataframe(df_merged)
    print(f"Built asymmetry DataFrame with {len(df_ai)} rows")

    # Build SUVR DataFrame
    df_suvr = build_suvr_dataframe(df_merged, putamen_vals)
    print(f"Built SUVR DataFrame with {len(df_suvr)} rows")

    results = {}

    # Within-subject analysis
    for analysis_type in ['SUVR', 'Asymmetry']:
        print(f"\n--- Within-Subject {analysis_type} Analysis ---")
        within_results = within_subject_analysis(df_merged, putamen_vals, analysis_type)

        analyze_and_print(within_results['recon_r'], within_results['asl_r'],
                          'FlowGAN', 'ASL', f'{analysis_type} Spearman r')

        fig = plot_within_subject_comparison(within_results, analysis_type, dataset)
        save_figure(fig, f'regional_within_subject_{analysis_type.lower()}_{dataset}_{atlas}', FIGURES_DIR)
        plt.close(fig)

        results[f'within_{analysis_type}'] = within_results

    # Across-subject analysis
    for analysis_type in ['SUVR', 'Asymmetry']:
        print(f"\n--- Across-Subject {analysis_type} Analysis ---")
        across_results = across_subject_analysis(df_merged, putamen_vals, analysis_type)

        if len(across_results) > 0:
            analyze_and_print(across_results['Spearman_r_FlowGAN'].values,
                              across_results['Spearman_r_ASL'].values,
                              'FlowGAN', 'ASL', f'{analysis_type} Spearman r')

            # Apply FDR correction if p-values available
            across_results['Corr_Diff'] = across_results['Spearman_r_FlowGAN'] - across_results['Spearman_r_ASL']

            save_table(across_results, f'regional_across_subject_{analysis_type.lower()}_{dataset}_{atlas}', TABLES_DIR)

            fig = plot_correlation_difference_bar(across_results, analysis_type, dataset)
            save_figure(fig, f'regional_corr_diff_bar_{analysis_type.lower()}_{dataset}_{atlas}', FIGURES_DIR)
            plt.close(fig)

            # Within-region correlation and bias analysis
            fig_region = plot_within_region_correlation_bias(across_results, analysis_type, dataset)
            save_figure(fig_region, f'regional_within_region_corr_bias_{analysis_type.lower()}_{dataset}_{atlas}', FIGURES_DIR)
            plt.close(fig_region)

            results[f'across_{analysis_type}'] = across_results

    # Bland-Altman plots for selected regions
    selected_regions = ['Hippocampus', 'insula', 'posteriorcingulate', 'Thalamus']

    # Asymmetry-based Bland-Altman plots
    available_regions_ai = [r for r in selected_regions if r in df_ai['Region'].unique()]
    if available_regions_ai:
        fig = plot_bland_altman_selected_regions(df_ai, available_regions_ai, dataset)
        save_figure(fig, f'regional_bland_altman_asymmetry_{dataset}_{atlas}', FIGURES_DIR)
        plt.close(fig)

    # SUVR-based Bland-Altman plots
    available_regions_suvr = [r for r in selected_regions if r in df_suvr['Region'].unique()]
    if available_regions_suvr:
        fig = plot_bland_altman_suvr_selected_regions(df_suvr, available_regions_suvr, dataset)
        save_figure(fig, f'regional_bland_altman_suvr_{dataset}_{atlas}', FIGURES_DIR)
        plt.close(fig)

    # Save asymmetry DataFrame for downstream analyses
    save_table(df_ai, f'asymmetry_index_{dataset}_{atlas}', TABLES_DIR)

    # Save SUVR DataFrame for downstream analyses
    save_table(df_suvr, f'suvr_values_{dataset}_{atlas}', TABLES_DIR)

    plt.close('all')

    return df_ai, df_suvr, results


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
    df_ai_tle, df_suvr_tle, results_tle = run_analysis('TLE', 'DKT')

    # Run for TLE with Harvard-Oxford
    print("\n" + "=" * 70)
    print("TLE - HARVARD-OXFORD ATLAS")
    print("=" * 70)
    df_ai_tle_ho, df_suvr_tle_ho, results_tle_ho = run_analysis('TLE', 'HarvardOxford')

    # Run for MCI with DKT
    print("\n" + "=" * 70)
    print("MCI - DKT ATLAS")
    print("=" * 70)
    df_ai_mci, df_suvr_mci, results_mci = run_analysis('MCI', 'DKT')

    # Run for MCI with Harvard-Oxford
    print("\n" + "=" * 70)
    print("MCI - HARVARD-OXFORD ATLAS")
    print("=" * 70)
    df_ai_mci_ho, df_suvr_mci_ho, results_mci_ho = run_analysis('MCI', 'HarvardOxford')

    print("\n" + "=" * 70)
    print("Regional Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 70)
