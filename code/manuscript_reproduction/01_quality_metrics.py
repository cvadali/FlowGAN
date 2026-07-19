"""
Script 01: Part 1 - Additional Quality Metrics and Plots

This script computes image quality metrics (SSIM, PSNR, RMSE, NCC) comparing:
- Real FDG-PET vs FlowGAN synthetic PET
- Real FDG-PET vs ASL

Works for both TLE and MCI datasets.
"""

import os
import pickle
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from statannotations.Annotator import Annotator

from utils import (
    get_quality_metrics, analyze_and_print, save_figure, save_table,
    BOXPLOT_PARAMS
)

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, 'figures', '01_quality_metrics')
TABLES_DIR = os.path.join(SCRIPT_DIR, 'tables', '01_quality_metrics')

# Paths to the raw imaging, taken from the environment.
#
# This script and 10_per_fold_quality_metrics.py are the only ones that read
# NIfTI volumes. That imaging is protected health information and is not part of
# this package, so these scripts cannot be run from the package alone; they are
# included for transparency about how the cached per-subject quality metrics in
# tables/10_per_fold_quality_metrics/ were produced. The notebook reads those
# cached CSVs, so nothing below is needed to reproduce any manuscript number.
#
# To run against your own data, point these at your directories:
#
#     export FLOWGAN_TLE_RECON_DIR=.../recon_niftis_smoothed_registered_to_original
#     export FLOWGAN_TLE_BIDS_DIR=.../BIDS_pet_t1_asl
#     export FLOWGAN_MCI_RECON_DIR=.../recon_niftis_smoothed_registered_to_original
#     export FLOWGAN_MCI_SOURCE_DIR=.../MCI_PCASL

# TLE paths
FLOWGAN_OUTPUTS = os.environ.get('FLOWGAN_TLE_RECON_DIR', '')
DATA_SOURCE_TLE = os.environ.get('FLOWGAN_TLE_BIDS_DIR', '')

# MCI paths
FLOWGAN_MCI_PATH = os.environ.get('FLOWGAN_MCI_RECON_DIR', '')
ORIGINAL_PET_PATH_MCI = os.environ.get('FLOWGAN_MCI_SOURCE_DIR', '')


def _require(path, var):
    """Fail with a useful message rather than an opaque path error."""
    if not path:
        raise SystemExit(
            f'{var} is not set.\n'
            'This script reads the raw imaging, which is not distributed with this '
            'package. Set the environment variables listed at the top of '
            '01_quality_metrics.py to point at your own data. The per-subject '
            'quality metrics used by the notebook are already cached in '
            'tables/10_per_fold_quality_metrics/, so you do not need to run this '
            'script to reproduce the manuscript.')
    return path


# ============================================================================
# Image Loading Functions
# ============================================================================

def load_flowgan_tle(sub):
    root = _require(FLOWGAN_OUTPUTS, 'FLOWGAN_TLE_RECON_DIR')
    return nib.load(os.path.join(root, sub + '_recon_pet.nii.gz')).get_fdata()


def load_pet_tle(sub):
    root = _require(DATA_SOURCE_TLE, 'FLOWGAN_TLE_BIDS_DIR')
    return nib.load(os.path.join(root, sub, sub + '_ses-clinical01_pet.nii.gz')).get_fdata()


def load_asl_tle(sub):
    root = _require(DATA_SOURCE_TLE, 'FLOWGAN_TLE_BIDS_DIR')
    return nib.load(os.path.join(root, sub, sub + '_ses-research3T_space-T1w_desc-gaussian-3.0_cbf.nii.gz')).get_fdata()


def load_flowgan_mci(sub):
    root = _require(FLOWGAN_MCI_PATH, 'FLOWGAN_MCI_RECON_DIR')
    return nib.load(os.path.join(root, sub + '_recon_pet.nii.gz')).get_fdata()


def load_pet_mci(sub):
    root = _require(ORIGINAL_PET_PATH_MCI, 'FLOWGAN_MCI_SOURCE_DIR')
    return nib.load(os.path.join(root, sub, 'derivatives', 'pet_registration_ants',
                                 sub + '_pet_antsregistered.nii.gz')).get_fdata()


def load_asl_mci(sub):
    root = _require(ORIGINAL_PET_PATH_MCI, 'FLOWGAN_MCI_SOURCE_DIR')
    return nib.load(os.path.join(root, sub, 'derivatives', 'pet_registration_ants',
                                 sub + '_cbf_registered.nii.gz')).get_fdata()


# ============================================================================
# Putamen Normalization
# ============================================================================

def get_putamen_normalization(df_merged, subject):
    """Get putamen normalization values for a subject."""
    left_putamen_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_putamen_og = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]

    left_putamen_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_putamen_recon = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]

    left_putamen_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Left')]
    right_putamen_asl = df_merged[(df_merged['region_name'] == 'Putamen') & (df_merged['side'] == 'Right')]

    try:
        norm_og = (left_putamen_og[left_putamen_og['subject'] == subject]['value_pet_original'].values[0] +
                   right_putamen_og[right_putamen_og['subject'] == subject]['value_pet_original'].values[0])
        norm_recon = (left_putamen_recon[left_putamen_recon['subject'] == subject]['value_pet_recon'].values[0] +
                      right_putamen_recon[right_putamen_recon['subject'] == subject]['value_pet_recon'].values[0])
        norm_asl = (left_putamen_asl[left_putamen_asl['subject'] == subject]['value_asl'].values[0] +
                    right_putamen_asl[right_putamen_asl['subject'] == subject]['value_asl'].values[0])
        return norm_og, norm_recon, norm_asl
    except (IndexError, KeyError):
        return None, None, None


# ============================================================================
# Quality Metrics Computation
# ============================================================================

def compute_quality_metrics_tle(df_merged):
    """Compute quality metrics for TLE dataset."""
    subjects = df_merged['subject'].unique()

    metrics_pet_recon = {'ssim': [], 'rmse': [], 'psnr': [], 'ncc': []}
    metrics_pet_asl = {'ssim': [], 'rmse': [], 'psnr': [], 'ncc': []}
    subjects_found = []

    for sub in tqdm(subjects, desc="Computing TLE quality metrics"):
        try:
            norm_og, norm_recon, norm_asl = get_putamen_normalization(df_merged, sub)
            if norm_og is None or norm_og == 0 or norm_recon == 0 or norm_asl == 0:
                continue

            pet_original = load_pet_tle(sub) / norm_og
            pet_recon = load_flowgan_tle(sub) / norm_recon
            asl_original = load_asl_tle(sub) / norm_asl

            # Handle ASL 4D array
            if asl_original.ndim == 4:
                asl_original = asl_original[:, :, :, 0]

            metric_pet_recon = get_quality_metrics(pet_original, pet_recon)
            metric_pet_asl = get_quality_metrics(pet_original, asl_original)

            for key in metrics_pet_recon.keys():
                metrics_pet_recon[key].append(metric_pet_recon[key])
                metrics_pet_asl[key].append(metric_pet_asl[key])

            subjects_found.append(sub)

        except Exception as e:
            print(f"Error processing {sub}: {e}")

    return metrics_pet_recon, metrics_pet_asl, subjects_found


def compute_quality_metrics_mci(df_merged):
    """Compute quality metrics for MCI dataset."""
    subjects = df_merged['subject'].unique()

    metrics_pet_recon = {'ssim': [], 'rmse': [], 'psnr': [], 'ncc': []}
    metrics_pet_asl = {'ssim': [], 'rmse': [], 'psnr': [], 'ncc': []}
    subjects_found = []

    for sub in tqdm(subjects, desc="Computing MCI quality metrics"):
        try:
            norm_og, norm_recon, norm_asl = get_putamen_normalization(df_merged, sub)
            if norm_og is None or norm_og == 0 or norm_recon == 0 or norm_asl == 0:
                continue

            pet_original = load_pet_mci(sub) / norm_og
            pet_recon = load_flowgan_mci(sub) / norm_recon
            asl_original = load_asl_mci(sub) / norm_asl

            metric_pet_recon = get_quality_metrics(pet_original, pet_recon)
            metric_pet_asl = get_quality_metrics(pet_original, asl_original)

            for key in metrics_pet_recon.keys():
                metrics_pet_recon[key].append(metric_pet_recon[key])
                metrics_pet_asl[key].append(metric_pet_asl[key])

            subjects_found.append(sub)

        except Exception as e:
            print(f"Error processing {sub}: {e}")

    return metrics_pet_recon, metrics_pet_asl, subjects_found


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_quality_metrics_comparison(metrics_pet_recon, metrics_pet_asl, dataset_name='TLE'):
    """Create comparison plots for all quality metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(8, 10))
    metrics = ['ssim', 'psnr', 'rmse', 'ncc']
    titles = ['SSIM', 'PSNR (dB)', 'RMSE', 'NCC']

    results_list = []

    for ax, metric, title in zip(axes.flatten(), metrics, titles):
        df_plot = pd.DataFrame({
            'Value': metrics_pet_recon[metric] + metrics_pet_asl[metric],
            'Comparison': ['Real PET vs Synthetic PET'] * len(metrics_pet_recon[metric]) +
                          ['Real PET vs ASL'] * len(metrics_pet_asl[metric])
        })

        sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=ax, **BOXPLOT_PARAMS)
        ax.set_title(f'{title}', fontweight='bold', fontsize=12)
        ax.set_xlabel('')
        ax.set_ylabel(title)
        sns.despine(ax=ax)

        # Add statistical annotation
        annotator = Annotator(ax, x='Comparison', y='Value', data=df_plot,
                          pairs=[('Real PET vs Synthetic PET', 'Real PET vs ASL')])
        annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
        annotator.apply_and_annotate()

        # Compute statistics
        results = analyze_and_print(
            group1=metrics_pet_recon[metric],
            group2=metrics_pet_asl[metric],
            group1_name='Real PET vs Synthetic PET',
            group2_name='Real PET vs ASL',
            metric_name=title
        )
        results_list.append({
            'Metric': title,
            'Synthetic_Mean': results['group1_mean'],
            'Synthetic_SD': results['group1_std'],
            'ASL_Mean': results['group2_mean'],
            'ASL_SD': results['group2_std'],
            'p_value': results['p_value'],
            'Cohens_d': results['cohens_d']
        })

    fig.suptitle(f'{dataset_name} - Image Quality Metrics Comparison', fontweight='bold', fontsize=14)
    plt.tight_layout()

    return fig, pd.DataFrame(results_list)


def plot_individual_metric(metrics_pet_recon, metrics_pet_asl, metric, title, dataset_name='TLE'):
    """Plot a single quality metric."""
    fig, ax = plt.subplots(figsize=(6, 5))

    df_plot = pd.DataFrame({
        'Value': metrics_pet_recon[metric] + metrics_pet_asl[metric],
        'Comparison': ['Real PET vs Synthetic PET'] * len(metrics_pet_recon[metric]) +
                      ['Real PET vs ASL'] * len(metrics_pet_asl[metric])
    })

    sns.boxplot(x='Comparison', y='Value', data=df_plot, ax=ax, **BOXPLOT_PARAMS)
    ax.set_title(f'{dataset_name} - {title}', fontweight='bold', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel(title)
    sns.despine(ax=ax)

    # Add statistical annotation
    annotator = Annotator(ax, x='Comparison', y='Value', data=df_plot,
                          pairs=[('Real PET vs Synthetic PET', 'Real PET vs ASL')])
    annotator.configure(test='Mann-Whitney', text_format='star', loc='inside')
    annotator.apply_and_annotate()

    plt.tight_layout()
    return fig


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis(dataset='TLE'):
    """Run quality metrics analysis for specified dataset."""
    print(f"\n{'='*60}")
    print(f"Running Quality Metrics Analysis - {dataset}")
    print(f"{'='*60}\n")

    # Load data
    if dataset == 'TLE':
        pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged.pkl')
        with open(pkl_path, 'rb') as f:
            df_merged = pickle.load(f)
        metrics_recon, metrics_asl, subjects = compute_quality_metrics_tle(df_merged)
    else:  # MCI
        pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(SCRIPT_DIR, 'df_pet_merged_mci.pkl')
        with open(pkl_path, 'rb') as f:
            df_merged = pickle.load(f)
        metrics_recon, metrics_asl, subjects = compute_quality_metrics_mci(df_merged)

    print(f"\nProcessed {len(subjects)} subjects")

    # Create comparison plot
    fig_all, results_df = plot_quality_metrics_comparison(metrics_recon, metrics_asl, dataset)
    save_figure(fig_all, f'quality_metrics_comparison_{dataset}', FIGURES_DIR)
    save_table(results_df, f'quality_metrics_stats_{dataset}', TABLES_DIR)

    # Create individual metric plots
    for metric, title in [('ssim', 'SSIM'), ('psnr', 'PSNR'), ('rmse', 'RMSE'), ('ncc', 'NCC')]:
        fig = plot_individual_metric(metrics_recon, metrics_asl, metric, title, dataset)
        save_figure(fig, f'quality_metric_{metric}_{dataset}', FIGURES_DIR)
        plt.close(fig)

    plt.close('all')

    return metrics_recon, metrics_asl, results_df


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Ensure output directories exist
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)

    # Run analysis for TLE
    print("\n" + "=" * 70)
    print("TLE ANALYSIS")
    print("=" * 70)
    metrics_tle_recon, metrics_tle_asl, results_tle = run_analysis('TLE')

    # Run analysis for MCI
    print("\n" + "=" * 70)
    print("MCI ANALYSIS")
    print("=" * 70)
    metrics_mci_recon, metrics_mci_asl, results_mci = run_analysis('MCI')

    print("\n" + "=" * 70)
    print("Quality Metrics Analysis Complete!")
    print(f"Figures saved to: {FIGURES_DIR}")
    print(f"Tables saved to: {TABLES_DIR}")
    print("=" * 70)
