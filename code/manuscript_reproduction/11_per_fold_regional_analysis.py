"""
Script 11: Per-FlowGAN-Fold Regional Analysis (Reviewer Response)

Reviewer-driven companion to 02_regional_analysis.py / 03_congruency_analysis.py
/ 04_lateralization_cohens_d.py. Mirrors what 10_per_fold_quality_metrics.py did
for the volume-level quality metrics, applied to the *regional* analyses:

  * Across-subject Spearman correlations per region (Synth vs Real, ASL vs Real)
  * Sign congruency per region (TLE asymmetry / MCI SUVR not applicable to sign)
  * Cohen's d per region (TLE: L-TLE vs R-TLE; MCI: MCI vs HC)

For each measure we report THREE columns:
  * pooled  : all subjects (the original 02/03/04 numbers)
  * dev     : 10 dev folds combined (folds 0..9)
  * holdout : holdout subjects only (folds 10 & 11)

For congruency we additionally report fold-mean +- fold-SD across the 10 dev
folds (per-fold rate is robust enough with ~6 subjects). For across-subject
correlations and Cohen's d we report the dev aggregate (per-fold means with n~6
have ~+-0.5 CI on Spearman r, so reporting them as fold-SD is misleading); the
holdout column is the truly-unseen check.

Outputs:
  tables/11_per_fold_regional_analysis/
    across_subject_{TLE,MCI}_{DKT,HarvardOxford}.csv   per-region pooled/dev/holdout r
    across_subject_summary_{TLE,MCI}_{DKT,HarvardOxford}.csv   median/IQR/Wilcoxon for each split
    congruency_{TLE,MCI}_{DKT,HarvardOxford}.csv       per-region congruency w/ fold mean+-SD
    cohens_d_{TLE,MCI}_{DKT,HarvardOxford}.csv         per-region Cohen's d pooled/dev/holdout
    summary_{TLE,MCI}_{DKT,HarvardOxford}.csv          one-row manuscript-ready summary
"""

import os
import sys
import json
import pickle
import importlib.util
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar


# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR  = os.path.join(SCRIPT_DIR, 'tables',  '11_per_fold_regional_analysis')

DEV_FOLDS     = [f'fold_{i}' for i in range(10)]
HOLDOUT_FOLDS = ['fold_10', 'fold_11']

FOLD_JSON_TLE = os.path.join(SCRIPT_DIR, 'data', 'subjects_in_each_fold_TLE.json')
FOLD_JSON_MCI = os.path.join(SCRIPT_DIR, 'data', 'subjects_in_each_fold_MCI.json')

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']

CLINICAL_DATA_TLE = os.path.join(SCRIPT_DIR, 'data', 'clinical_metadata.xlsx')
MCI_CONTROL_LIST  = os.path.join(SCRIPT_DIR, 'data', 'list_of_control_subjects.txt')
MCI_PATIENT_LIST  = os.path.join(SCRIPT_DIR, 'data', 'list_of_MCI_subjects.txt')


# ============================================================================
# Reuse modules 02_/03_/04_ via importlib (numeric-prefixed names)
# ============================================================================

def _import_mod(name: str):
    path = os.path.join(SCRIPT_DIR, name)
    spec = importlib.util.spec_from_file_location(
        f'mod_{name.replace(".", "_")}', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MOD02 = _import_mod('02_regional_analysis.py')
MOD03 = _import_mod('03_congruency_analysis.py')
MOD04 = _import_mod('04_lateralization_cohens_d.py')


# ============================================================================
# Fold mapping
# ============================================================================

def load_fold_map(json_path: str) -> Dict[str, str]:
    """Build {subject_id -> fold_name} from subjects_in_each_fold.json."""
    with open(json_path) as f:
        d = json.load(f)
    return {s: fold for fold, info in d.items() for s in info.get('test', [])}


def subject_splits(subjects: List[str], fold_map: Dict[str, str]):
    """Return dict: split_name -> set of subjects."""
    dev = {s for s in subjects if fold_map.get(s) in DEV_FOLDS}
    ho  = {s for s in subjects if fold_map.get(s) in HOLDOUT_FOLDS}
    pooled = set(subjects)
    fold_to_subjects = {f: {s for s in subjects if fold_map.get(s) == f}
                        for f in DEV_FOLDS}
    return pooled, dev, ho, fold_to_subjects


# ============================================================================
# Helper: filter df_ai / df_suvr by subject set
# ============================================================================

def filter_df(df: pd.DataFrame, subjects: Set[str], subject_col: str = 'Subject'):
    return df[df[subject_col].isin(subjects)].copy()


# ============================================================================
# Across-subject Spearman correlation per region
# ============================================================================

def across_subject_correlations(df_long: pd.DataFrame, val_real: str,
                                val_synth: str, val_asl: str,
                                regions: List[str]) -> pd.DataFrame:
    """Spearman r per region across subjects."""
    rows = []
    for region in regions:
        d = df_long[df_long['Region'] == region]
        if len(d) < 4:
            continue
        x  = d[val_real].values
        y1 = d[val_synth].values
        y2 = d[val_asl].values
        v1 = np.isfinite(x) & np.isfinite(y1)
        v2 = np.isfinite(x) & np.isfinite(y2)
        if v1.sum() < 4 or v2.sum() < 4:
            continue
        r_s, _ = stats.spearmanr(x[v1], y1[v1])
        r_a, _ = stats.spearmanr(x[v2], y2[v2])
        rows.append({'Region': region, 'r_synth': r_s, 'r_asl': r_a,
                     'n': int(v1.sum())})
    return pd.DataFrame(rows)


def summarize_corrs(df: pd.DataFrame) -> dict:
    """Median, IQR, Wilcoxon for Synth vs ASL per-region correlations."""
    if df is None or len(df) == 0:
        return {'n_regions': 0, 'synth_median': np.nan, 'synth_iqr': (np.nan, np.nan),
                'asl_median': np.nan,   'asl_iqr':   (np.nan, np.nan),
                'wilcoxon_p': np.nan, 'cohens_d_paired': np.nan,
                'n_synth_gt_asl': 0}
    s = df['r_synth'].values; a = df['r_asl'].values
    valid = np.isfinite(s) & np.isfinite(a)
    s = s[valid]; a = a[valid]
    if len(s) < 2:
        return {'n_regions': len(s), 'synth_median': np.nan, 'asl_median': np.nan,
                'wilcoxon_p': np.nan, 'cohens_d_paired': np.nan, 'n_synth_gt_asl': 0,
                'synth_iqr': (np.nan, np.nan), 'asl_iqr': (np.nan, np.nan)}
    try:
        _, p = stats.wilcoxon(s, a)
    except Exception:
        p = np.nan
    diff = s - a
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan
    return {
        'n_regions':       int(len(s)),
        'synth_median':    float(np.median(s)),
        'synth_iqr':       (float(np.percentile(s, 25)), float(np.percentile(s, 75))),
        'asl_median':      float(np.median(a)),
        'asl_iqr':         (float(np.percentile(a, 25)), float(np.percentile(a, 75))),
        'wilcoxon_p':      float(p) if p == p else np.nan,
        'cohens_d_paired': d,
        'n_synth_gt_asl':  int(np.sum(s > a)),
    }


# ============================================================================
# Congruency: per-region rates and per-fold mean+-SD
# ============================================================================

def congruency_rates_per_region(df_ai: pd.DataFrame, regions: List[str]
                                ) -> pd.DataFrame:
    rows = []
    for region in regions:
        d = df_ai[df_ai['Region'] == region]
        if len(d) == 0:
            continue
        ref   = d['PET AI Original'].values
        synth = d['PET AI Recon'].values
        asl   = d['ASL AI'].values
        cong_s = ((ref >= 0) & (synth >= 0)) | ((ref < 0) & (synth < 0))
        cong_a = ((ref >= 0) & (asl   >= 0)) | ((ref < 0) & (asl   < 0))
        rows.append({'Region': region,
                     'n': int(len(d)),
                     'cong_synth': float(np.mean(cong_s)),
                     'cong_asl':   float(np.mean(cong_a))})
    return pd.DataFrame(rows)


def per_fold_congruency(df_ai: pd.DataFrame, fold_to_subjects: Dict[str, Set[str]],
                        regions: List[str]) -> pd.DataFrame:
    """For each region, compute the fold-mean and fold-SD of (cong_synth, cong_asl)
       across the 10 dev folds (each fold's rate is across the ~6 fold subjects)."""
    rows = []
    for region in regions:
        synth_fold_rates = []
        asl_fold_rates   = []
        for fold, subs in fold_to_subjects.items():
            d = df_ai[(df_ai['Region'] == region) & (df_ai['Subject'].isin(subs))]
            if len(d) < 2:
                continue
            ref   = d['PET AI Original'].values
            synth = d['PET AI Recon'].values
            asl   = d['ASL AI'].values
            cong_s = ((ref >= 0) & (synth >= 0)) | ((ref < 0) & (synth < 0))
            cong_a = ((ref >= 0) & (asl   >= 0)) | ((ref < 0) & (asl   < 0))
            synth_fold_rates.append(np.mean(cong_s))
            asl_fold_rates.append(np.mean(cong_a))
        if len(synth_fold_rates) < 2:
            continue
        rows.append({
            'Region':         region,
            'n_folds':        len(synth_fold_rates),
            'fold_mean_synth': float(np.mean(synth_fold_rates)),
            'fold_sd_synth':   float(np.std(synth_fold_rates, ddof=1)),
            'fold_mean_asl':   float(np.mean(asl_fold_rates)),
            'fold_sd_asl':     float(np.std(asl_fold_rates, ddof=1)),
        })
    return pd.DataFrame(rows)


def summarize_congruency(df_pool: pd.DataFrame) -> dict:
    if df_pool is None or len(df_pool) == 0:
        return {'n_regions': 0, 'synth_mean': np.nan, 'asl_mean': np.nan,
                'n_synth_gt_asl': 0, 'wilcoxon_p': np.nan, 'cohens_d_paired': np.nan}
    s = df_pool['cong_synth'].values; a = df_pool['cong_asl'].values
    valid = np.isfinite(s) & np.isfinite(a)
    s = s[valid]; a = a[valid]
    if len(s) < 2:
        return {'n_regions': len(s), 'synth_mean': float(np.mean(s)) if len(s) else np.nan,
                'asl_mean': float(np.mean(a)) if len(a) else np.nan,
                'n_synth_gt_asl': int(np.sum(s > a)),
                'wilcoxon_p': np.nan, 'cohens_d_paired': np.nan}
    try:
        _, p = stats.wilcoxon(s, a)
    except Exception:
        p = np.nan
    diff = s - a
    d = float(np.mean(diff) / np.std(diff, ddof=1)) if np.std(diff, ddof=1) > 0 else np.nan
    return {'n_regions': int(len(s)),
            'synth_mean':      float(np.mean(s)),
            'asl_mean':        float(np.mean(a)),
            'n_synth_gt_asl':  int(np.sum(s > a)),
            'wilcoxon_p':      float(p) if p == p else np.nan,
            'cohens_d_paired': d}


# ============================================================================
# Cohen's d per region (TLE: L-TLE vs R-TLE asymmetry; MCI: MCI vs HC SUVR)
# ============================================================================

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1 = np.var(a, ddof=1); s2 = np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) /
                     (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def tle_cohens_d_per_region(df_ai: pd.DataFrame, df_left_label: pd.DataFrame,
                            regions: List[str]) -> pd.DataFrame:
    """TLE: Cohen's d between L-TLE and R-TLE per region per modality.
    df_left_label: DataFrame with columns ['Subject', 'isLeft']."""
    merged = df_ai.merge(df_left_label, on='Subject', how='inner')
    rows = []
    for region in regions:
        d = merged[merged['Region'] == region]
        l = d[d['isLeft'] == 1]; r = d[d['isLeft'] == 0]
        if len(l) < 2 or len(r) < 2:
            continue
        rows.append({
            'Region':                  region,
            'd_real':  cohens_d(l['PET AI Original'].values, r['PET AI Original'].values),
            'd_synth': cohens_d(l['PET AI Recon'].values,    r['PET AI Recon'].values),
            'd_asl':   cohens_d(l['ASL AI'].values,          r['ASL AI'].values),
            'n_left':  len(l), 'n_right': len(r),
        })
    return pd.DataFrame(rows)


def mci_cohens_d_per_region(df_suvr: pd.DataFrame, regions: List[str]
                            ) -> pd.DataFrame:
    """MCI: Cohen's d between MCI and HC per region per modality.
    df_suvr is built by 04_'s build_suvr_dataframe_mci and is wide-format."""
    rows = []
    for region in regions:
        real_col, synth_col, asl_col = f'{region}_real', f'{region}_synth', f'{region}_asl'
        if any(c not in df_suvr.columns for c in (real_col, synth_col, asl_col)):
            continue
        hc  = df_suvr[df_suvr['is_mci'] == 0]
        mci = df_suvr[df_suvr['is_mci'] == 1]
        if len(hc) < 2 or len(mci) < 2:
            continue
        rows.append({
            'Region':                  region,
            'd_real':  cohens_d(mci[real_col].values,  hc[real_col].values),
            'd_synth': cohens_d(mci[synth_col].values, hc[synth_col].values),
            'd_asl':   cohens_d(mci[asl_col].values,   hc[asl_col].values),
            'n_hc':    len(hc), 'n_mci': len(mci),
        })
    return pd.DataFrame(rows)


def summarize_cohens_d(df: pd.DataFrame) -> dict:
    if df is None or len(df) == 0:
        return {'n_regions': 0,
                'mean_abs_d_real': np.nan, 'mean_abs_d_synth': np.nan,
                'mean_abs_d_asl':  np.nan,
                'corr_real_asl':   np.nan, 'corr_synth_asl': np.nan,
                'n_synth_gt_asl_magnitude': 0}
    dr = df['d_real'].values; ds = df['d_synth'].values; da = df['d_asl'].values
    valid = np.isfinite(dr) & np.isfinite(ds) & np.isfinite(da)
    dr = dr[valid]; ds = ds[valid]; da = da[valid]
    out = {
        'n_regions':           int(len(dr)),
        'mean_abs_d_real':     float(np.mean(np.abs(dr))) if len(dr) else np.nan,
        'mean_abs_d_synth':    float(np.mean(np.abs(ds))) if len(ds) else np.nan,
        'mean_abs_d_asl':      float(np.mean(np.abs(da))) if len(da) else np.nan,
        'n_synth_gt_asl_magnitude': int(np.sum(np.abs(ds) > np.abs(da))),
    }
    if len(dr) >= 4:
        out['corr_real_asl']  = float(stats.spearmanr(dr, da)[0])
        out['corr_synth_asl'] = float(stats.spearmanr(ds, da)[0])
    else:
        out['corr_real_asl']  = np.nan
        out['corr_synth_asl'] = np.nan
    return out


# ============================================================================
# Main per-cohort, per-atlas worker
# ============================================================================

def load_df_merged(cohort: str, atlas: str) -> pd.DataFrame:
    if cohort == 'TLE':
        path = 'df_pet_merged_ho.pkl' if atlas == 'HarvardOxford' else 'df_pet_merged.pkl'
    else:
        path = 'df_pet_merged_mci_ho.pkl' if atlas == 'HarvardOxford' else 'df_pet_merged_mci.pkl'
    pkl = os.path.join(SCRIPT_DIR, path)
    with open(pkl, 'rb') as f:
        return pickle.load(f)


def regions_for(df_long: pd.DataFrame, label_col: str = 'Region') -> List[str]:
    regions = [r for r in df_long[label_col].unique() if r not in EXCLUDE_REGIONS]
    return regions


def run_one(cohort: str, atlas: str, fold_map: Dict[str, str]) -> None:
    print(f"\n{'='*70}\n{cohort} - {atlas}\n{'='*70}")
    df_merged = load_df_merged(cohort, atlas)
    subjects = list(df_merged['subject'].unique())
    print(f"  Loaded {len(subjects)} subjects")

    # Splits in subject space
    pooled, dev, holdout, fold_subjects = subject_splits(subjects, fold_map)
    n_dev = len(dev); n_ho = len(holdout); n_unmapped = len(subjects) - n_dev - n_ho
    print(f"  pooled={len(pooled)}  dev={n_dev}  holdout={n_ho}  unmapped={n_unmapped}")

    # ---------------- Across-subject correlations (Asymmetry for TLE, SUVR for MCI) ----------------
    df_ai   = MOD02.build_asymmetry_dataframe(df_merged)
    putamen = MOD02.get_putamen_normalization_values(df_merged)
    df_suvr = MOD02.build_suvr_dataframe(df_merged, putamen)
    regs_ai   = regions_for(df_ai)
    regs_suvr = regions_for(df_suvr)

    if cohort == 'TLE':
        df_corr_pool  = across_subject_correlations(filter_df(df_ai,   pooled),
                                                    'PET AI Original', 'PET AI Recon', 'ASL AI', regs_ai)
        df_corr_dev   = across_subject_correlations(filter_df(df_ai,   dev),
                                                    'PET AI Original', 'PET AI Recon', 'ASL AI', regs_ai)
        df_corr_ho    = across_subject_correlations(filter_df(df_ai,   holdout),
                                                    'PET AI Original', 'PET AI Recon', 'ASL AI', regs_ai)
        across_label  = 'asymmetry'
    else:
        df_corr_pool  = across_subject_correlations(filter_df(df_suvr, pooled),
                                                    'PET SUVR Original', 'PET SUVR FlowGAN', 'ASL rCBF', regs_suvr)
        df_corr_dev   = across_subject_correlations(filter_df(df_suvr, dev),
                                                    'PET SUVR Original', 'PET SUVR FlowGAN', 'ASL rCBF', regs_suvr)
        df_corr_ho    = across_subject_correlations(filter_df(df_suvr, holdout),
                                                    'PET SUVR Original', 'PET SUVR FlowGAN', 'ASL rCBF', regs_suvr)
        across_label  = 'suvr'

    # Wide table: per region pooled/dev/holdout
    wide = (df_corr_pool.rename(columns={'r_synth': 'r_synth_pool', 'r_asl': 'r_asl_pool', 'n': 'n_pool'})
            .merge(df_corr_dev.rename(columns={'r_synth': 'r_synth_dev', 'r_asl': 'r_asl_dev', 'n': 'n_dev'}),
                   on='Region', how='outer')
            .merge(df_corr_ho.rename(columns={'r_synth': 'r_synth_ho', 'r_asl': 'r_asl_ho', 'n': 'n_ho'}),
                   on='Region', how='outer'))
    wide.to_csv(os.path.join(TABLES_DIR, f'across_subject_{cohort}_{atlas}.csv'), index=False)

    sum_pool = summarize_corrs(df_corr_pool)
    sum_dev  = summarize_corrs(df_corr_dev)
    sum_ho   = summarize_corrs(df_corr_ho)
    pd.DataFrame([{'split': 'pooled',  **sum_pool},
                  {'split': 'dev',     **sum_dev},
                  {'split': 'holdout', **sum_ho}]
                 ).to_csv(os.path.join(TABLES_DIR, f'across_subject_summary_{cohort}_{atlas}.csv'), index=False)

    print(f"  ACROSS-SUBJECT {across_label} correlations:")
    for label, s in [('pool', sum_pool), ('dev', sum_dev), ('ho', sum_ho)]:
        print(f"    {label:7s}  n={s['n_regions']:3d}  synth median r={s['synth_median']:.3f} "
              f"[{s['synth_iqr'][0]:.2f},{s['synth_iqr'][1]:.2f}]  "
              f"asl median r={s['asl_median']:.3f}  p={s['wilcoxon_p']:.4f}  "
              f"d={s['cohens_d_paired']:.2f}  synth>asl={s['n_synth_gt_asl']}/{s['n_regions']}")

    # ---------------- Congruency (per region) ----------------
    cong_pool = congruency_rates_per_region(filter_df(df_ai, pooled),  regs_ai)
    cong_dev  = congruency_rates_per_region(filter_df(df_ai, dev),     regs_ai)
    cong_ho   = congruency_rates_per_region(filter_df(df_ai, holdout), regs_ai)
    cong_fold = per_fold_congruency(df_ai, fold_subjects, regs_ai)

    wide_c = (cong_pool.rename(columns={'cong_synth': 'cong_synth_pool', 'cong_asl': 'cong_asl_pool', 'n': 'n_pool'})
              .merge(cong_dev.rename(columns={'cong_synth': 'cong_synth_dev', 'cong_asl': 'cong_asl_dev', 'n': 'n_dev'}),
                     on='Region', how='outer')
              .merge(cong_ho.rename(columns={'cong_synth': 'cong_synth_ho', 'cong_asl': 'cong_asl_ho', 'n': 'n_ho'}),
                     on='Region', how='outer')
              .merge(cong_fold, on='Region', how='outer'))
    wide_c.to_csv(os.path.join(TABLES_DIR, f'congruency_{cohort}_{atlas}.csv'), index=False)

    sum_c_pool = summarize_congruency(cong_pool)
    sum_c_dev  = summarize_congruency(cong_dev)
    sum_c_ho   = summarize_congruency(cong_ho)
    pd.DataFrame([{'split': 'pooled',  **sum_c_pool},
                  {'split': 'dev',     **sum_c_dev},
                  {'split': 'holdout', **sum_c_ho}]
                 ).to_csv(os.path.join(TABLES_DIR, f'congruency_summary_{cohort}_{atlas}.csv'), index=False)

    print(f"  CONGRUENCY:")
    for label, s in [('pool', sum_c_pool), ('dev', sum_c_dev), ('ho', sum_c_ho)]:
        print(f"    {label:7s}  n={s['n_regions']:3d}  synth_mean={s['synth_mean']:.3f}  asl_mean={s['asl_mean']:.3f}  "
              f"p={s['wilcoxon_p']:.4f}  d={s['cohens_d_paired']:.2f}  synth>asl={s['n_synth_gt_asl']}/{s['n_regions']}")

    # ---------------- Cohen's d per region ----------------
    if cohort == 'TLE':
        df_left = MOD04.load_clinical_metadata(pet_subject_ids=subjects)
        d_pool = tle_cohens_d_per_region(filter_df(df_ai, pooled),  df_left, regs_ai)
        d_dev  = tle_cohens_d_per_region(filter_df(df_ai, dev),     df_left, regs_ai)
        d_ho   = tle_cohens_d_per_region(filter_df(df_ai, holdout), df_left, regs_ai)
    else:
        md_mci = MOD04.load_mci_metadata()
        df_suvr_mci_pool = MOD04.build_suvr_dataframe_mci(df_merged, md_mci, atlas=atlas)
        # Restrict the wide subject-by-region SUVR table to each split
        d_pool = mci_cohens_d_per_region(df_suvr_mci_pool[df_suvr_mci_pool['Subject'].isin(pooled)],
                                         regs_for_mci(df_suvr_mci_pool, atlas))
        d_dev  = mci_cohens_d_per_region(df_suvr_mci_pool[df_suvr_mci_pool['Subject'].isin(dev)],
                                         regs_for_mci(df_suvr_mci_pool, atlas))
        d_ho   = mci_cohens_d_per_region(df_suvr_mci_pool[df_suvr_mci_pool['Subject'].isin(holdout)],
                                         regs_for_mci(df_suvr_mci_pool, atlas))

    # Wide table
    if len(d_pool):
        wide_d = (d_pool.rename(columns={'d_real': 'd_real_pool', 'd_synth': 'd_synth_pool', 'd_asl': 'd_asl_pool'})
                  .merge(d_dev.rename(columns={'d_real': 'd_real_dev', 'd_synth': 'd_synth_dev', 'd_asl': 'd_asl_dev'}),
                         on='Region', how='outer', suffixes=('', '_dev'))
                  .merge(d_ho.rename(columns={'d_real': 'd_real_ho', 'd_synth': 'd_synth_ho', 'd_asl': 'd_asl_ho'}),
                         on='Region', how='outer', suffixes=('', '_ho')))
        # Drop n_* duplicate columns to keep things tidy
        keep = [c for c in wide_d.columns
                if c == 'Region' or c.startswith('d_')]
        wide_d[keep].to_csv(os.path.join(TABLES_DIR, f'cohens_d_{cohort}_{atlas}.csv'), index=False)

    sum_d_pool = summarize_cohens_d(d_pool)
    sum_d_dev  = summarize_cohens_d(d_dev)
    sum_d_ho   = summarize_cohens_d(d_ho)
    pd.DataFrame([{'split': 'pooled',  **sum_d_pool},
                  {'split': 'dev',     **sum_d_dev},
                  {'split': 'holdout', **sum_d_ho}]
                 ).to_csv(os.path.join(TABLES_DIR, f'cohens_d_summary_{cohort}_{atlas}.csv'), index=False)

    print(f"  COHEN'S d:")
    for label, s in [('pool', sum_d_pool), ('dev', sum_d_dev), ('ho', sum_d_ho)]:
        print(f"    {label:7s}  n={s['n_regions']:3d}  |d|_real={s['mean_abs_d_real']:.2f}  "
              f"|d|_synth={s['mean_abs_d_synth']:.2f}  |d|_asl={s['mean_abs_d_asl']:.2f}  "
              f"corr(synth,asl)={s['corr_synth_asl']:.2f}  "
              f"synth_mag>asl={s['n_synth_gt_asl_magnitude']}/{s['n_regions']}")


def regs_for_mci(df_suvr_mci: pd.DataFrame, atlas: str) -> List[str]:
    """MCI Cohen's d uses 04_'s MCI_REGIONS for DKT; all regions for HO."""
    regions = sorted(set(c.rsplit('_', 1)[0] for c in df_suvr_mci.columns
                       if c.endswith(('_real', '_synth', '_asl'))))
    if atlas == 'DKT':
        try:
            from utils import MCI_REGIONS
            regions = [r for r in regions if r in MCI_REGIONS and r not in EXCLUDE_REGIONS]
        except Exception:
            regions = [r for r in regions if r not in EXCLUDE_REGIONS]
    else:
        regions = [r for r in regions if r not in EXCLUDE_REGIONS]
    return regions


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)

    fold_map_tle = load_fold_map(FOLD_JSON_TLE)
    fold_map_mci = load_fold_map(FOLD_JSON_MCI)

    for cohort, fmap in [('TLE', fold_map_tle), ('MCI', fold_map_mci)]:
        for atlas in ['DKT', 'HarvardOxford']:
            try:
                run_one(cohort, atlas, fmap)
            except Exception as e:
                print(f"  FAILED for {cohort}/{atlas}: {e}")
                import traceback; traceback.print_exc()

    print("\nDone. Tables in:", TABLES_DIR)
