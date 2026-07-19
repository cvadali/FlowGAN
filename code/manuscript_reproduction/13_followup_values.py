"""
Script 13: Follow-up values for reviewer response

Two small additions to round out the manuscript revision:

  1. TLE within-subject SUVR-based Spearman correlation (across brain regions),
     reported per fold. The existing 10_per_fold_quality_metrics.py defaulted
     TLE to asymmetry-based; this script adds the SUVR-based number that the
     manuscript's r=0.91 headline is referring to.
  2. Cohen's d improvement-correlation r-values per split:
     corr( (d_real - d_asl), (d_synth - d_asl) ) over regions,
     computed on pooled / cross-validated / test splits for all 4 (cohort, atlas).

Outputs:
  tables/13_followup_values/
    tle_within_subject_suvr_{DKT,HarvardOxford}.csv   per-subject r + fold mean+-SD + test set
    cohens_d_improvement_corr_summary.csv             per split, all (cohort, atlas)
"""

import os
import json
import pickle
import importlib.util
from typing import Dict, List, Set

import numpy as np
import pandas as pd
from scipy import stats


SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR  = os.path.join(SCRIPT_DIR, 'tables', '13_followup_values')

DEV_FOLDS     = [f'fold_{i}' for i in range(10)]
HOLDOUT_FOLDS = ['fold_10', 'fold_11']

FOLD_JSON_TLE = os.path.join(SCRIPT_DIR, 'data', 'subjects_in_each_fold_TLE.json')
FOLD_JSON_MCI = os.path.join(SCRIPT_DIR, 'data', 'subjects_in_each_fold_MCI.json')

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']


def _import_mod(name: str):
    path = os.path.join(SCRIPT_DIR, name)
    spec = importlib.util.spec_from_file_location(f'mod_{name}', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


MOD04 = _import_mod('04_lateralization_cohens_d.py')


# ============================================================================
# Helpers
# ============================================================================

def load_fold_map(json_path: str) -> Dict[str, str]:
    with open(json_path) as f:
        d = json.load(f)
    return {s: fold for fold, info in d.items() for s in info.get('test', [])}


def load_df_merged(cohort: str, atlas: str) -> pd.DataFrame:
    pkl_name = {('TLE', 'DKT'):           'df_pet_merged.pkl',
                ('TLE', 'HarvardOxford'): 'df_pet_merged_ho.pkl',
                ('MCI', 'DKT'):           'df_pet_merged_mci.pkl',
                ('MCI', 'HarvardOxford'): 'df_pet_merged_mci_ho.pkl'}[(cohort, atlas)]
    return pickle.load(open(os.path.join(SCRIPT_DIR, pkl_name), 'rb'))


# ============================================================================
# (1) TLE within-subject SUVR-based per fold
# ============================================================================

def within_subject_suvr_per_subject(df_merged: pd.DataFrame) -> pd.DataFrame:
    """One row per subject: r_synth, r_asl (Spearman across regions of SUVR)."""
    rows = []
    for sub in df_merged['subject'].unique():
        df_s = df_merged[df_merged['subject'] == sub].copy()

        l_og = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Left')]['value_pet_original'].values
        r_og = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Right')]['value_pet_original'].values
        l_rc = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Left')]['value_pet_recon'].values
        r_rc = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Right')]['value_pet_recon'].values
        l_as = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Left')]['value_asl'].values
        r_as = df_s[(df_s['region_name'] == 'Putamen') & (df_s['side'] == 'Right')]['value_asl'].values

        if not all(len(x) > 0 for x in (l_og, r_og, l_rc, r_rc, l_as, r_as)):
            continue
        norm_og = float(l_og[0] + r_og[0])
        norm_rc = float(l_rc[0] + r_rc[0])
        norm_as = float(l_as[0] + r_as[0])
        if not all([norm_og, norm_rc, norm_as]):
            continue

        df_s['_og']  = df_s['value_pet_original'] / norm_og
        df_s['_rc']  = df_s['value_pet_recon']    / norm_rc
        df_s['_asl'] = df_s['value_asl']          / norm_as

        d = df_s.dropna(subset=['_og', '_rc', '_asl'])
        d = d[~d['region_name'].isin(EXCLUDE_REGIONS)]
        if len(d) < 5:
            continue
        r_rc_val, _  = stats.spearmanr(d['_og'], d['_rc'])
        r_asl_val, _ = stats.spearmanr(d['_og'], d['_asl'])
        rows.append({'subject': sub,
                     'r_synth': float(r_rc_val) if r_rc_val == r_rc_val else np.nan,
                     'r_asl':   float(r_asl_val) if r_asl_val == r_asl_val else np.nan})
    return pd.DataFrame(rows)


def summarize_by_fold(df_subj: pd.DataFrame, fold_map: Dict[str, str]
                      ) -> Dict[str, dict]:
    """Return {'cv': fold-level summary, 'test': test summary}."""
    df = df_subj.copy()
    df['fold'] = df['subject'].map(fold_map)
    df['split'] = df['fold'].apply(
        lambda f: 'test' if f in HOLDOUT_FOLDS
                  else ('cv' if f in DEV_FOLDS else 'unknown'))

    out = {}
    cv = df[df['split'] == 'cv']
    if len(cv) > 0:
        fmeans_s = cv.groupby('fold')['r_synth'].mean().reindex(DEV_FOLDS).values
        fmeans_a = cv.groupby('fold')['r_asl'  ].mean().reindex(DEV_FOLDS).values
        out['cv'] = {
            'n_folds_used':       int(np.isfinite(fmeans_s).sum()),
            'r_synth_fold_mean':  float(np.nanmean(fmeans_s)),
            'r_synth_fold_sd':    float(np.nanstd(fmeans_s, ddof=1)),
            'r_asl_fold_mean':    float(np.nanmean(fmeans_a)),
            'r_asl_fold_sd':      float(np.nanstd(fmeans_a, ddof=1)),
            'fold_means_synth':   ';'.join('NA' if not np.isfinite(v) else f'{v:.4f}'
                                            for v in fmeans_s),
            'fold_means_asl':     ';'.join('NA' if not np.isfinite(v) else f'{v:.4f}'
                                            for v in fmeans_a),
        }
    ho = df[df['split'] == 'test']
    if len(ho) > 0:
        out['test'] = {
            'n_test_subjects': int(len(ho)),
            'r_synth_mean':    float(ho['r_synth'].mean()),
            'r_synth_sd':      float(ho['r_synth'].std(ddof=1)),
            'r_asl_mean':      float(ho['r_asl'].mean()),
            'r_asl_sd':        float(ho['r_asl'].std(ddof=1)),
        }
    # Paired comparison on cv folds
    if len(cv) > 0:
        try:
            _, p = stats.wilcoxon(cv['r_synth'].dropna(), cv['r_asl'].dropna())
            out['cv']['wilcoxon_p_synth_vs_asl'] = float(p)
        except Exception:
            out['cv']['wilcoxon_p_synth_vs_asl'] = np.nan
    return out


def run_tle_within_subject_suvr():
    print("=" * 70)
    print("(1) TLE within-subject SUVR-based per fold")
    print("=" * 70)
    fm = load_fold_map(FOLD_JSON_TLE)

    for atlas in ['DKT', 'HarvardOxford']:
        df_merged = load_df_merged('TLE', atlas)
        df_subj = within_subject_suvr_per_subject(df_merged)
        df_subj['fold'] = df_subj['subject'].map(fm)
        df_subj.to_csv(os.path.join(TABLES_DIR,
                                    f'tle_within_subject_suvr_{atlas}.csv'),
                        index=False)
        s = summarize_by_fold(df_subj, fm)
        print(f"\nTLE - {atlas}:")
        if 'cv' in s:
            cv = s['cv']
            print(f"  CV (folds, mean+-SD):  synth = {cv['r_synth_fold_mean']:.3f} +- {cv['r_synth_fold_sd']:.3f}    "
                  f"asl = {cv['r_asl_fold_mean']:.3f} +- {cv['r_asl_fold_sd']:.3f}    "
                  f"Wilcoxon p={cv['wilcoxon_p_synth_vs_asl']:.4g}    n_folds={cv['n_folds_used']}")
        if 'test' in s:
            ho = s['test']
            print(f"  Test (n={ho['n_test_subjects']}, mean+-SD):  synth = {ho['r_synth_mean']:.3f} +- {ho['r_synth_sd']:.3f}    "
                  f"asl = {ho['r_asl_mean']:.3f} +- {ho['r_asl_sd']:.3f}")


# ============================================================================
# (2) Cohen's d improvement-correlation per split
# ============================================================================

def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * s1 + (len(b) - 1) * s2) /
                     (len(a) + len(b) - 2))
    if pooled == 0:
        return np.nan
    return float((np.mean(a) - np.mean(b)) / pooled)


def tle_cohens_d_per_region(df_ai, df_left_label, regions):
    merged = df_ai.merge(df_left_label, on='Subject', how='inner')
    rows = []
    for r in regions:
        d = merged[merged['Region'] == r]
        l = d[d['isLeft'] == 1]; rr = d[d['isLeft'] == 0]
        if len(l) < 2 or len(rr) < 2:
            continue
        rows.append({
            'Region':  r,
            'd_real':  cohens_d(l['PET AI Original'].values, rr['PET AI Original'].values),
            'd_synth': cohens_d(l['PET AI Recon'].values,    rr['PET AI Recon'].values),
            'd_asl':   cohens_d(l['ASL AI'].values,          rr['ASL AI'].values),
        })
    return pd.DataFrame(rows)


def mci_cohens_d_per_region(df_suvr_wide, regions):
    rows = []
    for r in regions:
        if any(f'{r}_{m}' not in df_suvr_wide.columns
               for m in ('real', 'synth', 'asl')):
            continue
        hc = df_suvr_wide[df_suvr_wide['is_mci'] == 0]
        mc = df_suvr_wide[df_suvr_wide['is_mci'] == 1]
        if len(hc) < 2 or len(mc) < 2:
            continue
        rows.append({
            'Region':  r,
            'd_real':  cohens_d(mc[f'{r}_real'].values,  hc[f'{r}_real'].values),
            'd_synth': cohens_d(mc[f'{r}_synth'].values, hc[f'{r}_synth'].values),
            'd_asl':   cohens_d(mc[f'{r}_asl'].values,   hc[f'{r}_asl'].values),
        })
    return pd.DataFrame(rows)


def improvement_corr(df_d: pd.DataFrame) -> tuple:
    """Spearman r between (d_real - d_asl) and (d_synth - d_asl) across regions."""
    if df_d is None or len(df_d) < 4:
        return (np.nan, np.nan, 0)
    real_imp  = df_d['d_real'].values  - df_d['d_asl'].values
    synth_imp = df_d['d_synth'].values - df_d['d_asl'].values
    v = np.isfinite(real_imp) & np.isfinite(synth_imp)
    if v.sum() < 4:
        return (np.nan, np.nan, int(v.sum()))
    r, p = stats.spearmanr(real_imp[v], synth_imp[v])
    return (float(r), float(p), int(v.sum()))


def run_cohens_d_improvement_corr():
    print("\n" + "=" * 70)
    print("(2) Cohen's d improvement-correlation per split")
    print("=" * 70)

    out_rows = []
    fm_tle = load_fold_map(FOLD_JSON_TLE)
    fm_mci = load_fold_map(FOLD_JSON_MCI)

    for cohort, fm in [('TLE', fm_tle), ('MCI', fm_mci)]:
        for atlas in ['DKT', 'HarvardOxford']:
            df_merged = load_df_merged(cohort, atlas)
            subs = list(df_merged['subject'].unique())
            cv_subs = {s for s in subs if fm.get(s) in DEV_FOLDS}
            te_subs = {s for s in subs if fm.get(s) in HOLDOUT_FOLDS}

            if cohort == 'TLE':
                df_ai = MOD04.build_asymmetry_dataframe(df_merged)
                df_left = MOD04.load_clinical_metadata(pet_subject_ids=subs)
                regions = [r for r in df_ai['Region'].unique()
                           if r not in EXCLUDE_REGIONS]
                d_pool = tle_cohens_d_per_region(df_ai,                                   df_left, regions)
                d_cv   = tle_cohens_d_per_region(df_ai[df_ai['Subject'].isin(cv_subs)],   df_left, regions)
                d_test = tle_cohens_d_per_region(df_ai[df_ai['Subject'].isin(te_subs)],   df_left, regions)
            else:
                md = MOD04.load_mci_metadata()
                df_w = MOD04.build_suvr_dataframe_mci(df_merged, md, atlas=atlas)
                regions = sorted(set(c.rsplit('_', 1)[0]
                                   for c in df_w.columns
                                   if c.endswith(('_real', '_synth', '_asl'))))
                from utils import MCI_REGIONS
                if atlas == 'DKT':
                    regions = [r for r in regions if r in MCI_REGIONS and r not in EXCLUDE_REGIONS]
                else:
                    regions = [r for r in regions if r not in EXCLUDE_REGIONS]
                d_pool = mci_cohens_d_per_region(df_w,                                       regions)
                d_cv   = mci_cohens_d_per_region(df_w[df_w['Subject'].isin(cv_subs)],        regions)
                d_test = mci_cohens_d_per_region(df_w[df_w['Subject'].isin(te_subs)],        regions)

            print(f"\n{cohort} - {atlas}:")
            for label, d in [('pool', d_pool), ('cv', d_cv), ('test', d_test)]:
                r, p, n = improvement_corr(d)
                print(f"  {label:6s} corr( (d_real-d_asl), (d_synth-d_asl) ) = "
                      f"{r:.3f} (p={p:.4g}, n_regions={n})")
                out_rows.append({'cohort': cohort, 'atlas': atlas, 'split': label,
                                 'improvement_corr_r': r, 'p': p, 'n_regions': n})

    df_summary = pd.DataFrame(out_rows)
    df_summary.to_csv(os.path.join(TABLES_DIR,
                                   'cohens_d_improvement_corr_summary.csv'),
                       index=False)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    run_tle_within_subject_suvr()
    run_cohens_d_improvement_corr()
    print("\nDone. Tables saved to:", TABLES_DIR)
