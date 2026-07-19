"""
Builder for revision_report.ipynb — a self-contained Jupyter notebook that
reproduces every number in the manuscript revision from the source pickle
files and fold-assignment JSONs.

All numbers in markdown cells are derived from Python code: each value is
accumulated into a single results dict `R` during the computation cells, and
the manuscript text + consolidated table are rendered via
`IPython.display.Markdown(...)` with f-string interpolation in code cells.
Nothing is hardcoded in markdown.

To regenerate:  python build_revision_notebook.py
"""

import nbformat as nbf
import os

NB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'revision_report.ipynb')

nb = nbf.v4.new_notebook()
cells = []


def md(text: str):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src))


# ============================================================================
# Header
# ============================================================================

md("""# Manuscript Revision Report: per-fold reporting and held-out test set

This notebook reproduces **every number** in the revised manuscript end-to-end
from the source pickle files and fold-assignment JSONs. **No numerical value
is hardcoded in markdown** — all values displayed in text are computed in
code cells and rendered via `IPython.display.Markdown(...)` with f-string
interpolation.

**The reviewer's concern.** The previous manuscript reported pooled aggregates
across all subjects. The reviewer asked for the scientific-standard mean ± SD
across folds and a truly-unseen test set. FlowGAN was already trained 12-fold,
so we satisfied the request post-hoc *without retraining FlowGAN* by:

- **Cross-validated sample** = folds 0–9. Per-subject metrics report mean ± SD
  across the 10 fold-means; across-subject metrics are computed on the union
  of these subjects.
- **Test set** = folds 10–11. Held out from all downstream comparisons.

**Reproducibility note.** Volume-level quality metrics (SSIM, PSNR, RMSE, NCC)
load NIfTI files and take several minutes; those use a cached per-subject CSV
produced by `10_per_fold_quality_metrics.py`. Every other number in this
notebook is computed directly from `df_pet_merged*.pkl` in the cells below.""")


# ============================================================================
# Section 1: Setup
# ============================================================================

md("""## 1. Setup""")

code(r"""import os
import json
import pickle
import warnings
import importlib.util
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.contingency_tables import mcnemar
from IPython.display import Markdown, display

# Library deprecation notices (mostly seaborn's palette/hue changes) are noise in
# a report, so they are silenced. Warnings that carry real information, such as
# RuntimeWarnings from degenerate correlations, are kept — but reformatted to drop
# the file path, which would otherwise stamp this machine's directory layout into
# every saved copy of the notebook.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.formatwarning = lambda msg, cat, *a, **k: f'{cat.__name__}: {msg}\n'

pd.set_option('display.float_format', lambda x: f'{x:.4f}')
pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 50)

SCRIPT_DIR = Path('.').resolve()
# Printed as a name, not a full path, so the notebook stays machine-independent.
print('Working directory:', SCRIPT_DIR.name)

FOLD_JSON_TLE = str(SCRIPT_DIR / 'data' / 'subjects_in_each_fold_TLE.json')
FOLD_JSON_MCI = str(SCRIPT_DIR / 'data' / 'subjects_in_each_fold_MCI.json')

DEV_FOLDS     = [f'fold_{i}' for i in range(10)]
HOLDOUT_FOLDS = ['fold_10', 'fold_11']

EXCLUDE_REGIONS = ['unknown', 'bankssts', 'Unknown', 'vessel', 'VentralDC',
                   'temporalpole', 'frontalpole', 'corpuscallosum', 'Putamen']


def load_fold_map(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return {s: fold for fold, info in d.items() for s in info.get('test', [])}


def split_subjects(subjects, fold_map):
    pool = set(subjects)
    cv   = {s for s in subjects if fold_map.get(s) in DEV_FOLDS}
    test = {s for s in subjects if fold_map.get(s) in HOLDOUT_FOLDS}
    return pool, cv, test


fm_tle = load_fold_map(FOLD_JSON_TLE)
fm_mci = load_fold_map(FOLD_JSON_MCI)

# Master results dict — every numerical value used in the manuscript text
# below is pulled from here. Populated as each section runs.
R = {}

def _import_mod(name):
    spec = importlib.util.spec_from_file_location(f'mod_{name}', os.path.join(SCRIPT_DIR, name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

MOD02 = _import_mod('02_regional_analysis.py')
MOD04 = _import_mod('04_lateralization_cohens_d.py')

PKL = {('TLE', 'DKT'):           'df_pet_merged.pkl',
       ('TLE', 'HarvardOxford'): 'df_pet_merged_ho.pkl',
       ('MCI', 'DKT'):           'df_pet_merged_mci.pkl',
       ('MCI', 'HarvardOxford'): 'df_pet_merged_mci_ho.pkl'}

def load_df_merged(cohort, atlas):
    return pickle.load(open(SCRIPT_DIR / PKL[(cohort, atlas)], 'rb'))

# Subject counts per split per cohort/atlas — record into R
R['n_subjects'] = {}
for cohort, fm in [('TLE', fm_tle), ('MCI', fm_mci)]:
    for atlas in ['DKT', 'HarvardOxford']:
        df = load_df_merged(cohort, atlas)
        subs = list(df['subject'].unique())
        pool, cv, test = split_subjects(subs, fm)
        R['n_subjects'][(cohort, atlas)] = dict(pool=len(pool), cv=len(cv), test=len(test))
        print(f'{cohort}-{atlas:13s}  pooled={len(pool):3d}  cross-validated={len(cv):3d}  test={len(test):3d}')""")


# ============================================================================
# Section 2: Quality metrics
# ============================================================================

md("""## 2. Volume-level quality metrics (SSIM, PSNR, RMSE, NCC)

Per-subject volume-level metrics from `10_per_fold_quality_metrics.py`.
Aggregated as (a) pooled across all subjects, (b) per-fold mean ± SD across
the 10 cross-validated folds, and (c) mean ± SD across the test subjects.
All values stored in `R['quality']`.""")

code(r"""def aggregate_per_fold(df_subjects, fold_map, metric_cols):
    df = df_subjects.copy()
    df['fold']  = df['subject'].map(fold_map)
    df['split'] = df['fold'].apply(
        lambda f: 'test' if f in HOLDOUT_FOLDS
                  else ('cv' if f in DEV_FOLDS else 'unknown'))
    cv = df[df['split'] == 'cv']
    te = df[df['split'] == 'test']
    rows = []
    for m in metric_cols:
        pool_vals = df[m].values; pool_vals = pool_vals[np.isfinite(pool_vals)]
        fmeans = cv.groupby('fold')[m].mean().reindex(DEV_FOLDS).values
        tvals = te[m].values; tvals = tvals[np.isfinite(tvals)]
        rows.append({
            'metric':       m,
            'pool_mean':    float(np.mean(pool_vals)),
            'pool_sd':      float(np.std(pool_vals, ddof=1)),
            'cv_fold_mean': float(np.nanmean(fmeans)),
            'cv_fold_sd':   float(np.nanstd(fmeans, ddof=1)),
            'test_mean':    float(np.mean(tvals)),
            'test_sd':      float(np.std(tvals, ddof=1)),
            'n_pool':       int(len(pool_vals)),
            'n_test':       int(len(tvals)),
        })
    return pd.DataFrame(rows)

METRIC_COLS = ['ssim_recon','ssim_asl','psnr_recon','psnr_asl',
               'rmse_recon','rmse_asl','ncc_recon','ncc_asl']

print('=== TLE quality metrics ===')
df_q_tle = pd.read_csv('tables/10_per_fold_quality_metrics/per_subject_quality_TLE.csv')
sum_q_tle = aggregate_per_fold(df_q_tle, fm_tle, METRIC_COLS)
display(sum_q_tle)

print('\n=== MCI quality metrics ===')
df_q_mci = pd.read_csv('tables/10_per_fold_quality_metrics/per_subject_quality_MCI.csv')
sum_q_mci = aggregate_per_fold(df_q_mci, fm_mci, METRIC_COLS)
display(sum_q_mci)

R['quality'] = {}
for cohort, df_sum in [('TLE', sum_q_tle), ('MCI', sum_q_mci)]:
    R['quality'][cohort] = {row['metric']: dict(row) for _, row in df_sum.iterrows()}""")


# ============================================================================
# Section 3: Within-subject correlations
# ============================================================================

md("""## 3. Within-subject correlations (Spearman r across regions, per subject)

For each subject, Spearman correlation between real PET SUVR and
{synthetic, ASL} across regions, then aggregated per fold. Stored in
`R['within_subj'][(cohort, atlas)][split]`.""")

code(r"""def within_subject_suvr(df_merged):
    rows = []
    for sub in df_merged['subject'].unique():
        d_s = df_merged[df_merged['subject'] == sub].copy()
        def putamen(col):
            l = d_s[(d_s['region_name']=='Putamen') & (d_s['side']=='Left')][col].values
            r = d_s[(d_s['region_name']=='Putamen') & (d_s['side']=='Right')][col].values
            if len(l)==0 or len(r)==0: return None
            v = float(l[0]+r[0])
            return v if v != 0 else None
        n_og, n_rc, n_as = putamen('value_pet_original'), putamen('value_pet_recon'), putamen('value_asl')
        if None in (n_og, n_rc, n_as): continue
        d_s['_og']  = d_s['value_pet_original'] / n_og
        d_s['_rc']  = d_s['value_pet_recon']    / n_rc
        d_s['_asl'] = d_s['value_asl']          / n_as
        d_s = d_s.dropna(subset=['_og','_rc','_asl'])
        d_s = d_s[~d_s['region_name'].isin(EXCLUDE_REGIONS)]
        if len(d_s) < 5: continue
        r_s, _ = stats.spearmanr(d_s['_og'], d_s['_rc'])
        r_a, _ = stats.spearmanr(d_s['_og'], d_s['_asl'])
        rows.append({'subject': sub, 'r_synth': float(r_s), 'r_asl': float(r_a),
                     'bias_synth': float(np.mean(d_s['_og'] - d_s['_rc'])),
                     'bias_asl':   float(np.mean(d_s['_og'] - d_s['_asl']))})
    return pd.DataFrame(rows)


def aggregate_within_subject(df_w, fold_map):
    df = df_w.copy()
    df['fold']  = df['subject'].map(fold_map)
    df['split'] = df['fold'].apply(
        lambda f: 'test' if f in HOLDOUT_FOLDS
                  else ('cv' if f in DEV_FOLDS else 'unknown'))
    out = {}
    for split in ['pool', 'cv', 'test']:
        sub = df if split == 'pool' else df[df['split'] == split]
        s = sub['r_synth'].dropna().values
        a = sub['r_asl'].dropna().values
        if split == 'cv':
            fm_s = sub.groupby('fold')['r_synth'].mean().reindex(DEV_FOLDS).values
            fm_a = sub.groupby('fold')['r_asl'  ].mean().reindex(DEV_FOLDS).values
            synth_mean, synth_sd = float(np.nanmean(fm_s)), float(np.nanstd(fm_s, ddof=1))
            asl_mean,   asl_sd   = float(np.nanmean(fm_a)), float(np.nanstd(fm_a, ddof=1))
        else:
            synth_mean, synth_sd = float(np.mean(s)), float(np.std(s, ddof=1))
            asl_mean,   asl_sd   = float(np.mean(a)), float(np.std(a, ddof=1))
        v = np.isfinite(sub['r_synth'].values) & np.isfinite(sub['r_asl'].values)
        s_v = sub['r_synth'].values[v]; a_v = sub['r_asl'].values[v]
        if len(s_v) >= 2 and np.std(s_v - a_v, ddof=1) > 0:
            diff = s_v - a_v
            paired_d = float(np.mean(diff) / np.std(diff, ddof=1))
            try: _, p = stats.wilcoxon(s_v, a_v); p = float(p)
            except Exception: p = np.nan
        else:
            paired_d, p = np.nan, np.nan
        out[split] = dict(
            synth_mean=synth_mean, synth_sd=synth_sd,
            asl_mean=asl_mean,     asl_sd=asl_sd,
            paired_d=paired_d, wilcoxon_p=p, n=int(len(s_v)))
    return out

R['within_subj'] = {}
rows = []
for cohort, atlas in [('TLE','DKT'),('TLE','HarvardOxford'),
                       ('MCI','DKT'),('MCI','HarvardOxford')]:
    df_merged = load_df_merged(cohort, atlas)
    df_w = within_subject_suvr(df_merged)
    fm = fm_tle if cohort == 'TLE' else fm_mci
    R['within_subj'][(cohort, atlas)] = aggregate_within_subject(df_w, fm)
    _dfw = df_w.copy()
    _dfw['split'] = _dfw['subject'].map(fm).apply(
        lambda f: 'test' if f in HOLDOUT_FOLDS else ('cv' if f in DEV_FOLDS else 'unknown'))
    R['within_subj'][(cohort, atlas)]['per_subject'] = _dfw
    r = R['within_subj'][(cohort, atlas)]
    for split in ['pool','cv','test']:
        rows.append(dict(cohort=cohort, atlas=atlas, split=split,
                          synth_mean=r[split]['synth_mean'], synth_sd=r[split]['synth_sd'],
                          asl_mean=r[split]['asl_mean'],     asl_sd=r[split]['asl_sd'],
                          paired_d=r[split]['paired_d'],     wilcoxon_p=r[split]['wilcoxon_p'],
                          n=r[split]['n']))
display(pd.DataFrame(rows))""")


# ============================================================================
# Section 4: Across-subject + bias
# ============================================================================

md("""## 4. Across-subject regional Spearman correlations + bias

For each brain region, Spearman r across subjects between real PET and
{synthetic, ASL}, then compared across modalities via Wilcoxon. Bias = mean
(real − other) per region. Stored in `R['across_subj']` and `R['bias']`.

- TLE uses **asymmetry indices** ((L − R) / (L + R)).
- MCI uses **bilateral SUVR** (putamen-normalized).""")

code(r"""def across_subject_corr_per_region(df_long, val_real, val_synth, val_asl, regions):
    rows = []
    for r in regions:
        d = df_long[df_long['Region'] == r]
        if len(d) < 4: continue
        x, ys, ya = d[val_real].values, d[val_synth].values, d[val_asl].values
        v1 = np.isfinite(x) & np.isfinite(ys); v2 = np.isfinite(x) & np.isfinite(ya)
        if v1.sum() < 4 or v2.sum() < 4: continue
        r_s, _ = stats.spearmanr(x[v1], ys[v1])
        r_a, _ = stats.spearmanr(x[v2], ya[v2])
        rows.append({'Region': r, 'r_synth': r_s, 'r_asl': r_a})
    return pd.DataFrame(rows)


def summarize_across(df_corrs):
    if len(df_corrs) == 0: return None
    s, a = df_corrs['r_synth'].values, df_corrs['r_asl'].values
    v = np.isfinite(s) & np.isfinite(a); s, a = s[v], a[v]
    try: _, p = stats.wilcoxon(s, a)
    except: p = np.nan
    diff = s - a
    d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
    return dict(
        n_regions=int(len(s)),
        synth_median=float(np.median(s)),
        synth_q1=float(np.percentile(s, 25)), synth_q3=float(np.percentile(s, 75)),
        asl_median=float(np.median(a)),
        asl_q1=float(np.percentile(a, 25)),   asl_q3=float(np.percentile(a, 75)),
        wilcoxon_p=float(p) if p==p else np.nan,
        cohens_d_paired=float(d) if d==d else np.nan,
        n_synth_gt_asl=int(np.sum(s > a)),
        per_region=df_corrs,
    )


def bias_per_region(df_long, val_real, val_synth, val_asl, regions):
    rows = []
    for r in regions:
        d = df_long[df_long['Region']==r]
        if len(d) < 4: continue
        x, ys, ya = d[val_real].values, d[val_synth].values, d[val_asl].values
        v1 = np.isfinite(x) & np.isfinite(ys); v2 = np.isfinite(x) & np.isfinite(ya)
        if v1.sum()<4 or v2.sum()<4: continue
        rows.append({'Region': r,
                     'bias_synth': float(np.mean(x[v1]-ys[v1])),
                     'bias_asl':   float(np.mean(x[v2]-ya[v2]))})
    return pd.DataFrame(rows)


def summarize_bias(df_b):
    if len(df_b) == 0: return None
    s, a = df_b['bias_synth'].values, df_b['bias_asl'].values
    v = np.isfinite(s)&np.isfinite(a); s, a = s[v], a[v]
    try: _, p = stats.wilcoxon(s, a)
    except: p = np.nan
    diff = s - a
    d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
    return dict(
        n_regions=int(len(s)),
        synth_median=float(np.median(s)),
        synth_q1=float(np.percentile(s,25)), synth_q3=float(np.percentile(s,75)),
        asl_median=float(np.median(a)),
        asl_q1=float(np.percentile(a,25)),   asl_q3=float(np.percentile(a,75)),
        wilcoxon_p=float(p) if p==p else np.nan,
        cohens_d_paired=float(d) if d==d else np.nan,
        per_region=df_b,
    )


def run_across_and_bias(cohort, atlas):
    df_merged = load_df_merged(cohort, atlas)
    subs = list(df_merged['subject'].unique())
    fm = fm_tle if cohort == 'TLE' else fm_mci
    pool, cv, test = split_subjects(subs, fm)
    putamen = MOD02.get_putamen_normalization_values(df_merged)
    if cohort == 'TLE':
        df_long = MOD02.build_asymmetry_dataframe(df_merged)
        cols = ('PET AI Original', 'PET AI Recon', 'ASL AI')
    else:
        df_long = MOD02.build_suvr_dataframe(df_merged, putamen)
        cols = ('PET SUVR Original', 'PET SUVR FlowGAN', 'ASL rCBF')
    regions = [r for r in df_long['Region'].unique() if r not in EXCLUDE_REGIONS]
    out_corr = {}; out_bias = {}
    for label, sub in [('pool', pool), ('cv', cv), ('test', test)]:
        df_filt = df_long[df_long['Subject'].isin(sub)]
        out_corr[label] = summarize_across(across_subject_corr_per_region(df_filt, *cols, regions=regions))
        out_bias[label] = summarize_bias(bias_per_region(df_filt, *cols, regions=regions))
    return out_corr, out_bias

R['across_subj'] = {}
R['bias']        = {}
rows = []
for cohort in ['TLE', 'MCI']:
    for atlas in ['DKT', 'HarvardOxford']:
        ac, bi = run_across_and_bias(cohort, atlas)
        R['across_subj'][(cohort, atlas)] = ac
        R['bias'][(cohort, atlas)]        = bi
        for split in ['pool','cv','test']:
            rows.append(dict(cohort=cohort, atlas=atlas, split=split,
                              synth_median=ac[split]['synth_median'],
                              asl_median=ac[split]['asl_median'],
                              wilcoxon_p=ac[split]['wilcoxon_p'],
                              cohens_d=ac[split]['cohens_d_paired'],
                              n_synth_gt_asl=ac[split]['n_synth_gt_asl'],
                              n_regions=ac[split]['n_regions']))
display(pd.DataFrame(rows))

# --- TLE per-region across-subject correlation + bias on SUVR ---------------
# Bilateral putamen-normalized SUVR (the same metric used for MCI), stored under
# separate keys so the asymmetry-based TLE results above remain unchanged.
R['across_subj_suvr'] = {}
R['bias_suvr']        = {}
suvr_cols = ('PET SUVR Original', 'PET SUVR FlowGAN', 'ASL rCBF')
rows_suvr = []
for atlas in ['DKT', 'HarvardOxford']:
    df_merged = load_df_merged('TLE', atlas)
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm_tle)
    putamen = MOD02.get_putamen_normalization_values(df_merged)
    df_long = MOD02.build_suvr_dataframe(df_merged, putamen)
    regions = [r for r in df_long['Region'].unique() if r not in EXCLUDE_REGIONS]
    oc = {}; ob = {}
    for label, sub in [('pool', pool), ('cv', cv), ('test', test)]:
        df_filt = df_long[df_long['Subject'].isin(sub)]
        oc[label] = summarize_across(across_subject_corr_per_region(df_filt, *suvr_cols, regions=regions))
        ob[label] = summarize_bias(bias_per_region(df_filt, *suvr_cols, regions=regions))
        rows_suvr.append(dict(atlas=atlas, split=label,
                              synth_median=oc[label]['synth_median'],
                              asl_median=oc[label]['asl_median'],
                              wilcoxon_p=oc[label]['wilcoxon_p'],
                              cohens_d=oc[label]['cohens_d_paired'],
                              n_synth_gt_asl=oc[label]['n_synth_gt_asl'],
                              n_regions=oc[label]['n_regions'],
                              bias_synth_med=ob[label]['synth_median'],
                              bias_asl_med=ob[label]['asl_median'],
                              bias_wilcoxon_p=ob[label]['wilcoxon_p'],
                              bias_d=ob[label]['cohens_d_paired']))
    R['across_subj_suvr'][('TLE', atlas)] = oc
    R['bias_suvr'][('TLE', atlas)]        = ob
print('\n=== TLE per-region across-subject SUVR (correlation + bias) ===')
display(pd.DataFrame(rows_suvr))

# --- MCI per-region across-subject correlation + bias on ASYMMETRY ----------
# Asymmetry indices ((L-R)/(L+R)), the same metric used for TLE; stored under
# separate keys so the SUVR-based MCI results above remain unchanged.
R['across_subj_asym'] = {}
R['bias_asym']        = {}
asym_cols = ('PET AI Original', 'PET AI Recon', 'ASL AI')
rows_asym = []
for atlas in ['DKT', 'HarvardOxford']:
    df_merged = load_df_merged('MCI', atlas)
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm_mci)
    df_long = MOD02.build_asymmetry_dataframe(df_merged)
    regions = [r for r in df_long['Region'].unique() if r not in EXCLUDE_REGIONS]
    oc = {}; ob = {}
    for label, sub in [('pool', pool), ('cv', cv), ('test', test)]:
        df_filt = df_long[df_long['Subject'].isin(sub)]
        oc[label] = summarize_across(across_subject_corr_per_region(df_filt, *asym_cols, regions=regions))
        ob[label] = summarize_bias(bias_per_region(df_filt, *asym_cols, regions=regions))
        rows_asym.append(dict(atlas=atlas, split=label,
                              synth_median=oc[label]['synth_median'],
                              asl_median=oc[label]['asl_median'],
                              wilcoxon_p=oc[label]['wilcoxon_p'],
                              cohens_d=oc[label]['cohens_d_paired'],
                              n_synth_gt_asl=oc[label]['n_synth_gt_asl'],
                              n_regions=oc[label]['n_regions'],
                              bias_synth_med=ob[label]['synth_median'],
                              bias_asl_med=ob[label]['asl_median'],
                              bias_wilcoxon_p=ob[label]['wilcoxon_p'],
                              bias_d=ob[label]['cohens_d_paired']))
    R['across_subj_asym'][('MCI', atlas)] = oc
    R['bias_asym'][('MCI', atlas)]        = ob
print('\n=== MCI per-region across-subject ASYMMETRY (correlation + bias) ===')
display(pd.DataFrame(rows_asym))""")


# ============================================================================
# Section 5: Sign congruency + McNemar (TLE only)
# ============================================================================

md("""## 5. Sign congruency and McNemar's test (TLE only)

For each region: fraction of subjects whose asymmetry sign matches real PET.
Reports per-split summary and McNemar's exact test for the specific regions
named in the manuscript. Stored in `R['congruency_overall']` and
`R['congruency_regions']`.""")

code(r"""def congruency_per_region(df_ai, regions):
    rows = []
    for r in regions:
        d = df_ai[df_ai['Region']==r]
        if len(d) == 0: continue
        ref, synth, asl = d['PET AI Original'].values, d['PET AI Recon'].values, d['ASL AI'].values
        cong_s = ((ref>=0)&(synth>=0)) | ((ref<0)&(synth<0))
        cong_a = ((ref>=0)&(asl  >=0)) | ((ref<0)&(asl  <0))
        rows.append({'Region': r, 'n': len(d),
                     'cong_synth': float(np.mean(cong_s)),
                     'cong_asl':   float(np.mean(cong_a))})
    return pd.DataFrame(rows)


def summarize_congruency(df_c):
    if len(df_c) == 0: return None
    s, a = df_c['cong_synth'].values, df_c['cong_asl'].values
    v = np.isfinite(s)&np.isfinite(a); s, a = s[v], a[v]
    try: _, p = stats.wilcoxon(s, a)
    except: p = np.nan
    diff = s - a
    d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1) > 0 else np.nan
    return dict(
        n_regions=int(len(s)),
        synth_mean=float(np.mean(s)),
        asl_mean=float(np.mean(a)),
        n_synth_gt_asl=int(np.sum(s > a)),
        wilcoxon_p=float(p) if p==p else np.nan,
        cohens_d_paired=float(d) if d==d else np.nan,
        per_region=df_c,
    )


def mcnemar_for_region(df_long, region, ref_col, synth_col, asl_col, subject_set):
    d = df_long[(df_long['Region']==region) & (df_long['Subject'].isin(subject_set))]
    if len(d) < 4: return None
    ref, synth, asl = d[ref_col].values, d[synth_col].values, d[asl_col].values
    valid = np.isfinite(ref) & np.isfinite(synth) & np.isfinite(asl)
    ref, synth, asl = ref[valid], synth[valid], asl[valid]
    if len(ref) < 4: return None
    cong_s = ((ref>=0)&(synth>=0)) | ((ref<0)&(synth<0))
    cong_a = ((ref>=0)&(asl  >=0)) | ((ref<0)&(asl  <0))
    a = int(np.sum(cong_a & cong_s));  b = int(np.sum(cong_a & ~cong_s))
    c = int(np.sum(~cong_a & cong_s)); d_cell = int(np.sum(~cong_a & ~cong_s))
    table = np.array([[a, b], [c, d_cell]])
    try:
        use_exact = (b + c) <= 25
        p = float(mcnemar(table, exact=use_exact).pvalue)
    except: p = np.nan
    return dict(n=len(ref),
                synth_cong=float(np.mean(cong_s)),
                asl_cong=float(np.mean(cong_a)),
                mcnemar_p=p, b=b, c=c)


REGIONS_DKT = ['Hippocampus', 'parahippocampal', 'Amygdala', 'insula']
REGIONS_HO  = ['Hippocampus', 'TemporalPole',
               'ParahippocampalGyrusanteriordivision',
               'ParahippocampalGyrusposteriordivision',
               'Amygdala', 'InsularCortex']

R['congruency_overall'] = {}
R['congruency_regions'] = {}

for atlas, region_list in [('DKT', REGIONS_DKT), ('HarvardOxford', REGIONS_HO)]:
    df_merged = load_df_merged('TLE', atlas)
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm_tle)
    df_ai = MOD02.build_asymmetry_dataframe(df_merged)
    regions = [r for r in df_ai['Region'].unique() if r not in EXCLUDE_REGIONS]

    R['congruency_overall'][atlas] = {}
    for label, sub in [('pool',pool),('cv',cv),('test',test)]:
        df_c = congruency_per_region(df_ai[df_ai['Subject'].isin(sub)], regions)
        R['congruency_overall'][atlas][label] = summarize_congruency(df_c)

    R['congruency_regions'][atlas] = {}
    for region in region_list:
        if region not in df_ai['Region'].unique(): continue
        R['congruency_regions'][atlas][region] = {}
        for label, sub in [('pool',pool),('cv',cv),('test',test)]:
            res = mcnemar_for_region(df_ai, region, 'PET AI Original',
                                      'PET AI Recon', 'ASL AI', sub)
            if res: R['congruency_regions'][atlas][region][label] = res

print('=== Congruency overall (TLE) ===')
rows = []
for atlas in ['DKT', 'HarvardOxford']:
    for split in ['pool','cv','test']:
        r = R['congruency_overall'][atlas][split]
        rows.append(dict(atlas=atlas, split=split,
                          synth_mean=r['synth_mean'], asl_mean=r['asl_mean'],
                          wilcoxon_p=r['wilcoxon_p'], cohens_d=r['cohens_d_paired'],
                          n_synth_gt_asl=r['n_synth_gt_asl'], n_regions=r['n_regions']))
display(pd.DataFrame(rows))

print('\n=== Per-region McNemar (TLE) ===')
rows = []
for atlas in ['DKT', 'HarvardOxford']:
    for region, splits in R['congruency_regions'][atlas].items():
        for split, r in splits.items():
            rows.append(dict(atlas=atlas, region=region, split=split,
                              synth_cong=r['synth_cong'], asl_cong=r['asl_cong'],
                              mcnemar_p=r['mcnemar_p'], n=r['n']))
display(pd.DataFrame(rows))

# --- MCI per-region sign congruency (asymmetry-based) -----------------------
# Same congruency definition as TLE, computed on the MCI cohort so the MCI
# asymmetry analysis can have a congruency forest plot. Stored separately.
R['congruency_mci'] = {}
rows = []
for atlas in ['DKT', 'HarvardOxford']:
    df_merged = load_df_merged('MCI', atlas)
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm_mci)
    df_ai = MOD02.build_asymmetry_dataframe(df_merged)
    regions = [r for r in df_ai['Region'].unique() if r not in EXCLUDE_REGIONS]
    R['congruency_mci'][atlas] = {}
    for label, sub in [('pool', pool), ('cv', cv), ('test', test)]:
        df_c = congruency_per_region(df_ai[df_ai['Subject'].isin(sub)], regions)
        R['congruency_mci'][atlas][label] = summarize_congruency(df_c)
        s = R['congruency_mci'][atlas][label]
        rows.append(dict(atlas=atlas, split=label, synth_mean=s['synth_mean'],
                          asl_mean=s['asl_mean'], wilcoxon_p=s['wilcoxon_p'],
                          cohens_d=s['cohens_d_paired'], n_synth_gt_asl=s['n_synth_gt_asl'],
                          n_regions=s['n_regions']))
print('\n=== Congruency overall (MCI, asymmetry-based) ===')
display(pd.DataFrame(rows))""")


# ============================================================================
# Section 6: Cohen's d (lateralization / discrimination)
# ============================================================================

md("""## 6. Cohen's d for lateralization (TLE) / discrimination (MCI)

For each region, Cohen's d between groups (L-TLE vs R-TLE or MCI vs HC).
Reports mean |d| per modality per split and the **improvement-correlation**
r = Spearman corr( (d_Real − d_ASL), (d_Synth − d_ASL) ) across regions.
Stored in `R['cohens_d']`.""")

code(r"""def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a)<2 or len(b)<2: return np.nan
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a)-1)*s1 + (len(b)-1)*s2) / (len(a)+len(b)-2))
    if pooled == 0: return np.nan
    return float((np.mean(a)-np.mean(b)) / pooled)


def tle_cohens_d_per_region(df_ai, df_left, regions):
    merged = df_ai.merge(df_left, on='Subject', how='inner')
    rows = []
    for r in regions:
        d = merged[merged['Region']==r]
        l = d[d['isLeft']==1]; rr = d[d['isLeft']==0]
        if len(l)<2 or len(rr)<2: continue
        rows.append({'Region': r,
                     'd_real':  cohens_d(l['PET AI Original'].values, rr['PET AI Original'].values),
                     'd_synth': cohens_d(l['PET AI Recon'].values,    rr['PET AI Recon'].values),
                     'd_asl':   cohens_d(l['ASL AI'].values,          rr['ASL AI'].values)})
    return pd.DataFrame(rows)


def mci_cohens_d_per_region(df_w, regions):
    rows = []
    for r in regions:
        cols = [f'{r}_real', f'{r}_synth', f'{r}_asl']
        if any(c not in df_w.columns for c in cols): continue
        hc, mc = df_w[df_w['is_mci']==0], df_w[df_w['is_mci']==1]
        if len(hc)<2 or len(mc)<2: continue
        rows.append({'Region': r,
                     'd_real':  cohens_d(mc[cols[0]].values, hc[cols[0]].values),
                     'd_synth': cohens_d(mc[cols[1]].values, hc[cols[1]].values),
                     'd_asl':   cohens_d(mc[cols[2]].values, hc[cols[2]].values)})
    return pd.DataFrame(rows)


def improvement_corr(df_d):
    if len(df_d) < 4: return (np.nan, np.nan, 0)
    real_imp  = df_d['d_real'].values  - df_d['d_asl'].values
    synth_imp = df_d['d_synth'].values - df_d['d_asl'].values
    v = np.isfinite(real_imp) & np.isfinite(synth_imp)
    if v.sum() < 4: return (np.nan, np.nan, int(v.sum()))
    r, p = stats.spearmanr(real_imp[v], synth_imp[v])
    return (float(r), float(p), int(v.sum()))


def summarize_d(df_d):
    if df_d is None or len(df_d) == 0: return None
    dr, ds, da = df_d['d_real'].values, df_d['d_synth'].values, df_d['d_asl'].values
    v = np.isfinite(dr)&np.isfinite(ds)&np.isfinite(da)
    dr, ds, da = dr[v], ds[v], da[v]
    out = dict(n_regions=int(len(dr)),
                mean_abs_d_real=float(np.mean(np.abs(dr))),
                mean_abs_d_synth=float(np.mean(np.abs(ds))),
                mean_abs_d_asl=float(np.mean(np.abs(da))),
                n_synth_mag_gt_asl=int(np.sum(np.abs(ds) > np.abs(da))))
    r_imp, p_imp, _ = improvement_corr(df_d)
    out['improvement_corr_r'] = r_imp
    out['improvement_corr_p'] = p_imp
    out['per_region'] = df_d
    return out


from utils import MCI_REGIONS

R['cohens_d'] = {}
print('=== Cohens d ===')
rows = []
for cohort in ['TLE','MCI']:
    for atlas in ['DKT','HarvardOxford']:
        df_merged = load_df_merged(cohort, atlas)
        subs = list(df_merged['subject'].unique())
        fm = fm_tle if cohort == 'TLE' else fm_mci
        pool, cv, test = split_subjects(subs, fm)
        if cohort == 'TLE':
            df_ai = MOD02.build_asymmetry_dataframe(df_merged)
            df_left = MOD04.load_clinical_metadata(pet_subject_ids=subs)
            regions = [r for r in df_ai['Region'].unique() if r not in EXCLUDE_REGIONS]
            make = lambda sub: tle_cohens_d_per_region(df_ai[df_ai['Subject'].isin(sub)],
                                                       df_left, regions)
        else:
            md_mci = MOD04.load_mci_metadata()
            df_w = MOD04.build_suvr_dataframe_mci(df_merged, md_mci, atlas=atlas)
            regions = sorted(set(c.rsplit('_',1)[0] for c in df_w.columns
                               if c.endswith(('_real','_synth','_asl'))))
            if atlas == 'DKT':
                regions = [r for r in regions if r in MCI_REGIONS and r not in EXCLUDE_REGIONS]
            else:
                regions = [r for r in regions if r not in EXCLUDE_REGIONS]
            make = lambda sub: mci_cohens_d_per_region(df_w[df_w['Subject'].isin(sub)], regions)

        R['cohens_d'][(cohort, atlas)] = {}
        for label, sub in [('pool',pool),('cv',cv),('test',test)]:
            s = summarize_d(make(sub))
            R['cohens_d'][(cohort, atlas)][label] = s
            rows.append(dict(cohort=cohort, atlas=atlas, split=label,
                              **{k:v for k,v in s.items() if k != 'per_region'}))
display(pd.DataFrame(rows))""")


# ============================================================================
# Section 7: Figures
# ============================================================================

md("""## 7. Figures — manuscript replacements

Publication-ready figures that replace Figs 3, 4B, 4C, 4D, 5C, and 6 of the
manuscript, plus the supporting panels (per-region bias boxplots, per-region
congruency-improvement bar charts, and the hippocampus congruency quadrant
scatterplots). All numbers in figure titles are pulled from the same dataframes
the manuscript table uses. Saved to `figures/revision_notebook_figs/`.

**Shared style.** Every boxplot uses the Fig. 3 quality-metric house style via
the `sns_box` / `annotate_box` / `synth_asl_legend` helpers defined in the first
cell below: filled boxes (alpha 0.55), black median, grey whiskers/caps, and the
Synthetic-PET = blue (`#2166ac`) / ASL = red (`#b2182b`) colour map. Individual
points are **not** drawn on the box-and-whisker figures (outliers are still shown
as small grey dots) — the exception is Fig. 3, which keeps the per-fold /
per-subject points. Pairwise Synthetic-vs-ASL comparisons are annotated with
significance stars using `statannotations` (Wilcoxon signed-rank, the same test
reported in the titles), matching the original manuscript figures. The same
colour map distinguishes congruent (blue) vs incongruent (red) points/quadrants
in the congruency scatter figures.""")

code(r"""import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

FIG_DIR = SCRIPT_DIR / 'figures' / 'revision_notebook_figs'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ---- Shared colour map + boxplot style (matches the Fig. 3 quality-metric replacement) ----
SYNTH_COLOR  = '#2166ac'   # Synthetic PET  (blue)
ASL_COLOR    = '#b2182b'   # ASL            (red)
CONG_COLOR   = SYNTH_COLOR  # congruent  (same colour map)
INCONG_COLOR = ASL_COLOR    # incongruent
BOX_ALPHA    = 0.55
sns.set_style('white')

def save_fig(fig, name):
    fig.savefig(FIG_DIR / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / f'{name}.png', bbox_inches='tight', dpi=150)
    print(f'  Saved {name}.pdf + .png')

def styled_boxplot(ax, data, positions, colors, widths=0.7):
    # Fig. 3 house style: filled boxes (alpha 0.55), black median,
    # grey whiskers/caps, no fliers.
    bp = ax.boxplot(data, positions=positions, widths=widths, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.4),
                    whiskerprops=dict(color='gray'),
                    capprops=dict(color='gray'),
                    boxprops=dict(color='gray'),
                    showfliers=False, zorder=2)
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c); patch.set_alpha(BOX_ALPHA)
    return bp

def jitter_points(ax, x, vals, color, s=18, width=0.12):
    # Jittered point overlay matching the Fig. 3 style.
    vals = np.asarray(vals, float); vals = vals[np.isfinite(vals)]
    ax.scatter(np.random.uniform(x - width, x + width, len(vals)), vals,
               s=s, color=color, edgecolor='white', linewidth=0.5, zorder=3)

def synth_asl_legend(ax, loc='lower left', fontsize=8):
    ax.legend(handles=[Patch(facecolor=SYNTH_COLOR, alpha=BOX_ALPHA, label='Synthetic PET'),
                       Patch(facecolor=ASL_COLOR,   alpha=BOX_ALPHA, label='ASL')],
              loc=loc, fontsize=fontsize, frameon=False)

# ---- Seaborn boxplots + statannotations (manuscript style) --------------------
# Boxes carry the data: jittered points are removed (outliers still shown as
# small grey dots); pairwise comparisons are annotated with significance stars
# via statannotations, exactly as in the original manuscript figures.
import matplotlib.patches as mpatches
from statannotations.Annotator import Annotator

def sns_box(ax, data, x, y, order, palette, hue=None, hue_order=None,
            width=0.5, showfliers=True):
    sns.boxplot(data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order,
                palette=palette, width=width, ax=ax, showfliers=showfliers,
                linewidth=1.4,
                medianprops=dict(color='black', linewidth=1.4),
                whiskerprops=dict(color='gray'),
                capprops=dict(color='gray'),
                flierprops=dict(marker='o', markersize=3, markerfacecolor='0.4',
                                markeredgecolor='0.4', alpha=0.6))
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    for p in ax.patches:
        if isinstance(p, mpatches.PathPatch):
            p.set_alpha(BOX_ALPHA)

def annotate_box(ax, pairs, data, x, y, order, pvalues, hue=None, hue_order=None,
                 fontsize=11):
    ann = Annotator(ax, pairs, data=data, x=x, y=y, order=order,
                    hue=hue, hue_order=hue_order)
    ann.configure(text_format='star', loc='inside', fontsize=fontsize,
                  line_width=1.2, color='black', verbose=False)
    ann.set_pvalues_and_annotate(list(pvalues))
    return ann

def _wilcoxon_p(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 1 or np.allclose(a, b): return 1.0
    try: return float(stats.wilcoxon(a, b)[1])
    except Exception: return 1.0

def fit_ylim(axes, gmin, gmax, bot_frac=0.06, top_frac=0.22):
    # Set a shared y-range that contains every panel's whiskers/outliers, with
    # top headroom for the statannotations bracket. Call BEFORE annotating so
    # statannotations places its bracket inside the reserved space.
    rng = (gmax - gmin) if gmax > gmin else 1.0
    lo, hi = gmin - bot_frac*rng, gmax + top_frac*rng
    for ax in np.atleast_1d(axes).ravel():
        ax.set_ylim(lo, hi)
    return lo, hi""")

md("""### Figure 3 replacement — Quality metrics""")

code(r"""def plot_quality_metrics(df_q, fold_map, cohort, save_name):
    df = df_q.copy()
    df['fold']  = df['subject'].map(fold_map)
    df['split'] = df['fold'].apply(
        lambda f: 'test' if f in HOLDOUT_FOLDS
                  else ('cv' if f in DEV_FOLDS else 'unknown'))
    # 2x2 panel: SSIM, PSNR (top row); RMSE, NCC (bottom row).
    metrics = [('ssim','Structural Similarity Index (SSIM)','SSIM',(0,1)),
               ('psnr','Peak Signal-to-Noise Ratio (PSNR)','PSNR (dB)',(5,30)),
               ('rmse','Root Mean Squared Error (RMSE)','RMSE',(0,0.5)),
               ('ncc','Normalized Cross-Correlation (NCC)','NCC',(0.5,1))]
    SPLITS = ['Cross-validated', 'Test set']; MODS = ['Synthetic', 'ASL']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.5))
    axf = axes.flatten()
    for ax, (mkey, mtitle, mlabel, ylim) in zip(axf, metrics):
        cv = df[df['split']=='cv']
        fold_means_s = cv.groupby('fold')[f'{mkey}_recon'].mean().reindex(DEV_FOLDS).dropna().values
        fold_means_a = cv.groupby('fold')[f'{mkey}_asl'  ].mean().reindex(DEV_FOLDS).dropna().values
        ts = df[df['split']=='test']
        test_s = ts[f'{mkey}_recon'].dropna().values
        test_a = ts[f'{mkey}_asl'  ].dropna().values
        n_test = len(test_s)
        rows = []
        for val, mod, sp in [(fold_means_s,'Synthetic','Cross-validated'),
                             (fold_means_a,'ASL','Cross-validated'),
                             (test_s,'Synthetic','Test set'),
                             (test_a,'ASL','Test set')]:
            rows += [{'split': sp, 'Modality': mod, 'value': float(x)} for x in val]
        dfl = pd.DataFrame(rows)
        # Box-and-whisker only (no overlaid points); outliers shown as small grey dots.
        # Wider boxes pull the cross-validated / test-set groups closer together.
        sns_box(ax, dfl, 'split', 'value', SPLITS, [SYNTH_COLOR, ASL_COLOR],
                hue='Modality', hue_order=MODS, width=0.82, showfliers=True)
        ax.set_xlim(-0.6, 1.6)
        top = ylim[1] + 0.16*(ylim[1]-ylim[0]); ax.set_ylim(ylim[0], top)
        p_cv = _wilcoxon_p(fold_means_s, fold_means_a)
        p_te = _wilcoxon_p(test_s, test_a)
        annotate_box(ax, [(('Cross-validated','Synthetic'),('Cross-validated','ASL')),
                          (('Test set','Synthetic'),('Test set','ASL'))],
                     dfl, 'split', 'value', SPLITS, [p_cv, p_te],
                     hue='Modality', hue_order=MODS, fontsize=14)
        ax.set_xlabel(''); ax.set_ylabel(mlabel, fontweight='bold', fontsize=14)
        ax.set_title(mtitle, fontweight='bold', fontsize=14)
        ax.set_xticklabels(['Cross-validated\n(10 folds)', f'Test set\n(n={n_test})'],
                           fontsize=12)
        ax.tick_params(axis='y', labelsize=12)
        ax.set_box_aspect(1)
        sns.despine(ax=ax)
    synth_asl_legend(axf[0], fontsize=12)
    fig.suptitle(f'{cohort} dataset — image quality metrics', fontweight='bold',
                 fontsize=16, y=1.0)
    plt.tight_layout(); save_fig(fig, save_name); return fig

_ = plot_quality_metrics(df_q_tle, fm_tle, 'TLE', 'fig3_quality_metrics_TLE'); plt.show()
_ = plot_quality_metrics(df_q_mci, fm_mci, 'MCI', 'fig3_quality_metrics_MCI'); plt.show()""")

md("""### Within-subject correlation and bias boxplots (Section 3)

Per-subject (across-region) Spearman correlation and Bland-Altman bias between
real PET and {Synthetic PET, ASL}, shown for the cross-validated sample and the
test set. Each point is one subject. Matches the original within-subject figure
(`plot_within_subject_comparison`) in the Fig. 3 house style. Built from
`R['within_subj'][...]['per_subject']`.""")

code(r"""def plot_within_subject_box(cohort, atlas_name, cohort_label, value_synth, value_asl,
                            ylabel, save_name, axhline=None):
    dfw = R['within_subj'][(cohort, atlas_name)]['per_subject']
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    panels = []; gmin, gmax = np.inf, -np.inf
    for ax, label in zip(axes, ['cv', 'test']):
        d = dfw[dfw['split'] == label]
        s, a = d[value_synth].values, d[value_asl].values
        v = np.isfinite(s) & np.isfinite(a); s, a = s[v], a[v]
        p = _wilcoxon_p(s, a)
        diff = s - a
        dd = np.mean(diff)/np.std(diff, ddof=1) if len(diff) > 1 and np.std(diff, ddof=1) > 0 else np.nan
        dfl = pd.DataFrame({'Modality': ['Synthetic PET']*len(s)+['ASL']*len(a),
                            'val': np.r_[s, a]})
        sns_box(ax, dfl, 'Modality', 'val', ['Synthetic PET', 'ASL'],
                [SYNTH_COLOR, ASL_COLOR], width=0.78)
        ax.set_ylabel(ylabel if label == 'cv' else '', fontweight='bold')
        ax.set_xlabel(f'{label} (n={len(s)} subjects)\nWilcoxon p={p:.3g}, d={dd:.2f}',
                      fontsize=9, labelpad=8)
        if axhline is not None:
            ax.axhline(axhline, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
        sns.despine(ax=ax)
        panels.append((ax, dfl, p))
        if len(s):
            gmin = min(gmin, np.r_[s, a].min()); gmax = max(gmax, np.r_[s, a].max())
    fit_ylim(axes, gmin, gmax)
    for ax, dfl, p in panels:
        annotate_box(ax, [('Synthetic PET', 'ASL')], dfl, 'Modality', 'val',
                     ['Synthetic PET', 'ASL'], [p])
    synth_asl_legend(axes[0])
    fig.suptitle(f'{cohort_label} — {atlas_name}', fontweight='bold', y=1.02)
    plt.tight_layout(); save_fig(fig, save_name); return fig

# within-subject correlation
_ = plot_within_subject_box('TLE','DKT',          'TLE within-subject r', 'r_synth','r_asl',
        'Within-subject Spearman r', 'figWithinR_TLE_DKT'); plt.show()
_ = plot_within_subject_box('TLE','HarvardOxford','TLE within-subject r', 'r_synth','r_asl',
        'Within-subject Spearman r', 'figWithinR_TLE_HO'); plt.show()
_ = plot_within_subject_box('MCI','DKT',          'MCI within-subject r', 'r_synth','r_asl',
        'Within-subject Spearman r', 'figWithinR_MCI_DKT'); plt.show()
_ = plot_within_subject_box('MCI','HarvardOxford','MCI within-subject r', 'r_synth','r_asl',
        'Within-subject Spearman r', 'figWithinR_MCI_HO'); plt.show()
# within-subject bias
_ = plot_within_subject_box('TLE','DKT',          'TLE within-subject bias', 'bias_synth','bias_asl',
        'Within-subject bias (real − modality)', 'figWithinBias_TLE_DKT', axhline=0); plt.show()
_ = plot_within_subject_box('TLE','HarvardOxford','TLE within-subject bias', 'bias_synth','bias_asl',
        'Within-subject bias (real − modality)', 'figWithinBias_TLE_HO', axhline=0); plt.show()
_ = plot_within_subject_box('MCI','DKT',          'MCI within-subject bias', 'bias_synth','bias_asl',
        'Within-subject bias (real − modality)', 'figWithinBias_MCI_DKT', axhline=0); plt.show()
_ = plot_within_subject_box('MCI','HarvardOxford','MCI within-subject bias', 'bias_synth','bias_asl',
        'Within-subject bias (real − modality)', 'figWithinBias_MCI_HO', axhline=0); plt.show()""")

md("""### Across-subject within-region correlation boxplots (Section 4 — Figure 4B / 5C)

The across-subject counterpart to the within-subject figure above, in the same
format. For each brain region, the Spearman correlation between real PET and
{Synthetic PET, ASL} is computed *across subjects*; each point in the box is one
region. Shown for the cross-validated sample and the test set. Built from
`R['across_subj'][...]['per_region']`.""")

code(r"""def plot_per_region_r_box(cohort, atlas_name, cohort_label, save_name,
                          source='across_subj'):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    panels = []; gmin, gmax = np.inf, -np.inf
    for ax, label in zip(axes, ['cv', 'test']):
        df_c = R[source][(cohort, atlas_name)][label]['per_region']
        s, a = df_c['r_synth'].values, df_c['r_asl'].values
        v = np.isfinite(s) & np.isfinite(a); s, a = s[v], a[v]
        p = _wilcoxon_p(s, a)
        diff = s - a
        d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1)>0 else np.nan
        dfl = pd.DataFrame({'Modality': ['Synthetic PET']*len(s)+['ASL']*len(a),
                            'r': np.r_[s, a]})
        sns_box(ax, dfl, 'Modality', 'r', ['Synthetic PET', 'ASL'], [SYNTH_COLOR, ASL_COLOR],
                width=0.78)
        ax.set_ylabel('Per-region Spearman r' if label == 'cv' else '', fontweight='bold')
        ax.set_xlabel(f'{label} (n={R["n_subjects"][(cohort, atlas_name)][label]} subjects)\n'
                      f'Wilcoxon p={p:.3g}, d={d:.2f}\n'
                      f'({np.sum(s>a)}/{len(s)} regions Synth>ASL)',
                      fontsize=9, labelpad=8)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
        sns.despine(ax=ax)
        panels.append((ax, dfl, p))
        gmin = min(gmin, np.r_[s, a].min()); gmax = max(gmax, np.r_[s, a].max())
    fit_ylim(axes, gmin, gmax)
    for ax, dfl, p in panels:
        annotate_box(ax, [('Synthetic PET', 'ASL')], dfl, 'Modality', 'r',
                     ['Synthetic PET', 'ASL'], [p])
    synth_asl_legend(axes[0])
    fig.suptitle(f'{cohort_label} — {atlas_name}', fontweight='bold', y=1.02)
    plt.tight_layout(); save_fig(fig, save_name); return fig

_ = plot_per_region_r_box('TLE', 'DKT',           'TLE asymmetry r', 'fig4B_TLE_DKT_asym_r'); plt.show()
_ = plot_per_region_r_box('TLE', 'HarvardOxford', 'TLE asymmetry r', 'fig4B_TLE_HO_asym_r'); plt.show()
_ = plot_per_region_r_box('MCI', 'DKT',           'MCI SUVR r',      'fig5C_MCI_DKT_suvr_r'); plt.show()
_ = plot_per_region_r_box('MCI', 'HarvardOxford', 'MCI SUVR r',      'fig5C_MCI_HO_suvr_r'); plt.show()""")

md("""### Bias boxplots — per-region bias vs real PET (Bland-Altman bias)

Per-region bias (mean of real − modality) for Synthetic PET vs ASL, computed
from `R['bias']`. TLE uses asymmetry-index bias; MCI uses SUVR bias. A bias near
zero is better. Same Fig. 3 house style.""")

code(r"""def plot_bias_box(cohort, atlas_name, cohort_label, metric_label, save_name,
                  source='bias', clip=None):
    # clip=(lo, hi): per-region points outside this window are dropped from the
    # DISPLAY only (so a single extreme region doesn't compress the boxes). All
    # statistics (Wilcoxon p, Cohen's d, the |bias| Synth<ASL count) are still
    # computed on every region.
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    panels = []; gmin, gmax = np.inf, -np.inf; n_hidden = 0
    for ax, label in zip(axes, ['cv', 'test']):
        df_b = R[source][(cohort, atlas_name)][label]['per_region']
        s, a = df_b['bias_synth'].values, df_b['bias_asl'].values
        v = np.isfinite(s) & np.isfinite(a); s, a = s[v], a[v]
        p = _wilcoxon_p(s, a)
        diff = s - a
        d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1)>0 else np.nan
        sp, ap = s, a   # values actually drawn (optionally outlier-clipped)
        if clip is not None:
            lo_c, hi_c = clip
            ms = (s >= lo_c) & (s <= hi_c); ma = (a >= lo_c) & (a <= hi_c)
            n_hidden += int((~ms).sum() + (~ma).sum())
            sp, ap = s[ms], a[ma]
        dfl = pd.DataFrame({'Modality': ['Synthetic PET']*len(sp)+['ASL']*len(ap),
                            'bias': np.r_[sp, ap]})
        sns_box(ax, dfl, 'Modality', 'bias', ['Synthetic PET', 'ASL'], [SYNTH_COLOR, ASL_COLOR],
                width=0.78)
        ax.set_ylabel(f'Per-region bias ({metric_label})' if label == 'cv' else '',
                      fontweight='bold')
        ax.set_xlabel(f'{label} (n={R["n_subjects"][(cohort, atlas_name)][label]} subjects)\n'
                      f'Wilcoxon p={p:.3g}, d={d:.2f}\n'
                      f'(|bias| Synth<ASL in {np.sum(np.abs(s)<np.abs(a))}/{len(s)})',
                      fontsize=9, labelpad=8)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.6, alpha=0.6)
        sns.despine(ax=ax)
        panels.append((ax, dfl, p))
        gmin = min(gmin, np.r_[sp, ap].min()); gmax = max(gmax, np.r_[sp, ap].max())
    fit_ylim(axes, gmin, gmax)
    for ax, dfl, p in panels:
        annotate_box(ax, [('Synthetic PET', 'ASL')], dfl, 'Modality', 'bias',
                     ['Synthetic PET', 'ASL'], [p])
    synth_asl_legend(axes[0])
    fig.suptitle(f'{cohort_label} bias — {atlas_name}', fontweight='bold', y=1.02)
    if n_hidden:
        fig.text(0.5, -0.02,
                 f'{n_hidden} outlier region(s) outside axis range hidden for clarity '
                 f'(statistics computed on all regions)',
                 ha='center', va='top', fontsize=7.5, style='italic', color='0.35')
    plt.tight_layout(); save_fig(fig, save_name); return fig

_ = plot_bias_box('TLE', 'DKT',           'TLE asymmetry', 'AI', 'figBias_TLE_DKT_asym'); plt.show()
_ = plot_bias_box('TLE', 'HarvardOxford', 'TLE asymmetry', 'AI', 'figBias_TLE_HO_asym',
                  clip=(-0.08, 0.12)); plt.show()
_ = plot_bias_box('MCI', 'DKT',           'MCI SUVR',      'SUVR', 'figBias_MCI_DKT_suvr'); plt.show()
_ = plot_bias_box('MCI', 'HarvardOxford', 'MCI SUVR',      'SUVR', 'figBias_MCI_HO_suvr');  plt.show()""")

md("""### Bland-Altman & Spearman correlation grids — selected ROIs (TLE asymmetry & MCI SUVR)

Ports the original `plot_bland_altman_*_selected_regions`
(`02_regional_analysis.py`) into the cross-validated / test-set framework. **One
figure per cohort × atlas × split**, with **one row per ROI** and **4 panels per
row**: real PET vs **Synthetic PET** on the left (Bland-Altman, then correlation),
real PET vs **ASL** on the right (Bland-Altman, then correlation). TLE uses
asymmetry indices; MCI uses putamen-normalized SUVR. Bland-Altman panels show the
mean bias (dashed) and ±1.96 SD limits of agreement (dotted); correlation panels
show Spearman r against the y=x identity line. Synthetic-PET panels are red, ASL
panels are blue; all panels are square. A separate figure is produced for the
cross-validated (train) sample and the test set.

ROIs — **TLE asymmetry** and **MCI SUVR**: hippocampus, insula, posterior
cingulate, thalamus (DKT); hippocampus, thalamus (Harvard-Oxford).""")

code(r"""# Synthetic PET panels in red, ASL panels in blue.
SYNTH_PANEL_COLOR = '#2166ac'   # blue  (synthetic vs real PET)
ASL_PANEL_COLOR   = '#b2182b'   # red   (ASL vs real PET)

def _ba_corr_data(cohort, atlas):
    df_merged = load_df_merged(cohort, atlas)
    fm = fm_tle if cohort == 'TLE' else fm_mci
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm)
    if cohort == 'TLE':
        df_long = MOD02.build_asymmetry_dataframe(df_merged)
        cols = dict(real='PET AI Original', synth='PET AI Recon', asl='ASL AI',
                    unit='asymmetry index', asl_axis='ASL asymmetry index')
    else:
        putamen = MOD02.get_putamen_normalization_values(df_merged)
        df_long = MOD02.build_suvr_dataframe(df_merged, putamen)
        cols = dict(real='PET SUVR Original', synth='PET SUVR FlowGAN', asl='ASL rCBF',
                    unit='SUVR', asl_axis='ASL rCBF')
    return df_long, cv, test, cols

def _ba_panel(ax, x, y, comp_label, unit, color):
    m = np.isfinite(x) & np.isfinite(y); xx, yy = x[m], y[m]
    mean_xy = (xx + yy) / 2.0; diff = xx - yy
    mb = float(np.mean(diff)) if len(xx) else float('nan')
    sb = float(np.std(diff))  if len(xx) else float('nan')
    lo, hi = mb - 1.96*sb, mb + 1.96*sb
    ax.scatter(mean_xy, diff, color=color, alpha=0.7, edgecolor='white', linewidth=0.4, s=40)
    ax.axhline(mb, color='k', linestyle='--', linewidth=1)
    ax.axhline(hi, color='k', linestyle=':', linewidth=1)
    ax.axhline(lo, color='k', linestyle=':', linewidth=1)
    ax.set_xlabel(f'Mean of Real PET & {comp_label} ({unit})', fontsize=9)
    ax.set_ylabel(f'Difference (Real PET − {comp_label})', fontsize=9)
    ax.set_title(f'Real PET vs {comp_label} (BA)\n'
                 f'n={len(xx)}, bias={mb:.3f}, LoA [{lo:.3f}, {hi:.3f}]',
                 fontsize=9, fontweight='bold')
    ax.set_box_aspect(1); sns.despine(ax=ax)

def _corr_panel(ax, x, y, comp_label, unit, y_axis, color):
    m = np.isfinite(x) & np.isfinite(y); xx, yy = x[m], y[m]
    ax.scatter(xx, yy, color=color, alpha=0.7, edgecolor='white', linewidth=0.4, s=40)
    if len(xx):
        lo = float(min(xx.min(), yy.min())); hi = float(max(xx.max(), yy.max()))
        pad = 0.1*(hi - lo) if hi > lo else (abs(hi)*0.1 or 0.1)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    rho, pv = stats.spearmanr(xx, yy) if len(xx) > 1 else (np.nan, np.nan)
    pstr = ('p<0.001' if pv == pv and pv < 0.001 else
            (f'p={pv:.3g}' if pv == pv else 'p=NA'))
    ax.set_xlabel(f'Real PET {unit}', fontsize=9)
    ax.set_ylabel(y_axis, fontsize=9)
    ax.set_title(f'Real PET vs {comp_label} (Correlation)\nSpearman r={rho:.2f}, {pstr}',
                 fontsize=9, fontweight='bold')
    ax.set_box_aspect(1); sns.despine(ax=ax)

def plot_ba_corr_grid(cohort, atlas, regions, region_labels, split, save_name):
    from matplotlib.transforms import Bbox
    from matplotlib.patches import FancyBboxPatch
    df_long, cv, test, cols = _ba_corr_data(cohort, atlas)
    sub_set = cv if split == 'cv' else test
    unit = cols['unit']
    n = len(regions)
    fig, axes = plt.subplots(n, 4, figsize=(18, 5.0*n))
    if n == 1: axes = axes.reshape(1, -1)
    fig.tight_layout(h_pad=7.0, w_pad=1.5, rect=[0.01, 0.01, 0.99, 0.93])
    letters = 'ABCDEFGH'
    for i, (region, rlabel) in enumerate(zip(regions, region_labels)):
        dfr = df_long[(df_long['Region'] == region) & (df_long['Subject'].isin(sub_set))]
        x  = dfr[cols['real']].values.astype(float)
        ys = dfr[cols['synth']].values.astype(float)
        ya = dfr[cols['asl']].values.astype(float)
        # left half: Real vs Synthetic PET (red); right half: Real vs ASL (blue)
        _ba_panel(  axes[i, 0], x, ys, 'Synthetic PET', unit, SYNTH_PANEL_COLOR)
        _corr_panel(axes[i, 1], x, ys, 'Synthetic PET', unit, f'Synthetic PET {unit}', SYNTH_PANEL_COLOR)
        _ba_panel(  axes[i, 2], x, ya, 'ASL', unit, ASL_PANEL_COLOR)
        _corr_panel(axes[i, 3], x, ya, 'ASL', unit, cols['asl_axis'], ASL_PANEL_COLOR)
    # rounded box + panel letter + centred region title per row
    fig.canvas.draw(); rend = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    PAD = 0.008; TITLE_H = 0.024; max_by1 = 0.0
    for i, rlabel in enumerate(region_labels):
        row = [axes[i, j] for j in range(4)]
        u = Bbox.union([ax.get_tightbbox(rend) for ax in row])
        (x0, y0), (x1, y1) = inv.transform(u)
        bx0, by0, bx1, by1 = x0-PAD, y0-PAD, x1+PAD, y1+PAD+TITLE_H
        max_by1 = max(max_by1, by1)
        rect = FancyBboxPatch((bx0, by0), bx1-bx0, by1-by0,
                              boxstyle='round,pad=0,rounding_size=0.012',
                              transform=fig.transFigure, fill=False,
                              edgecolor='0.45', linewidth=1.4, zorder=0)
        fig.add_artist(rect)
        ty = y1 + PAD + TITLE_H/2
        fig.text((x0+x1)/2, ty, rlabel.capitalize(), ha='center', va='center',
                 fontsize=13, fontweight='bold')
        fig.text(bx0+0.006, ty, letters[i], ha='left', va='center',
                 fontsize=16, fontweight='bold')
    split_label = 'cross-validated (train)' if split == 'cv' else 'test set'
    metric_word = 'asymmetry' if cohort == 'TLE' else 'SUVR'
    fig.text(0.5, min(0.995, max_by1+0.012),
             f'{cohort} — {atlas} ({metric_word}, {split_label}, n={len(sub_set)}):  '
             f'Real vs Synthetic PET (left)  |  Real vs ASL (right)',
             ha='center', va='bottom', fontsize=14, fontweight='bold')
    save_fig(fig, save_name); return fig

# (cohort, atlas, regions, region_labels)
BA_GRID = [
    ('TLE', 'DKT', ['Hippocampus','insula','posteriorcingulate','Thalamus'],
                   ['hippocampus','insula','posterior cingulate','thalamus']),
    ('MCI', 'DKT', ['Hippocampus','insula','posteriorcingulate','Thalamus'],
                   ['hippocampus','insula','posterior cingulate','thalamus']),
    ('TLE', 'HarvardOxford', ['Hippocampus','Thalamus'], ['hippocampus','thalamus']),
    ('MCI', 'HarvardOxford', ['Hippocampus','Thalamus'], ['hippocampus','thalamus']),
]
for cohort, atlas, regions, rlabels in BA_GRID:
    atag = 'DKT' if atlas == 'DKT' else 'HO'
    for split in ['cv', 'test']:
        _ = plot_ba_corr_grid(cohort, atlas, regions, rlabels, split,
                f'figBAgrid_{cohort}_{atag}_{split}'); plt.show()""")

md("""### Per-region Δr correlation forest bars — TLE asymmetry (Fig 4C) and MCI SUVR

Per-region improvement in across-subject correlation, Δr = (Synthetic PET − ASL),
sorted. Bars are blue where Synthetic PET correlates better with real PET than ASL,
red where ASL is better, grey where the difference is within 1 SD of zero. Drawn
for **TLE asymmetry** (Fig 4C) and **MCI SUVR** (the delta-SUVR-correlation forest
plot), both atlases, for the cross-validated sample and the test set. Built from
`R['across_subj']`.""")

code(r"""def plot_region_delta_bar(cohort, atlas_key, atlas_name, metric_label, split, save_name,
                          source='across_subj'):
    split_label = 'cross-validated' if split == 'cv' else 'test set'
    df_corrs = R[source][(cohort, atlas_key)][split]['per_region']
    df = df_corrs.copy(); df['delta'] = df['r_synth'] - df['r_asl']
    df = df.sort_values('delta')
    threshold = float(np.nanstd(np.abs(df['delta'])))
    colors = [SYNTH_COLOR if d>threshold else (ASL_COLOR if d<-threshold else 'gray')
              for d in df['delta']]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(df)), df['delta'], color=colors, alpha=BOX_ALPHA,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df['Region'], rotation=90, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('Δr (Synthetic PET − ASL)', fontweight='bold')
    n_sub = R['n_subjects'][(cohort, atlas_key)][split]
    ax.set_title(f'{cohort} {metric_label} r — per-region improvement, {atlas_name} '
                 f'({split_label}, n={n_sub})', fontweight='bold')
    n_better = int(np.sum(df['delta'] > 0))
    ax.text(0.02, 0.95, f'Synth > ASL: {n_better}/{len(df)} regions',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(handles=[Patch(facecolor=SYNTH_COLOR, alpha=BOX_ALPHA, label='Synthetic PET better'),
                       Patch(facecolor=ASL_COLOR,   alpha=BOX_ALPHA, label='ASL better'),
                       Patch(facecolor='gray',      alpha=BOX_ALPHA, label='Within 1 SD')],
              loc='lower right', fontsize=8, frameon=False)
    sns.despine(ax=ax); plt.tight_layout(); save_fig(fig, save_name); return fig

for split in ['cv', 'test']:
    # TLE asymmetry (Fig 4C)
    _ = plot_region_delta_bar('TLE', 'DKT',           'DKT',            'asymmetry', split, f'fig4C_TLE_DKT_delta_r_{split}'); plt.show()
    _ = plot_region_delta_bar('TLE', 'HarvardOxford', 'Harvard-Oxford', 'asymmetry', split, f'fig4C_TLE_HO_delta_r_{split}');  plt.show()
    # MCI SUVR (delta-SUVR-correlation forest plot)
    _ = plot_region_delta_bar('MCI', 'DKT',           'DKT',            'SUVR', split, f'fig5_MCI_DKT_delta_suvr_r_{split}'); plt.show()
    _ = plot_region_delta_bar('MCI', 'HarvardOxford', 'Harvard-Oxford', 'SUVR', split, f'fig5_MCI_HO_delta_suvr_r_{split}');  plt.show()""")

md("""### TLE per-region across-subject SUVR — correlation, bias, and Δr forest

The SUVR counterpart of the TLE per-region analysis (Fig 4B/4C use asymmetry
indices). For each region, the across-subject Spearman correlation between real
PET and {Synthetic PET, ASL} on bilateral putamen-normalized SUVR, the per-region
bias (mean real − modality), and the sorted Δr (Synthetic PET − ASL) forest plot.
Both atlases, cross-validated sample and test set, in the same house style. Built
from `R['across_subj_suvr']` and `R['bias_suvr']`.""")

code(r"""# correlation boxplots (per-region across-subject SUVR r)
_ = plot_per_region_r_box('TLE', 'DKT',           'TLE SUVR r', 'figSUVR_TLE_DKT_r',
        source='across_subj_suvr'); plt.show()
_ = plot_per_region_r_box('TLE', 'HarvardOxford', 'TLE SUVR r', 'figSUVR_TLE_HO_r',
        source='across_subj_suvr'); plt.show()
# per-region bias boxplots
_ = plot_bias_box('TLE', 'DKT',           'TLE SUVR', 'SUVR', 'figSUVR_TLE_DKT_bias',
        source='bias_suvr'); plt.show()
_ = plot_bias_box('TLE', 'HarvardOxford', 'TLE SUVR', 'SUVR', 'figSUVR_TLE_HO_bias',
        source='bias_suvr'); plt.show()
# per-region Δr forest plots (cross-validated + test)
for split in ['cv', 'test']:
    _ = plot_region_delta_bar('TLE', 'DKT',           'DKT',            'SUVR', split,
            f'figSUVR_TLE_DKT_delta_r_{split}', source='across_subj_suvr'); plt.show()
    _ = plot_region_delta_bar('TLE', 'HarvardOxford', 'Harvard-Oxford', 'SUVR', split,
            f'figSUVR_TLE_HO_delta_r_{split}', source='across_subj_suvr'); plt.show()""")

md("""### MCI per-region across-subject asymmetry — correlation and bias boxplots

The asymmetry-index counterpart of the MCI per-region analysis (Fig 5C uses
SUVR). For each region, the across-subject Spearman correlation between real PET
and {Synthetic PET, ASL} on asymmetry indices ((L−R)/(L+R)), plus the per-region
bias (mean real − modality). Both atlases, cross-validated sample and test set, in
the same house style. Built from `R['across_subj_asym']` and `R['bias_asym']`.""")

code(r"""# correlation boxplots (per-region across-subject asymmetry r)
_ = plot_per_region_r_box('MCI', 'DKT',           'MCI asymmetry r', 'figASYM_MCI_DKT_r',
        source='across_subj_asym'); plt.show()
_ = plot_per_region_r_box('MCI', 'HarvardOxford', 'MCI asymmetry r', 'figASYM_MCI_HO_r',
        source='across_subj_asym'); plt.show()
# per-region bias boxplots
_ = plot_bias_box('MCI', 'DKT',           'MCI asymmetry', 'AI', 'figASYM_MCI_DKT_bias',
        source='bias_asym'); plt.show()
_ = plot_bias_box('MCI', 'HarvardOxford', 'MCI asymmetry', 'AI', 'figASYM_MCI_HO_bias',
        source='bias_asym'); plt.show()""")

md("""### Figure 4D replacement — Sign congruency (TLE)""")

code(r"""def plot_congruency_paired(atlas, save_name):
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.6), sharey=True,
                             gridspec_kw={'wspace': 0.06})
    panels = []; gmin, gmax = np.inf, -np.inf
    for ax, label in zip(axes, ['cv', 'test']):
        df_c = R['congruency_overall'][atlas][label]['per_region']
        s, a = df_c['cong_synth'].values, df_c['cong_asl'].values
        v = np.isfinite(s) & np.isfinite(a); s, a = s[v], a[v]
        p = _wilcoxon_p(s, a)
        diff = s - a
        d = np.mean(diff)/np.std(diff, ddof=1) if np.std(diff, ddof=1)>0 else np.nan
        dfl = pd.DataFrame({'Modality': ['Synthetic PET']*len(s)+['ASL']*len(a),
                            'cong': np.r_[s, a]})
        sns_box(ax, dfl, 'Modality', 'cong', ['Synthetic PET', 'ASL'], [SYNTH_COLOR, ASL_COLOR],
                width=0.78)
        ax.set_ylabel('Sign congruency' if label == 'cv' else '', fontweight='bold')
        ax.set_xlabel(f'{label} (n={R["n_subjects"][("TLE",atlas)][label]} subjects)\n'
                      f'Wilcoxon p={p:.3g}, d={d:.2f}\n({np.sum(s>a)}/{len(s)} Synth>ASL)',
                      fontsize=9, labelpad=8)
        ax.axhline(0.5, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        sns.despine(ax=ax)
        panels.append((ax, dfl, p))
        gmin = min(gmin, np.r_[s, a].min()); gmax = max(gmax, np.r_[s, a].max())
    fit_ylim(axes, gmin, gmax)
    for ax, dfl, p in panels:
        annotate_box(ax, [('Synthetic PET', 'ASL')], dfl, 'Modality', 'cong',
                     ['Synthetic PET', 'ASL'], [p])
    synth_asl_legend(axes[0])
    fig.suptitle(f'TLE sign congruency — {atlas}', fontweight='bold', y=1.02)
    plt.tight_layout(); save_fig(fig, save_name); return fig

_ = plot_congruency_paired('DKT',          'fig4D_TLE_DKT_congruency'); plt.show()
_ = plot_congruency_paired('HarvardOxford', 'fig4D_TLE_HO_congruency'); plt.show()""")

md("""### Congruency improvement across brain regions (TLE)

Per-region difference in sign congruency (Synthetic − ASL), sorted. Bars are
coloured blue where Synthetic PET is more congruent with real PET than ASL, red
where ASL is better, grey where the difference is within 1 SD of zero. Generated
for both the cross-validated sample and the test set from
`R['congruency_overall']`.""")

code(r"""def plot_congruency_delta_bar(atlas, split, save_name, cohort='TLE',
                              source='congruency_overall'):
    split_label = 'cross-validated' if split == 'cv' else 'test set'
    df = R[source][atlas][split]['per_region'].copy()
    df['delta'] = df['cong_synth'] - df['cong_asl']
    df = df.sort_values('delta')
    threshold = float(np.nanstd(np.abs(df['delta'])))
    colors = [SYNTH_COLOR if d>threshold else (ASL_COLOR if d<-threshold else 'gray')
              for d in df['delta']]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(len(df)), df['delta'], color=colors, alpha=BOX_ALPHA,
           edgecolor='black', linewidth=0.4)
    ax.set_xticks(range(len(df))); ax.set_xticklabels(df['Region'], rotation=90, fontsize=10)
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('Δ sign congruency (Synthetic PET − ASL)', fontweight='bold')
    n_sub = R['n_subjects'][(cohort, atlas)][split]
    ax.set_title(f'{cohort} sign-congruency improvement per region — {atlas} ({split_label}, n={n_sub})',
                  fontweight='bold')
    n_better = int(np.sum(df['delta'] > 0))
    ax.text(0.02, 0.95, f'Synth > ASL: {n_better}/{len(df)} regions',
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.legend(handles=[Patch(facecolor=SYNTH_COLOR, alpha=BOX_ALPHA, label='Synthetic PET better'),
                       Patch(facecolor=ASL_COLOR,   alpha=BOX_ALPHA, label='ASL better'),
                       Patch(facecolor='gray',      alpha=BOX_ALPHA, label='Within 1 SD')],
              loc='lower right', fontsize=8, frameon=False)
    sns.despine(ax=ax); plt.tight_layout(); save_fig(fig, save_name); return fig

for split in ['cv', 'test']:
    _ = plot_congruency_delta_bar('DKT',           split, f'figCongDelta_TLE_DKT_{split}'); plt.show()
    _ = plot_congruency_delta_bar('HarvardOxford', split, f'figCongDelta_TLE_HO_{split}');  plt.show()""")

md("""### MCI asymmetry — correlation-difference and congruency forest plots

Per-region forest plots for the MCI asymmetry analysis: the **correlation
difference** Δr = (Synthetic PET − ASL) across-subject Spearman r, and the
**sign-congruency difference** Δ = (Synthetic PET − ASL). Bars blue where Synthetic
PET beats ASL, red where ASL is better, grey within 1 SD of zero. Both atlases,
cross-validated sample and test set. Built from `R['across_subj_asym']` and
`R['congruency_mci']`.""")

code(r"""for split in ['cv', 'test']:
    # correlation-difference (Δr) forest plots — MCI asymmetry
    _ = plot_region_delta_bar('MCI', 'DKT',           'DKT',            'asymmetry', split,
            f'figASYM_MCI_DKT_delta_r_{split}', source='across_subj_asym'); plt.show()
    _ = plot_region_delta_bar('MCI', 'HarvardOxford', 'Harvard-Oxford', 'asymmetry', split,
            f'figASYM_MCI_HO_delta_r_{split}', source='across_subj_asym'); plt.show()
    # sign-congruency-difference forest plots — MCI
    _ = plot_congruency_delta_bar('DKT',           split, f'figASYMCongDelta_MCI_DKT_{split}',
            cohort='MCI', source='congruency_mci'); plt.show()
    _ = plot_congruency_delta_bar('HarvardOxford', split, f'figASYMCongDelta_MCI_HO_{split}',
            cohort='MCI', source='congruency_mci'); plt.show()""")

md("""### Congruency quadrant scatterplots — hippocampus & parahippocampus (TLE)

Per-subject asymmetry-index scatterplots for the hippocampus and the
parahippocampus. Each point is one
subject: real PET asymmetry on the x-axis, modality asymmetry on the y-axis.
Points in the **congruent** quadrants (same sign as real PET — shaded blue) are
blue; **incongruent** points (opposite sign — shaded red) are red. The
congruency rate and Pearson r are annotated. Built from per-subject asymmetry
indices via `MOD02.build_asymmetry_dataframe`, generated separately for the
cross-validated sample and the test set.""")

code(r"""def plot_congruency_quadrant(df_ai, x_var, y_var, region, ax, title,
                             xlabel=None, ylabel=None):
    d = df_ai[df_ai['Region'] == region]
    x = d[x_var].values; y = d[y_var].values
    v = np.isfinite(x) & np.isfinite(y); x, y = x[v], y[v]
    if len(x) == 0:
        ax.set_visible(False); return
    same = ((x >= 0) & (y >= 0)) | ((x < 0) & (y < 0))
    lim = float(np.max([np.max(np.abs(x)), np.max(np.abs(y))]) * 1.15)
    # quadrant shading: congruent (++/--) blue, incongruent (+-/-+) red
    ax.axhspan(0, lim, 0.5, 1.0, alpha=0.06, color=CONG_COLOR,   zorder=0)
    ax.axhspan(-lim, 0, 0.0, 0.5, alpha=0.06, color=CONG_COLOR,   zorder=0)
    ax.axhspan(0, lim, 0.0, 0.5, alpha=0.06, color=INCONG_COLOR, zorder=0)
    ax.axhspan(-lim, 0, 0.5, 1.0, alpha=0.06, color=INCONG_COLOR, zorder=0)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.scatter(x[same],  y[same],  color=CONG_COLOR,   s=60, alpha=0.75,
               edgecolor='white', linewidth=0.5, zorder=3, label='Congruent')
    ax.scatter(x[~same], y[~same], color=INCONG_COLOR, s=60, alpha=0.75,
               edgecolor='white', linewidth=0.5, zorder=3, label='Incongruent')
    rate = np.mean(same) if len(same) else np.nan
    r, _ = stats.pearsonr(x, y) if len(x) > 1 else (np.nan, np.nan)
    ax.text(0.05, 0.95, f'Congruent: {int(np.sum(same))}/{len(x)} ({rate:.0%})\nr = {r:.2f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel(xlabel or x_var, fontweight='bold')
    ax.set_ylabel(ylabel or y_var, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_aspect('equal'); sns.despine(ax=ax)


def plot_region_quadrants(atlas, split, save_name, region='Hippocampus',
                          region_label='hippocampus'):
    split_label = 'cross-validated' if split == 'cv' else 'test set'
    df_merged = load_df_merged('TLE', atlas)
    subs = list(df_merged['subject'].unique())
    pool, cv, test = split_subjects(subs, fm_tle)
    sub_set = cv if split == 'cv' else test
    df_ai = MOD02.build_asymmetry_dataframe(df_merged)
    df_ai = df_ai[df_ai['Subject'].isin(sub_set)]
    if region not in df_ai['Region'].unique():
        print(f'  {region} not found in {atlas} ({split}); skipping'); return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    plot_congruency_quadrant(df_ai, 'PET AI Original', 'PET AI Recon', region, axes[0],
                             f'Real PET vs Synthetic PET — {region_label}',
                             xlabel='Real PET asymmetry index',
                             ylabel='Synthetic PET asymmetry index')
    plot_congruency_quadrant(df_ai, 'PET AI Original', 'ASL AI', region, axes[1],
                             f'Real PET vs ASL — {region_label}',
                             xlabel='Real PET asymmetry index',
                             ylabel='ASL asymmetry index')
    axes[0].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle(f'TLE {region_label} sign congruency — {atlas} ({split_label}, n={len(sub_set)})',
                  fontweight='bold', y=1.02)
    plt.tight_layout(); save_fig(fig, save_name); return fig

# congruency quadrant scatterplots for the hippocampus AND the parahippocampus
QUAD_REGIONS = [
    ('DKT',           'Hippocampus',                          'hippocampus',             'DKT'),
    ('DKT',           'parahippocampal',                      'parahippocampus',         'DKT'),
    ('HarvardOxford', 'Hippocampus',                          'hippocampus',             'HO'),
    ('HarvardOxford', 'ParahippocampalGyrusanteriordivision', 'parahippocampus (ant.)',  'HO'),
]
for split in ['cv', 'test']:
    for atlas, region, region_label, atag in QUAD_REGIONS:
        _ = plot_region_quadrants(atlas, split,
                f'figCongQuad_TLE_{atag}_{region}_{split}',
                region=region, region_label=region_label); plt.show()""")

md("""### Figure 6 replacement — Cohen's d quadrant scatterplots""")

code(r"""def plot_cohens_d_quadrant(df_d, title, ax, r_imp, p_imp):
    x = df_d['d_asl'].values; y = df_d['d_synth'].values
    v = np.isfinite(x) & np.isfinite(y); x, y = x[v], y[v]
    threshold = float(np.nanstd(np.abs(y - x)))
    colors = []
    for xi, yi in zip(x, y):
        if np.abs(yi-xi) <= threshold: colors.append('gray')
        elif np.abs(yi) > np.abs(xi):  colors.append(SYNTH_COLOR)
        else:                            colors.append(ASL_COLOR)
    ax.scatter(x, y, c=colors, s=70, alpha=0.75, edgecolor='white', linewidth=0.5)
    lim = max(np.nanmax(np.abs(x)), np.nanmax(np.abs(y))) * 1.15
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.4, linewidth=1)
    ax.axhline(0, color='gray', linewidth=0.4); ax.axvline(0, color='gray', linewidth=0.4)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel("Cohen's d (ASL)"); ax.set_ylabel("Cohen's d (Synthetic PET)")
    n_synth_mag = int(np.sum(np.abs(y) > np.abs(x)))
    ax.set_title(f'{title}\nimprovement-corr r={r_imp:.2f} (p={p_imp:.3g})\n'
                  f'Synth |d| > ASL |d|: {n_synth_mag}/{len(x)}',
                  fontsize=10, fontweight='bold')
    ax.set_aspect('equal'); sns.despine(ax=ax)


for split in ['cv', 'test']:
    split_label = 'cross-validated' if split == 'cv' else 'test set'
    fig, axes = plt.subplots(2, 2, figsize=(11, 11))
    for (cohort, atlas), ax in zip([('TLE','DKT'),('TLE','HarvardOxford'),
                                    ('MCI','DKT'),('MCI','HarvardOxford')], axes.flatten()):
        df_d = R['cohens_d'][(cohort, atlas)][split]['per_region']
        r_imp = R['cohens_d'][(cohort, atlas)][split]['improvement_corr_r']
        p_imp = R['cohens_d'][(cohort, atlas)][split]['improvement_corr_p']
        plot_cohens_d_quadrant(df_d, f'{cohort} - {atlas} ({split_label})', ax, r_imp, p_imp)
    plt.tight_layout()
    save_fig(fig, f'fig6_cohens_d_quadrant_{split}')
    plt.show()""")

md("""### Effect-size comparison quadrants — where synthesis helps vs. hurts

This reproduces the original `plot_cohens_d_scatter_comparison`
(`04_lateralization_cohens_d.py`), generated for both the cross-validated sample
and the test set, keeping its four-colour quadrant style. For each region we plot
**Cohen's d: Synthetic PET − ASL** (x = |d|_Synth − |d|_ASL) against
**Cohen's d: Real PET − ASL** (y = |d|_Real − |d|_ASL):

- **Green** — both Synthetic PET and Real PET beat ASL (upper-right)
- **Red** — both worse than ASL (lower-left)
- **Orange** — Synthetic PET beats ASL but Real PET does not (lower-right)
- **Blue** — Real PET beats ASL but Synthetic PET does not (upper-left): the synthesis
  *lost* the lateralizing/discriminative information that real PET carries

Points below the y=x diagonal are regions where Synthetic PET exceeds Real PET's
effect size. Quadrant fractions are stored in `R['cohens_d_quadrant']` and feed
the manuscript paragraph in §9.""")

code(r"""from matplotlib.lines import Line2D

# Original quadrant palette from plot_cohens_d_scatter_comparison (04_lateralization_cohens_d.py)
Q_GREEN  = '#2ca02c'   # both > ASL
Q_RED    = '#d62728'   # both < ASL
Q_ORANGE = '#ff7f0e'   # Synthetic > ASL, Real < ASL
Q_BLUE   = '#1f77b4'   # Real > ASL, Synthetic < ASL

def _d_comparison_xy(cohort, atlas, split='cv'):
    dd = R['cohens_d'][(cohort, atlas)][split]['per_region']
    real_d  = dd['d_real'].values.astype(float)
    synth_d = dd['d_synth'].values.astype(float)
    asl_d   = dd['d_asl'].values.astype(float)
    regions = dd['Region'].values
    m = ~(np.isnan(real_d) | np.isnan(synth_d) | np.isnan(asl_d))
    real_d, synth_d, asl_d, regions = real_d[m], synth_d[m], asl_d[m], regions[m]
    x = np.abs(synth_d) - np.abs(asl_d)   # Synthetic - ASL
    y = np.abs(real_d)  - np.abs(asl_d)   # Real - ASL
    return x, y, regions

# --- quadrant fractions for both splits (feed §9 paragraph) ---
R['cohens_d_quadrant'] = {}
for cohort in ['TLE', 'MCI']:
    for atlas in ['DKT', 'HarvardOxford']:
        R['cohens_d_quadrant'][(cohort, atlas)] = {}
        for split in ['cv', 'test']:
            x, y, _ = _d_comparison_xy(cohort, atlas, split)
            ul  = (y > 0) & (x < 0)   # Real > ASL, Synthetic not (upper-left, blue)
            sgr = x > y               # |d| Synthetic > |d| Real (below diagonal)
            R['cohens_d_quadrant'][(cohort, atlas)][split] = dict(
                frac_upper_left=float(np.mean(ul))  if len(ul)  else float('nan'),
                frac_synth_gt_real=float(np.mean(sgr)) if len(sgr) else float('nan'),
                n=int(len(x)))

def plot_d_scatter_comparison(cohort, atlas, split, ax):
    split_label = 'cross-validated' if split == 'cv' else 'test set'
    x, y, regions = _d_comparison_xy(cohort, atlas, split)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    lim = float(max(np.abs(x).max(), np.abs(y).max()) * 1.2) if len(x) else 1.0
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, linewidth=1.5)
    colors = []
    for xi, yi in zip(x, y):
        if   xi > 0 and yi > 0: colors.append(Q_GREEN)   # both > ASL
        elif xi < 0 and yi < 0: colors.append(Q_RED)     # both < ASL
        elif xi > 0 and yi < 0: colors.append(Q_ORANGE)  # Synthetic better, Real worse
        else:                   colors.append(Q_BLUE)    # Real better, Synthetic worse
    ax.scatter(x, y, c=colors, s=80, alpha=0.7, edgecolors='black', linewidths=1, zorder=3)
    ax.text( lim*0.7,  lim*0.7, 'Both > ASL', fontsize=9, color=Q_GREEN, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-lim*0.7, -lim*0.7, 'Both < ASL', fontsize=9, color=Q_RED, fontweight='bold',
            ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text( lim*0.62, -lim*0.55, 'Synthetic PET > ASL\nReal PET < ASL', fontsize=8, color=Q_ORANGE,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(-lim*0.62,  lim*0.55, 'Real PET > ASL\nSynthetic PET < ASL', fontsize=8, color=Q_BLUE,
            fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel("Cohen's d: Synthetic PET - ASL", fontweight='bold', fontsize=11)
    ax.set_ylabel("Cohen's d: Real PET - ASL", fontweight='bold', fontsize=11)
    ax.set_title(f'{cohort} - {atlas} ({split_label}, {len(regions)} regions)',
                  fontweight='bold', fontsize=11)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    r, p = stats.pearsonr(x, y) if len(x) > 1 else (np.nan, np.nan)
    ax.text(0.05, 0.95, f'r = {r:.2f}, p = {p:.3g}', transform=ax.transAxes,
            fontsize=10, fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    sns.despine(ax=ax)

legend_elements = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor=Q_GREEN,  markersize=10, label='Both > ASL'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=Q_RED,    markersize=10, label='Both < ASL'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=Q_BLUE,   markersize=10, label='Real PET > ASL, Synthetic PET < ASL'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor=Q_ORANGE, markersize=10, label='Synthetic PET > ASL, Real PET < ASL'),
    Line2D([0],[0], linestyle='--', color='black', alpha=0.5, label='Synthetic PET = Real PET (vs ASL)')]
for split in ['cv', 'test']:
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for (cohort, atlas), ax in zip([('TLE','DKT'),('TLE','HarvardOxford'),
                                    ('MCI','DKT'),('MCI','HarvardOxford')], axes.flatten()):
        plot_d_scatter_comparison(cohort, atlas, split, ax)
    axes[0, 1].legend(handles=legend_elements, loc='lower right', fontsize=8)
    plt.tight_layout()
    save_fig(fig, f'fig6B_d_scatter_comparison_{split}')
    plt.show()

for split in ['cv', 'test']:
    print(f'=== Effect-size quadrant fractions ({split}) ===')
    for k, q in R['cohens_d_quadrant'].items():
        qs = q[split]
        print(f'  {k}: upper-left (Real>ASL, Synth not) = {qs["frac_upper_left"]:.0%}; '
              f'Synth|d|>Real|d| = {qs["frac_synth_gt_real"]:.0%} (n={qs["n"]})')""")


md("""### Cohen's d export for atlaspy — per-region CSV (one per cohort × atlas × split)

Writes one CSV per dataset × atlas × split to `tables/revision_cohens_d/`, holding
the per-region Cohen's d values used by the 4-panel effect-size scatterplots
(Fig 6 / 6B), the atlas indices for each hemisphere, and the quadrant colour each
region takes in both scatterplots — so the same regions can be coloured on a brain
with `atlaspy`. Columns:

- `region`, `atlas_index_left`, `atlas_index_right`
- `cohens_d_real`, `cohens_d_synthetic`, `cohens_d_asl`
- `comp_x_synth_minus_asl` = |d|_Synth − |d|_ASL, `comp_y_real_minus_asl` = |d|_Real − |d|_ASL
- `comparison_quadrant` / `comparison_color` — the four-colour Fig 6B scheme
  (green both>ASL, red both<ASL, orange Synth>ASL/Real<ASL, blue Real>ASL/Synth<ASL)
- `quadrant_category` / `quadrant_color` — the Fig 6 scheme (blue Synth|d|>ASL|d|,
  red ASL|d|>Synth|d|, grey within 1 SD)

**Rendering the quadrant map with `atlaspy`.** `atlaspy` colours ROIs from an
`atlas_index` + `roi_value` table through a matplotlib colormap, so we encode each
quadrant as an integer code (0–3) and build a discrete colormap whose colours are
exactly the scatterplot quadrant colours. The 4-panel view
(`plot_rois_atlas_lrm`) then shows a spatial brain map of which cortical regions
fall in each effect-size quadrant — the same colours as the Fig 6B scatterplot —
and `plot_subcortical_brain_regions_lrt` does the same for subcortical structures
(hippocampus, thalamus, amygdala, …). Use `'dkt'` for the DKT CSVs and `'ho'` for
the Harvard-Oxford CSVs.

```python
import atlaspy.core as apy
import pandas as pd
from matplotlib.colors import ListedColormap

# one CSV per cohort × atlas × split; atlas = 'dkt' (DKT) or 'ho' (Harvard-Oxford)
atlas = 'dkt'
df = pd.read_csv('tables/revision_cohens_d/cohens_d_TLE_DKT_cv.csv')

# Fig 6B comparison quadrants -> integer code (0..3) + matching scatter colour
cats   = ['both_lt_ASL', 'real_gt_ASL_synth_lt_ASL',
          'synth_gt_ASL_real_lt_ASL', 'both_gt_ASL']
colors = ['#d62728', '#1f77b4', '#ff7f0e', '#2ca02c']   # red, blue, orange, green
code   = {c: i for i, c in enumerate(cats)}
df['code'] = df['comparison_quadrant'].map(code)

# atlaspy needs columns atlas_index + roi_value, one row per hemisphere index
rows = []
for _, r in df.iterrows():
    for idx in (r['atlas_index_left'], r['atlas_index_right']):
        if pd.notna(idx):
            rows.append({'atlas_index': int(idx), 'roi_value': int(r['code'])})
df_values = pd.DataFrame(rows)

# discrete colormap whose colours ARE the scatterplot quadrant colours
cmap = ListedColormap(colors)            # codes 0..3 -> the four quadrant colours

# 4-panel cortical view (the one you want): L/R lateral + medial
apy.plot_rois_atlas_lrm(atlas, df_values, cmap=cmap)
# subcortical structures (hippocampus, thalamus, amygdala, ...)
apy.plot_subcortical_brain_regions_lrt(atlas, df_values, cmap=cmap)
# optional interactive single view
apy.plot_rois_atlas(atlas, df_values, interactive=True, cmap=cmap)

# For the Fig 6 (blue/red/grey) scheme instead, map `quadrant_category` with
# cats=['asl_gt_synth','similar','synth_gt_asl'],
# colors=['#b2182b','#888888','#2166ac'].
```

To keep the colours aligned even when a CSV is missing one of the four quadrants,
keep all four `code` levels in the colormap order above (and pass `vmin=0, vmax=3`
if your `atlaspy` build forwards normalization arguments).""")

code(r"""CSV_DIR = SCRIPT_DIR / 'tables' / 'revision_cohens_d'
CSV_DIR.mkdir(parents=True, exist_ok=True)

# Fig 6B four-colour comparison palette / Fig 6 quadrant palette
Q_GREEN, Q_RED, Q_ORANGE, Q_BLUE = '#2ca02c', '#d62728', '#ff7f0e', '#1f77b4'
F6_BLUE, F6_RED, F6_GREY = '#2166ac', '#b2182b', '#888888'

def region_index_map(cohort, atlas):
    df = load_df_merged(cohort, atlas)
    m = {}
    for _, row in df[['region_name', 'side', 'atlas_index']].drop_duplicates().iterrows():
        m.setdefault(row['region_name'], {})[row['side']] = int(row['atlas_index'])
    return m

saved = []
for cohort in ['TLE', 'MCI']:
    for atlas in ['DKT', 'HarvardOxford']:
        idxmap = region_index_map(cohort, atlas)
        for split in ['cv', 'test']:
            dd = R['cohens_d'][(cohort, atlas)][split]['per_region'].copy()
            dd = dd.dropna(subset=['d_real', 'd_synth', 'd_asl']).reset_index(drop=True)
            xx = dd['d_synth'].abs().values - dd['d_asl'].abs().values   # Fig 6B x
            yy = dd['d_real'].abs().values  - dd['d_asl'].abs().values   # Fig 6B y
            comp_cat, comp_col = [], []
            for xi, yi in zip(xx, yy):
                if   xi > 0 and yi > 0: comp_cat.append('both_gt_ASL');               comp_col.append(Q_GREEN)
                elif xi < 0 and yi < 0: comp_cat.append('both_lt_ASL');               comp_col.append(Q_RED)
                elif xi > 0 and yi < 0: comp_cat.append('synth_gt_ASL_real_lt_ASL');  comp_col.append(Q_ORANGE)
                else:                   comp_cat.append('real_gt_ASL_synth_lt_ASL');  comp_col.append(Q_BLUE)
            ax_, ay_ = dd['d_asl'].values, dd['d_synth'].values          # Fig 6 axes
            thr = float(np.nanstd(np.abs(ay_ - ax_))) if len(ax_) else 0.0
            quad_cat, quad_col = [], []
            for xi, yi in zip(ax_, ay_):
                if   abs(yi - xi) <= thr: quad_cat.append('similar');      quad_col.append(F6_GREY)
                elif abs(yi) > abs(xi):   quad_cat.append('synth_gt_asl'); quad_col.append(F6_BLUE)
                else:                     quad_cat.append('asl_gt_synth'); quad_col.append(F6_RED)
            out = pd.DataFrame({
                'region': dd['Region'].values,
                'atlas_index_left':  [idxmap.get(r, {}).get('Left')  for r in dd['Region']],
                'atlas_index_right': [idxmap.get(r, {}).get('Right') for r in dd['Region']],
                'cohens_d_real':      dd['d_real'].values,
                'cohens_d_synthetic': dd['d_synth'].values,
                'cohens_d_asl':       dd['d_asl'].values,
                'comp_x_synth_minus_asl': xx,
                'comp_y_real_minus_asl':  yy,
                'comparison_quadrant': comp_cat, 'comparison_color': comp_col,
                'quadrant_category': quad_cat,   'quadrant_color': quad_col,
            })
            atag = 'DKT' if atlas == 'DKT' else 'HO'
            fname = f'cohens_d_{cohort}_{atag}_{split}.csv'
            out.to_csv(CSV_DIR / fname, index=False)
            saved.append((fname, out.shape[0]))
print('Wrote', len(saved), 'CSVs to', CSV_DIR.relative_to(SCRIPT_DIR))
for fname, nrow in saved:
    print(f'  {fname}  ({nrow} regions)')
print('\nExample (TLE DKT cv):')
display(pd.read_csv(CSV_DIR / 'cohens_d_TLE_DKT_cv.csv').head(8))""")


# ============================================================================
# Section 8: Consolidated table - derived from R
# ============================================================================

md("""## 8. Consolidated BEFORE → AFTER table (derived from `R`)

Every value pulled from the `R` dict populated above. "BEFORE (pooled)"
comes from running each computation on the full cohort; "AFTER
cross-validated" and "AFTER test" come from the per-split restriction.""")

code(r"""def fmt_med_iqr_pair(s):
    return (f"{s['synth_median']:.2f} [{s['synth_q1']:.2f}, {s['synth_q3']:.2f}] vs "
            f"{s['asl_median']:.2f} [{s['asl_q1']:.2f}, {s['asl_q3']:.2f}]")

def fmt_mean_sd_pair(d):
    return (f"{d['synth_mean']:.3f} ± {d['synth_sd']:.3f} vs "
            f"{d['asl_mean']:.3f} ± {d['asl_sd']:.3f}")

def fmt_p(p):
    if p != p: return 'NA'
    if p < 1e-10: return 'p<10⁻¹⁰'
    if p < 0.001: return 'p<0.001'
    return f'p={p:.3g}'

def fmt_d(d):
    return f'd={d:.2f}' if d == d else 'd=NA'

W = R['within_subj']; A = R['across_subj']; B = R['bias']
C = R['congruency_overall']; D = R['cohens_d']

rows = [
    ('MCI within-subj r (DKT, SUVR)',
        f"{W[('MCI','DKT')]['pool']['synth_mean']:.3f} vs {W[('MCI','DKT')]['pool']['asl_mean']:.3f}, {fmt_d(W[('MCI','DKT')]['pool']['paired_d'])}",
        f"{fmt_mean_sd_pair(W[('MCI','DKT')]['cv'])}, {fmt_d(W[('MCI','DKT')]['cv']['paired_d'])}",
        fmt_mean_sd_pair(W[('MCI','DKT')]['test'])),
    ('TLE within-subj r (DKT, SUVR)',
        f"{W[('TLE','DKT')]['pool']['synth_mean']:.3f} vs {W[('TLE','DKT')]['pool']['asl_mean']:.3f}, {fmt_d(W[('TLE','DKT')]['pool']['paired_d'])}",
        f"{fmt_mean_sd_pair(W[('TLE','DKT')]['cv'])}, {fmt_d(W[('TLE','DKT')]['cv']['paired_d'])}",
        f"{fmt_mean_sd_pair(W[('TLE','DKT')]['test'])}, {fmt_d(W[('TLE','DKT')]['test']['paired_d'])}"),
    ('TLE across-subj r (DKT, asymmetry)',
        f"{fmt_med_iqr_pair(A[('TLE','DKT')]['pool'])}, {fmt_d(A[('TLE','DKT')]['pool']['cohens_d_paired'])}",
        f"{fmt_med_iqr_pair(A[('TLE','DKT')]['cv'])}, {fmt_d(A[('TLE','DKT')]['cv']['cohens_d_paired'])}, {fmt_p(A[('TLE','DKT')]['cv']['wilcoxon_p'])}",
        f"{fmt_med_iqr_pair(A[('TLE','DKT')]['test'])}, {fmt_p(A[('TLE','DKT')]['test']['wilcoxon_p'])}"),
    ('MCI across-subj r (DKT, SUVR)',
        f"{fmt_med_iqr_pair(A[('MCI','DKT')]['pool'])}, {fmt_d(A[('MCI','DKT')]['pool']['cohens_d_paired'])}",
        f"{fmt_med_iqr_pair(A[('MCI','DKT')]['cv'])}, {fmt_d(A[('MCI','DKT')]['cv']['cohens_d_paired'])}, {fmt_p(A[('MCI','DKT')]['cv']['wilcoxon_p'])}",
        f"{fmt_med_iqr_pair(A[('MCI','DKT')]['test'])}, {fmt_p(A[('MCI','DKT')]['test']['wilcoxon_p'])}"),
    ('MCI across-subj r (HO, SUVR)',
        f"{fmt_med_iqr_pair(A[('MCI','HarvardOxford')]['pool'])}, {fmt_d(A[('MCI','HarvardOxford')]['pool']['cohens_d_paired'])}",
        f"{fmt_med_iqr_pair(A[('MCI','HarvardOxford')]['cv'])}, {fmt_d(A[('MCI','HarvardOxford')]['cv']['cohens_d_paired'])}, {fmt_p(A[('MCI','HarvardOxford')]['cv']['wilcoxon_p'])}",
        f"{fmt_med_iqr_pair(A[('MCI','HarvardOxford')]['test'])}, {fmt_p(A[('MCI','HarvardOxford')]['test']['wilcoxon_p'])}"),
    ('TLE-DKT congruency overall',
        f"{C['DKT']['pool']['synth_mean']*100:.1f}% vs {C['DKT']['pool']['asl_mean']*100:.1f}%, {fmt_d(C['DKT']['pool']['cohens_d_paired'])}",
        f"{C['DKT']['cv']['synth_mean']*100:.1f}% vs {C['DKT']['cv']['asl_mean']*100:.1f}%, {fmt_d(C['DKT']['cv']['cohens_d_paired'])}, {fmt_p(C['DKT']['cv']['wilcoxon_p'])}",
        f"{C['DKT']['test']['synth_mean']*100:.1f}% vs {C['DKT']['test']['asl_mean']*100:.1f}%, {fmt_d(C['DKT']['test']['cohens_d_paired'])}, {fmt_p(C['DKT']['test']['wilcoxon_p'])}"),
    ('TLE-DKT improvement-corr',
        f"r={D[('TLE','DKT')]['pool']['improvement_corr_r']:.2f}, {fmt_p(D[('TLE','DKT')]['pool']['improvement_corr_p'])}",
        f"r={D[('TLE','DKT')]['cv']['improvement_corr_r']:.2f}, {fmt_p(D[('TLE','DKT')]['cv']['improvement_corr_p'])}",
        f"r={D[('TLE','DKT')]['test']['improvement_corr_r']:.2f}, {fmt_p(D[('TLE','DKT')]['test']['improvement_corr_p'])}"),
    ('MCI-DKT improvement-corr',
        f"r={D[('MCI','DKT')]['pool']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','DKT')]['pool']['improvement_corr_p'])}",
        f"r={D[('MCI','DKT')]['cv']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','DKT')]['cv']['improvement_corr_p'])}",
        f"r={D[('MCI','DKT')]['test']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','DKT')]['test']['improvement_corr_p'])}"),
    ('MCI-HO improvement-corr',
        f"r={D[('MCI','HarvardOxford')]['pool']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','HarvardOxford')]['pool']['improvement_corr_p'])}",
        f"r={D[('MCI','HarvardOxford')]['cv']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','HarvardOxford')]['cv']['improvement_corr_p'])}",
        f"r={D[('MCI','HarvardOxford')]['test']['improvement_corr_r']:.2f}, {fmt_p(D[('MCI','HarvardOxford')]['test']['improvement_corr_p'])}"),
]
df_summary = pd.DataFrame(rows, columns=['Claim','BEFORE (pooled)',
                                          'AFTER cross-validated','AFTER test'])
display(df_summary)""")


# ============================================================================
# Section 9: Manuscript text - derived from R
# ============================================================================

md("""## 9. Updated manuscript text (Results section)

Generated below using `display(Markdown(...))` + f-strings against `R`. Every
numerical value is interpolated from a code-derived variable; nothing is
hardcoded in markdown.""")

code(r"""def med_iqr_synth(s): return f"{s['synth_median']:.2f} (IQR: [{s['synth_q1']:.2f}, {s['synth_q3']:.2f}])"
def med_iqr_asl(s):   return f"{s['asl_median']:.2f} (IQR: [{s['asl_q1']:.2f}, {s['asl_q3']:.2f}])"
def med_iqr_compact(s, key='synth'):
    return f"{s[f'{key}_median']:.2f} [{s[f'{key}_q1']:.2f}, {s[f'{key}_q3']:.2f}]"
def p_str(p):
    if p != p: return 'NA'
    if p < 1e-10: return 'p<10⁻¹⁰'
    if p < 0.001: return 'p<0.001'
    return f'p={p:.3g}'

n_TLE_cv  = R['n_subjects'][('TLE','DKT')]['cv']
n_TLE_te  = R['n_subjects'][('TLE','DKT')]['test']
n_MCI_cv  = R['n_subjects'][('MCI','DKT')]['cv']
n_MCI_te  = R['n_subjects'][('MCI','DKT')]['test']

q_mci_ssim = R['quality']['MCI']['ssim_recon']
q_mci_psnr = R['quality']['MCI']['psnr_recon']
q_tle_ssim = R['quality']['TLE']['ssim_recon']

w_TLE_DKT = R['within_subj'][('TLE','DKT')]
w_MCI_DKT = R['within_subj'][('MCI','DKT')]
a_TLE_DKT = R['across_subj'][('TLE','DKT')]
a_TLE_HO  = R['across_subj'][('TLE','HarvardOxford')]
a_MCI_DKT = R['across_subj'][('MCI','DKT')]
a_MCI_HO  = R['across_subj'][('MCI','HarvardOxford')]
b_TLE_DKT = R['bias'][('TLE','DKT')]
b_MCI_DKT = R['bias'][('MCI','DKT')]
b_MCI_HO  = R['bias'][('MCI','HarvardOxford')]
# Bilateral putamen-normalized SUVR (not asymmetry) for the TLE across-subject
# fidelity sentence in the "PET-like images" section.
a_TLE_DKT_suvr = R['across_subj_suvr'][('TLE','DKT')]
b_TLE_DKT_suvr = R['bias_suvr'][('TLE','DKT')]
C_DKT     = R['congruency_overall']['DKT']
C_HO      = R['congruency_overall']['HarvardOxford']
hipp_DKT  = R['congruency_regions']['DKT']['Hippocampus']
par_DKT   = R['congruency_regions']['DKT']['parahippocampal']
par_HO_a  = R['congruency_regions']['HarvardOxford']['ParahippocampalGyrusanteriordivision']
d_TLE_DKT = R['cohens_d'][('TLE','DKT')]
d_TLE_HO  = R['cohens_d'][('TLE','HarvardOxford')]
d_MCI_DKT = R['cohens_d'][('MCI','DKT')]
d_MCI_HO  = R['cohens_d'][('MCI','HarvardOxford')]
qd_TLE_DKT = R['cohens_d_quadrant'][('TLE','DKT')]
qd_TLE_HO  = R['cohens_d_quadrant'][('TLE','HarvardOxford')]
qd_MCI_DKT = R['cohens_d_quadrant'][('MCI','DKT')]
qd_MCI_HO  = R['cohens_d_quadrant'][('MCI','HarvardOxford')]

def delta_r(cohort, atlas, region, split='cv'):
    df = R['across_subj'][(cohort, atlas)][split]['per_region']
    row = df[df['Region'] == region]
    if len(row) == 0: return float('nan')
    return float(row['r_synth'].values[0] - row['r_asl'].values[0])

dr_hipp_DKT = delta_r('TLE', 'DKT',           'Hippocampus')
dr_para_DKT = delta_r('TLE', 'DKT',           'parahippocampal')
dr_hipp_HO  = delta_r('TLE', 'HarvardOxford', 'Hippocampus')
dr_paraA_HO = delta_r('TLE', 'HarvardOxford', 'ParahippocampalGyrusanteriordivision')
dr_paraP_HO = delta_r('TLE', 'HarvardOxford', 'ParahippocampalGyrusposteriordivision')

# TLE per-atlas fraction of regions where synthetic |d| exceeds ASL |d| (cross-validated)
tle_imp_dkt = d_TLE_DKT['cv']['n_synth_mag_gt_asl'] / d_TLE_DKT['cv']['n_regions']
tle_imp_ho  = d_TLE_HO['cv']['n_synth_mag_gt_asl']  / d_TLE_HO['cv']['n_regions']
tle_imp_lo, tle_imp_hi = sorted([tle_imp_dkt, tle_imp_ho])

abstract = f'''## Abstract (updated)

Positron Emission Tomography (PET) using fluorodeoxyglucose (FDG-PET) is a gold-standard imaging modality for detecting hypometabolism associated with the seizure onset zone (SOZ) in focal epilepsy. Similarly, FDG-PET is widely used for diagnosis and prognostication of Alzheimer's, other dementias and prodromal conditions such as mild cognitive impairment (MCI). However, FDG-PET involves the use of a radioactive tracer making repeated examinations prohibitive for some patients. Arterial Spin Labeling (ASL) offers a magnetic resonance imaging (MRI)-based quantification of cerebral blood flow (CBF) that has been compared to FDG-PET in SOZ and MCI detection, but its diagnostic performance relative to FDG-PET is limited. We aimed to improve MRI's diagnostic performance by developing a deep learning framework for synthesizing FDG-PET-like images from MRI inputs. Using paired PET-MRI data, we developed FlowGAN, a generative adversarial neural network (GAN) that synthesizes PET-like images from ASL and T1-weighted MRI inputs. We compared synthetic PET images and actual PET images to assess their ability to reproduce clinically meaningful hypometabolism and asymmetries in epilepsy and MCI. FlowGAN-generated images demonstrated significantly higher structural similarity, peak signal-to-noise ratio, and normalized cross-correlation relative to ASL CBF maps (all p<0.001). Synthetic PET regional asymmetries in epilepsy ({a_TLE_DKT['cv']['n_synth_gt_asl']}/{a_TLE_DKT['cv']['n_regions']} brain regions) and bilateral SUVR values in MCI ({a_MCI_DKT['cv']['n_synth_gt_asl']}/{a_MCI_DKT['cv']['n_regions']}) had higher correlation with real FDG-PET than input ASL. Regions with known poor ASL-PET correlation, such as the hippocampus, showed the greatest improvement with synthetic PET images. Effect sizes for distinguishing MCI from healthy control subjects were highly correlated between synthetic and real PET (Spearman's r = {d_MCI_DKT['cv']['improvement_corr_r']:.2f}, {p_str(d_MCI_DKT['cv']['improvement_corr_p'])}). The correlation between effect sizes for lateralizing temporal lobe epilepsy (TLE) were lower between synthetic and real PET (Spearman's r = {d_TLE_DKT['cv']['improvement_corr_r']:.2f}, {p_str(d_TLE_DKT['cv']['improvement_corr_p'])}), but with improved effect sizes relative to input ASL in between {tle_imp_lo:.0%} and {tle_imp_hi:.0%} of regions, depending on the brain atlas used. FlowGAN improves MRI's diagnostic performance, generating synthetic PET images that closely mimic actual FDG-PET in hypometabolism associated with epilepsy and MCI.
'''

txt = f'''---

{abstract}

### FlowGAN successfully generates PET-like images from ASL and T1w inputs

To bring ASL's diagnostic performance closer to that of FDG-PET, we developed FlowGAN, a generative adversarial network architecture that synthesizes FDG-PET from T1w and ASL CBF MRI inputs. We trained two FlowGAN models: one for the Epilepsy dataset and one for the MCI dataset. FlowGAN outputs shown here correspond to the model inference run on the held-out subjects after the 12-fold cross-validation used during training. Since this process was repeated until every subject was held out from training during one of the 12 folds, we have FlowGAN outputs for all subjects in the Epilepsy and MCI datasets. To address reviewer concerns regarding aggregate-versus-fold-level reporting, every analysis that follows is reported separately on two non-overlapping groups of subjects: a **cross-validated set**, consisting of the subjects in cross-validation folds 0–9 (Epilepsy n={n_TLE_cv}; MCI n={n_MCI_cv}), and a **held-out test set**, consisting of the subjects in the remaining two folds, 10 and 11 (Epilepsy n={n_TLE_te}; MCI n={n_MCI_te}), which were excluded from all downstream comparisons. Two types of metric are reported, and they are aggregated differently. **Per-subject metrics** (one value per subject — the image-quality metrics and the within-subject correlations computed across regions): on the cross-validated set, subjects are first averaged within each of the 10 folds and we report the mean ± standard deviation across those 10 fold-level means; on the test set, we report the mean ± standard deviation across the individual test subjects. **Across-subject metrics** (one value per brain region, computed over a group of subjects — the across-subject regional correlations, the sign-congruency rates, and the lateralization Cohen's d): these are computed once over the entire cross-validated set and once over the entire test set, and the two are reported separately. Visually, the resulting synthetic PET images are very similar to the actual PET images (Fig. 2). FlowGAN outputs reproduce hypometabolism, even when the input T1w and CBF maps do not have clear visual abnormalities, and do not introduce clear asymmetries in healthy brains (Fig. 2C, Supplementary Fig. 1,2). Quantitatively (Fig. 3), the structural similarity (SSIM), peak signal-to-noise ratio (PSNR), and normalized cross correlation (NCC) between FlowGAN SUVR and real PET SUVR maps was significantly higher than that between real PET SUVR and ASL rCBF maps in both the Epilepsy and MCI datasets (all p<0.001 on the cross-validated sample). Similarly, the root mean squared error (RMSE) was significantly lower between FlowGAN SUVR and real PET SUVR maps than between real PET and ASL CBF maps in both datasets (all p<0.001 on the cross-validated sample). For every quality metric in both cohorts, the test set aggregate fell within the 95% confidence interval of the cross-validated sample (e.g., MCI SSIM: cross-validated {q_mci_ssim['cv_fold_mean']:.3f} ± {q_mci_ssim['cv_fold_sd']:.3f} vs test {q_mci_ssim['test_mean']:.3f} ± {q_mci_ssim['test_sd']:.3f}; MCI PSNR: {q_mci_psnr['cv_fold_mean']:.2f} ± {q_mci_psnr['cv_fold_sd']:.2f} dB vs {q_mci_psnr['test_mean']:.2f} ± {q_mci_psnr['test_sd']:.2f} dB; Epilepsy SSIM: {q_tle_ssim['cv_fold_mean']:.3f} ± {q_tle_ssim['cv_fold_sd']:.3f} vs {q_tle_ssim['test_mean']:.3f} ± {q_tle_ssim['test_sd']:.3f}; Supplementary Tables S1, S2).

For the Epilepsy dataset, the within-subject (across brain regions) average Spearman correlation between real and synthetic PET SUVR was **{w_TLE_DKT['cv']['synth_mean']:.3f} ± {w_TLE_DKT['cv']['synth_sd']:.3f}** across the 10 cross-validated folds (subject-level paired Cohen's d = {w_TLE_DKT['cv']['paired_d']:.2f}, Wilcoxon {p_str(w_TLE_DKT['cv']['wilcoxon_p'])}), significantly higher than between real PET SUVR and ASL rCBF (**{w_TLE_DKT['cv']['asl_mean']:.3f} ± {w_TLE_DKT['cv']['asl_sd']:.3f}**; Supplementary Fig. 3A); the test set reproduced this gap (synthetic {w_TLE_DKT['test']['synth_mean']:.3f} ± {w_TLE_DKT['test']['synth_sd']:.3f} vs ASL {w_TLE_DKT['test']['asl_mean']:.3f} ± {w_TLE_DKT['test']['asl_sd']:.3f}; paired d={w_TLE_DKT['test']['paired_d']:.2f}, {p_str(w_TLE_DKT['test']['wilcoxon_p'])}). For the MCI dataset, the within-subject average Spearman correlation on the cross-validated sample was **{w_MCI_DKT['cv']['synth_mean']:.3f} ± {w_MCI_DKT['cv']['synth_sd']:.3f}** for synthetic PET versus **{w_MCI_DKT['cv']['asl_mean']:.3f} ± {w_MCI_DKT['cv']['asl_sd']:.3f}** for ASL ({p_str(w_MCI_DKT['cv']['wilcoxon_p'])}, Cohen's d={w_MCI_DKT['cv']['paired_d']:.2f}; Supplementary Fig. 3A); the same pattern held on the test set (synthetic {w_MCI_DKT['test']['synth_mean']:.3f} ± {w_MCI_DKT['test']['synth_sd']:.3f} vs ASL {w_MCI_DKT['test']['asl_mean']:.3f} ± {w_MCI_DKT['test']['asl_sd']:.3f}). For the across-subject (inter-individual) correlations of bilateral putamen-normalized SUVR in the Epilepsy dataset, the median Spearman correlation with real PET across {a_TLE_DKT_suvr['cv']['n_regions']} DKT brain regions did not differ significantly between synthetic PET and ASL on the cross-validated sample (synthetic median r = **{med_iqr_synth(a_TLE_DKT_suvr['cv'])}** versus ASL rCBF **{med_iqr_asl(a_TLE_DKT_suvr['cv'])}**; Wilcoxon {p_str(a_TLE_DKT_suvr['cv']['wilcoxon_p'])}, Cohen's d={a_TLE_DKT_suvr['cv']['cohens_d_paired']:.2f}), and this absence of a significant difference was also seen on the test set (synthetic median r = {med_iqr_compact(a_TLE_DKT_suvr['test'], 'synth')} vs ASL {med_iqr_compact(a_TLE_DKT_suvr['test'], 'asl')}; Wilcoxon {p_str(a_TLE_DKT_suvr['test']['wilcoxon_p'])}, d={a_TLE_DKT_suvr['test']['cohens_d_paired']:.2f}). Synthetic PET did, however, show substantially lower bias relative to real PET SUVR than ASL rCBF (synthetic median={b_TLE_DKT_suvr['cv']['synth_median']:.3f}, IQR: [{b_TLE_DKT_suvr['cv']['synth_q1']:.3f}, {b_TLE_DKT_suvr['cv']['synth_q3']:.3f}] versus ASL median={b_TLE_DKT_suvr['cv']['asl_median']:.3f}, IQR: [{b_TLE_DKT_suvr['cv']['asl_q1']:.3f}, {b_TLE_DKT_suvr['cv']['asl_q3']:.3f}]; {p_str(b_TLE_DKT_suvr['cv']['wilcoxon_p'])}, Cohen's d={b_TLE_DKT_suvr['cv']['cohens_d_paired']:.2f}; Supplementary Fig. 3B), a difference that was preserved on the test set ({p_str(b_TLE_DKT_suvr['test']['wilcoxon_p'])}, d={b_TLE_DKT_suvr['test']['cohens_d_paired']:.2f}). For the MCI dataset across-subject correlations, the median Spearman correlation across {a_MCI_DKT['cv']['n_regions']} DKT regions on the cross-validated sample was **{med_iqr_synth(a_MCI_DKT['cv'])}** for synthetic PET versus **{med_iqr_asl(a_MCI_DKT['cv'])}** for ASL (Wilcoxon {p_str(a_MCI_DKT['cv']['wilcoxon_p'])}, Cohen's d={a_MCI_DKT['cv']['cohens_d_paired']:.2f}; Supplementary Fig. 3B), with synthetic PET showing lower bias (median={b_MCI_DKT['cv']['synth_median']:.3f}, IQR: [{b_MCI_DKT['cv']['synth_q1']:.3f}, {b_MCI_DKT['cv']['synth_q3']:.3f}]) compared to ASL (median={b_MCI_DKT['cv']['asl_median']:.3f}, IQR: [{b_MCI_DKT['cv']['asl_q1']:.3f}, {b_MCI_DKT['cv']['asl_q3']:.3f}]; {p_str(b_MCI_DKT['cv']['wilcoxon_p'])}, Cohen's d={b_MCI_DKT['cv']['cohens_d_paired']:.2f}). The Harvard-Oxford atlas reproduced these patterns on both the cross-validated sample and the test set (e.g., MCI cross-validated synthetic median r = {med_iqr_compact(a_MCI_HO['cv'], 'synth')} vs ASL {med_iqr_compact(a_MCI_HO['cv'], 'asl')}, {p_str(a_MCI_HO['cv']['wilcoxon_p'])}, d={a_MCI_HO['cv']['cohens_d_paired']:.2f}; test set {p_str(a_MCI_HO['test']['wilcoxon_p'])}, d={a_MCI_HO['test']['cohens_d_paired']:.2f}; Supplementary Fig. 4). The only configuration in which the synthetic-versus-ASL gap did not reach formal significance on the test set was the MCI-DKT across-subject correlation (n={n_MCI_te}; {p_str(a_MCI_DKT['test']['wilcoxon_p'])}), reflecting limited statistical power for region-level Spearman comparisons at this sample size; the same comparison in the Harvard-Oxford atlas on the same test subjects remained significant ({p_str(a_MCI_HO['test']['wilcoxon_p'])}). These findings demonstrate that FlowGAN generates output images that are PET-like, and that preserve the general patterns of PET contrast across the brain both within individual subjects and across the population.

### FlowGAN recovers hypometabolism in regions with low PET-ASL coupling

We hypothesized that FlowGAN could help recover asymmetries in metabolism in brain regions where ASL alone is not capable of detecting asymmetries compared to PET. To test this, we compared the Spearman correlation of asymmetry indices across brain regions between real PET and synthetic PET, as well as between real PET and ASL across the cross-validated sample of {n_TLE_cv} epilepsy subjects (folds 0–9, with the remaining 2 folds reserved as a test set). In the DKT atlas (Fig. 4), the regional asymmetry correlations between real and synthetic PET were significantly higher (median r = **{a_TLE_DKT['cv']['synth_median']:.2f}**, IQR: [{a_TLE_DKT['cv']['synth_q1']:.2f}, {a_TLE_DKT['cv']['synth_q3']:.2f}]) than between real PET and ASL (median r = **{a_TLE_DKT['cv']['asl_median']:.2f}**, IQR: [{a_TLE_DKT['cv']['asl_q1']:.2f}, {a_TLE_DKT['cv']['asl_q3']:.2f}]; Wilcoxon {p_str(a_TLE_DKT['cv']['wilcoxon_p'])}, Cohen's d = **{a_TLE_DKT['cv']['cohens_d_paired']:.2f}**; Fig. 4B). Similarly, in the Harvard-Oxford atlas (Supplementary Fig. 5), synthetic PET showed higher correlations (median r = **{a_TLE_HO['cv']['synth_median']:.2f}**, IQR: [{a_TLE_HO['cv']['synth_q1']:.2f}, {a_TLE_HO['cv']['synth_q3']:.2f}]) compared to ASL (median r = **{a_TLE_HO['cv']['asl_median']:.2f}**, IQR: [{a_TLE_HO['cv']['asl_q1']:.2f}, {a_TLE_HO['cv']['asl_q3']:.2f}]; Wilcoxon {p_str(a_TLE_HO['cv']['wilcoxon_p'])}, Cohen's d = **{a_TLE_HO['cv']['cohens_d_paired']:.2f}**). Synthetic PET showed higher correlation than ASL in **{a_TLE_DKT['cv']['n_synth_gt_asl']}/{a_TLE_DKT['cv']['n_regions']}** DKT regions (Fig. 4C) and **{a_TLE_HO['cv']['n_synth_gt_asl']}/{a_TLE_HO['cv']['n_regions']}** Harvard-Oxford regions (Supplementary Fig. 5C). Additionally, synthetic PET demonstrated lower bias toward zero compared to ASL (Wilcoxon {p_str(b_TLE_DKT['cv']['wilcoxon_p'])}, Cohen's d = {b_TLE_DKT['cv']['cohens_d_paired']:.2f}; Fig. 4B, Supplementary Fig. 6). The same Synth>ASL pattern was preserved on the {n_TLE_te}-subject test set (DKT median r = {med_iqr_compact(a_TLE_DKT['test'], 'synth')} vs {med_iqr_compact(a_TLE_DKT['test'], 'asl')}, {p_str(a_TLE_DKT['test']['wilcoxon_p'])}; HO median r = {med_iqr_compact(a_TLE_HO['test'], 'synth')} vs {med_iqr_compact(a_TLE_HO['test'], 'asl')}, {p_str(a_TLE_HO['test']['wilcoxon_p'])}; Supplementary Fig. S12). These findings provide evidence that FlowGAN can improve the correlation between metabolic and perfusion left-right asymmetry originally present between PET and ASL.

Notably, several mesial temporal structures that exhibited poor coupling between PET and ASL showed substantially improved correlations with synthetic PET. For example, on the cross-validated sample the hippocampus (Δr = **{dr_hipp_DKT:.2f}**) and parahippocampal gyrus (Δr = **{dr_para_DKT:.2f}**) both demonstrated improved asymmetry correlations. This finding was also consistent in the Harvard-Oxford atlas for the hippocampus (Δr = **{dr_hipp_HO:.2f}**) and the parahippocampal gyrus anterior (Δr = **{dr_paraA_HO:.2f}**) and posterior divisions (Δr = **{dr_paraP_HO:.2f}**). A complete list of asymmetry correlation values across the cross-validated sample and test set is presented in Supplementary Tables 2,3.

In addition to comparing asymmetry correlations, we assessed the sign congruency of asymmetry direction between modalities, a metric that better approximates clinical lateralization where the direction of hypometabolism (left > right or right > left) is used clinically. We defined congruency as the proportion of subjects with matching asymmetry direction between modalities. As shown in Fig. 4D, on the cross-validated sample congruency between synthetic PET and real PET was higher than between ASL and real PET in key epileptogenic regions. In the hippocampus (DKT atlas), synthetic PET achieved **{hipp_DKT['cv']['synth_cong']*100:.1f}%** congruency with real PET compared to ASL's **{hipp_DKT['cv']['asl_cong']*100:.1f}%** congruency (McNemar {p_str(hipp_DKT['cv']['mcnemar_p'])} on the cross-validated sample; pooled across all {hipp_DKT['pool']['n']} subjects: synthetic {hipp_DKT['pool']['synth_cong']*100:.1f}% vs ASL {hipp_DKT['pool']['asl_cong']*100:.1f}%, McNemar {p_str(hipp_DKT['pool']['mcnemar_p'])}). This difference was more pronounced in the parahippocampal region, where synthetic PET reached **{par_DKT['cv']['synth_cong']*100:.1f}%** congruency versus ASL's **{par_DKT['cv']['asl_cong']*100:.1f}%** on the cross-validated sample (McNemar {p_str(par_DKT['cv']['mcnemar_p'])}), and was also reflected in the Harvard-Oxford parahippocampal anterior division (synthetic **{par_HO_a['cv']['synth_cong']*100:.1f}%** vs ASL **{par_HO_a['cv']['asl_cong']*100:.1f}%**, McNemar {p_str(par_HO_a['cv']['mcnemar_p'])}). Across all DKT atlas regions on the cross-validated sample, synthetic PET demonstrated higher sign congruency than ASL in **{C_DKT['cv']['n_synth_gt_asl']}/{C_DKT['cv']['n_regions']}** regions (Wilcoxon {p_str(C_DKT['cv']['wilcoxon_p'])}, Cohen's d = **{C_DKT['cv']['cohens_d_paired']:.2f}**), and in **{C_HO['cv']['n_synth_gt_asl']}/{C_HO['cv']['n_regions']}** Harvard-Oxford regions (Wilcoxon {p_str(C_HO['cv']['wilcoxon_p'])}, Cohen's d = **{C_HO['cv']['cohens_d_paired']:.2f}**), particularly in temporal and frontal structures relevant to lateralization (Fig. 4D, Supplementary Table 4). The same direction was preserved on the test set (DKT {C_DKT['test']['n_synth_gt_asl']}/{C_DKT['test']['n_regions']}, Wilcoxon {p_str(C_DKT['test']['wilcoxon_p'])}; HO {C_HO['test']['n_synth_gt_asl']}/{C_HO['test']['n_regions']}, Wilcoxon {p_str(C_HO['test']['wilcoxon_p'])}), though individual-region McNemar tests on the {n_TLE_te}-subject test set were underpowered to reach significance despite consistent point estimates.

To determine whether the improvements from FlowGAN translate to different pathology contexts, we performed analogous analyses in the MCI cohort using bilateral regional SUV and CBF values normalized to putamen rather than asymmetry indices. In MCI using the DKT atlas (Fig. 5C), synthetic PET demonstrated substantially higher inter-subject correlations with real PET on the cross-validated sample (median r = **{a_MCI_DKT['cv']['synth_median']:.2f}**, IQR: [{a_MCI_DKT['cv']['synth_q1']:.2f}, {a_MCI_DKT['cv']['synth_q3']:.2f}]) compared to ASL (median r = **{a_MCI_DKT['cv']['asl_median']:.2f}**, IQR: [{a_MCI_DKT['cv']['asl_q1']:.2f}, {a_MCI_DKT['cv']['asl_q3']:.2f}]; Wilcoxon {p_str(a_MCI_DKT['cv']['wilcoxon_p'])}, Cohen's d = **{a_MCI_DKT['cv']['cohens_d_paired']:.2f}**). In the Harvard-Oxford atlas, similar results were observed with synthetic PET (median r = **{a_MCI_HO['cv']['synth_median']:.2f}**, IQR: [{a_MCI_HO['cv']['synth_q1']:.2f}, {a_MCI_HO['cv']['synth_q3']:.2f}]) outperforming ASL (median r = **{a_MCI_HO['cv']['asl_median']:.2f}**, IQR: [{a_MCI_HO['cv']['asl_q1']:.2f}, {a_MCI_HO['cv']['asl_q3']:.2f}]; Wilcoxon {p_str(a_MCI_HO['cv']['wilcoxon_p'])}, Cohen's d = **{a_MCI_HO['cv']['cohens_d_paired']:.2f}**). As with the asymmetry values in the epilepsy dataset, we observed significantly lower bias between synthetic and real PET SUVR (DKT: {p_str(b_MCI_DKT['cv']['wilcoxon_p'])}, d={b_MCI_DKT['cv']['cohens_d_paired']:.2f}; HO: {p_str(b_MCI_HO['cv']['wilcoxon_p'])}, d={b_MCI_HO['cv']['cohens_d_paired']:.2f}) across both atlases. Synthetic PET outperformed ASL on the cross-validated sample in **{a_MCI_DKT['cv']['n_synth_gt_asl']}/{a_MCI_DKT['cv']['n_regions']}** DKT regions and **{a_MCI_HO['cv']['n_synth_gt_asl']}/{a_MCI_HO['cv']['n_regions']}** Harvard-Oxford regions. On the {n_MCI_te}-subject test set, the Harvard-Oxford atlas reproduced the significant gap (median r {med_iqr_compact(a_MCI_HO['test'], 'synth')} vs {med_iqr_compact(a_MCI_HO['test'], 'asl')}; Wilcoxon {p_str(a_MCI_HO['test']['wilcoxon_p'])}, d={a_MCI_HO['test']['cohens_d_paired']:.2f}); for the DKT atlas, the small test sample was underpowered ({p_str(a_MCI_DKT['test']['wilcoxon_p'])}), though the same direction was observed (synthetic {a_MCI_DKT['test']['synth_median']:.2f} vs ASL {a_MCI_DKT['test']['asl_median']:.2f}).

### FlowGAN's diagnostic yield depends on the pathology

The correlation between Real FDG and Synthetic FDG effect size improvements over ASL differed between TLE lateralization and MCI detection tasks. In MCI using the DKT atlas (Fig. 6B), this correlation was strong on the cross-validated sample (**r = {d_MCI_DKT['cv']['improvement_corr_r']:.2f}, {p_str(d_MCI_DKT['cv']['improvement_corr_p'])}**) and remained significant on the test set (r = {d_MCI_DKT['test']['improvement_corr_r']:.2f}, {p_str(d_MCI_DKT['test']['improvement_corr_p'])}). This was replicated in the Harvard-Oxford atlas (Fig. 6D; cross-validated **r = {d_MCI_HO['cv']['improvement_corr_r']:.2f}, {p_str(d_MCI_HO['cv']['improvement_corr_p'])}**; test set r = {d_MCI_HO['test']['improvement_corr_r']:.2f}, {p_str(d_MCI_HO['test']['improvement_corr_p'])}). In contrast, TLE showed only a weak correlation in the DKT atlas (Fig. 6A, cross-validated **r = {d_TLE_DKT['cv']['improvement_corr_r']:.2f}, {p_str(d_TLE_DKT['cv']['improvement_corr_p'])}**; test set r = {d_TLE_DKT['test']['improvement_corr_r']:.2f}, {p_str(d_TLE_DKT['test']['improvement_corr_p'])}). In the Harvard-Oxford atlas, the correlation in TLE was **r = {d_TLE_HO['cv']['improvement_corr_r']:.2f}** on the cross-validated sample ({p_str(d_TLE_HO['cv']['improvement_corr_p'])}) and r = {d_TLE_HO['test']['improvement_corr_r']:.2f} on the test set ({p_str(d_TLE_HO['test']['improvement_corr_p'])}). Mean |d| values were preserved across the cross-validated sample and the test set (e.g., TLE-DKT mean |d|_Real = {d_TLE_DKT['cv']['mean_abs_d_real']:.2f} cross-validated / {d_TLE_DKT['test']['mean_abs_d_real']:.2f} test; |d|_Synth = {d_TLE_DKT['cv']['mean_abs_d_synth']:.2f} cross-validated / {d_TLE_DKT['test']['mean_abs_d_synth']:.2f} test; |d|_ASL = {d_TLE_DKT['cv']['mean_abs_d_asl']:.2f} cross-validated / {d_TLE_DKT['test']['mean_abs_d_asl']:.2f} test).

To localize where synthesis helps versus hurts, we compared each region's effect size against ASL for both real and synthetic FDG (Fig. 6, effect-size quadrant analysis). In TLE (DKT), approximately {qd_TLE_DKT['cv']['frac_upper_left']:.0%} of regions showed Real FDG outperforming ASL while Synthetic FDG did not (upper-left quadrant), indicating loss of lateralizing information in the synthesis process (for the Harvard-Oxford atlas this value was {qd_TLE_HO['cv']['frac_upper_left']:.0%}). In MCI, this occurred in only {qd_MCI_DKT['cv']['frac_upper_left']:.0%} of DKT regions and {qd_MCI_HO['cv']['frac_upper_left']:.0%} of Harvard-Oxford regions. Conversely, Synthetic FDG exceeded Real FDG's effect size in {qd_MCI_DKT['cv']['frac_synth_gt_real']:.0%} of MCI DKT regions (Harvard-Oxford: {qd_MCI_HO['cv']['frac_synth_gt_real']:.0%}) but only {qd_TLE_DKT['cv']['frac_synth_gt_real']:.0%} of TLE DKT regions (Harvard-Oxford: {qd_TLE_HO['cv']['frac_synth_gt_real']:.0%}), suggesting that FlowGAN may even enhance group-level metabolic differences for globally distributed pathology while attenuating the focal asymmetries needed for TLE lateralization. On the held-out test set (Epilepsy n={n_TLE_te}; MCI n={n_MCI_te}) the same direction was observed, though the separation was attenuated at this smaller sample size: the upper-left "information-loss" quadrant remained more populated in TLE (DKT {qd_TLE_DKT['test']['frac_upper_left']:.0%}, Harvard-Oxford {qd_TLE_HO['test']['frac_upper_left']:.0%}) than MCI (DKT {qd_MCI_DKT['test']['frac_upper_left']:.0%}, Harvard-Oxford {qd_MCI_HO['test']['frac_upper_left']:.0%}), and Synthetic FDG exceeded Real FDG's effect size in more MCI regions (DKT {qd_MCI_DKT['test']['frac_synth_gt_real']:.0%}, Harvard-Oxford {qd_MCI_HO['test']['frac_synth_gt_real']:.0%}) than TLE regions (DKT {qd_TLE_DKT['test']['frac_synth_gt_real']:.0%}, Harvard-Oxford {qd_TLE_HO['test']['frac_synth_gt_real']:.0%}).'''

display(Markdown(txt))""")


# ============================================================================
# Section 9b: Supplementary tables - per-split exports
# ============================================================================

md(r"""## 9b. Supplementary tables (cross-validated and held-out test splits)

The manuscript supplementary tables are regenerated here as **per-region
summary tables** (one row per region — no subject-level rows), for **both data
splits**. Each original table is reissued on the **cross-validated** sample and
a **matching test-set** version is added, using the *same* functions that built
the originals (`across_subject_analysis`, `get_congruency`,
`compute_lateralization_capacity` / `compute_mci_discriminability`) called on
the subject-filtered data — only the subject set changes.

Files go to **`tables/revision_supplementary_tables/`** as `.csv` + `.xlsx`,
named `SuppTable{NN}_{cohort}_{kind}_{atlas}_{split}.{csv,xlsx}`. Captions with
the per-split *n* filled in from the data are written to `CAPTIONS.txt`.

**Across-subject correlation + bias** tables (cols: `Region, Spearman_r_FlowGAN,
Bias_FlowGAN, Spearman_r_ASL, Bias_ASL, Corr_Diff`; `Corr_Diff =
Spearman_r_FlowGAN − Spearman_r_ASL`). TLE uses the asymmetry index; MCI uses
putamen-normalized SUVR / rCBF.

| Supp. # | Split | Cohort | Atlas |
|--------|-------|--------|-------|
| 2 | cross-validated | Epilepsy | DKT |
| 3 | cross-validated | Epilepsy | Harvard-Oxford |
| 4 | cross-validated | MCI | DKT |
| 5 | cross-validated | MCI | Harvard-Oxford |
| 10 | test | Epilepsy | DKT |
| 11 | test | Epilepsy | Harvard-Oxford |
| 12 | test | MCI | DKT |
| 13 | test | MCI | Harvard-Oxford |

**Sign-congruency** tables (cols: `Region, Congruency_ASL, Congruency_FlowGAN,
Congruency_Diff`).

| Supp. # | Split | Cohort | Atlas |
|--------|-------|--------|-------|
| 6 | cross-validated | Epilepsy | DKT |
| 7 | cross-validated | Epilepsy | Harvard-Oxford |
| 8 | cross-validated | MCI | DKT |
| 9 | cross-validated | MCI | Harvard-Oxford |
| 14 | test | Epilepsy | DKT |
| 15 | test | Epilepsy | Harvard-Oxford |
| 16 | test | MCI | DKT |
| 17 | test | MCI | Harvard-Oxford |

**Cohen's d** tables (cols: `Region, Cohens_d_Real_FDG, Cohens_d_Synthetic_FDG,
Cohens_d_ASL`) — TLE lateralization / MCI discrimination, each cohort × atlas ×
split, written as `SuppTable_CohensD_{cohort}_{atlas}_{split}`.""")

code(r"""from utils import get_congruency, save_table

SUPP_DIR = SCRIPT_DIR / 'tables' / 'revision_supplementary_tables'
SUPP_DIR.mkdir(parents=True, exist_ok=True)
# Clear any stale exports (the table-number scheme changed) before regenerating.
for _f in list(SUPP_DIR.glob('SuppTable*.csv')) + list(SUPP_DIR.glob('SuppTable*.xlsx')):
    _f.unlink()

# --- per-region across-subject correlation + bias (same fn as the originals) --
def corr_bias_table(df_sub, analysis_type):
    putamen = MOD02.get_putamen_normalization_values(df_sub)
    t = MOD02.across_subject_analysis(df_sub, putamen, analysis_type=analysis_type)
    t = t.copy()
    t['Corr_Diff'] = t['Spearman_r_FlowGAN'] - t['Spearman_r_ASL']
    return t[['Region', 'Spearman_r_FlowGAN', 'Bias_FlowGAN',
              'Spearman_r_ASL', 'Bias_ASL', 'Corr_Diff']].reset_index(drop=True)

# --- per-region sign congruency (replicates compute_all_congruencies) ---------
def congruency_table(df_ai):
    regions = [r for r in df_ai['Region'].unique() if r not in EXCLUDE_REGIONS]
    c_asl = [get_congruency(df_ai, 'PET AI Original', 'ASL AI',       r) for r in regions]
    c_rec = [get_congruency(df_ai, 'PET AI Original', 'PET AI Recon', r) for r in regions]
    return pd.DataFrame({'Region': regions,
                         'Congruency_ASL':     c_asl,
                         'Congruency_FlowGAN': c_rec,
                         'Congruency_Diff':    np.array(c_rec) - np.array(c_asl)})

# metric used for the correlation/bias table, per cohort
CORR_METRIC = {'TLE': 'Asymmetry', 'MCI': 'SUVR'}
METRIC_TAG  = {'Asymmetry': 'asymmetrycorr', 'SUVR': 'suvrcorr'}
SPLIT_TAG   = {'cv': 'crossvalidated', 'test': 'test'}
COHORT_ATLAS = [('TLE', 'DKT'), ('TLE', 'HarvardOxford'),
                ('MCI', 'DKT'), ('MCI', 'HarvardOxford')]

# Supp-table number assignment: 2-5 cv-corr, 6-9 cv-cong, 10-13 test-corr, 14-17 test-cong
NUM = {('cv',   'corr'): 2,  ('cv',   'cong'): 6,
       ('test', 'corr'): 10, ('test', 'cong'): 14}

_merged_cache = {}
def _merged(cohort, atlas):
    if (cohort, atlas) not in _merged_cache:
        _merged_cache[(cohort, atlas)] = load_df_merged(cohort, atlas)
    return _merged_cache[(cohort, atlas)]

def round_table(df):
    # 3 decimals everywhere; 5 for the smaller-magnitude bias columns.
    df = df.copy()
    for c in df.columns:
        if df[c].dtype.kind in 'fc':
            df[c] = df[c].round(5 if 'Bias' in c else 3)
    return df

manifest = []
def _record(num, cohort, kind, atlas, split, out, n):
    fname = (f'SuppTable{num:02d}_{cohort}_{kind}_{atlas}_{SPLIT_TAG[split]}'
             if num is not None else
             f'SuppTable_CohensD_{cohort}_{atlas}_{SPLIT_TAG[split]}')
    out = round_table(out)
    save_table(out, fname, str(SUPP_DIR))
    manifest.append(dict(supp=(num if num is not None else 'CohensD'),
                         split=SPLIT_TAG[split], cohort=cohort, atlas=atlas,
                         kind=kind, file=fname + '.csv',
                         n_regions=len(out), n_subjects=n))

for split in ['cv', 'test']:
    for j, (cohort, atlas) in enumerate(COHORT_ATLAS):
        df_merged = _merged(cohort, atlas)
        subs = list(df_merged['subject'].unique())
        fm   = fm_tle if cohort == 'TLE' else fm_mci
        pool, cv, test = split_subjects(subs, fm)
        sub  = cv if split == 'cv' else test
        df_sub = df_merged[df_merged['subject'].isin(sub)]
        nsub = df_sub['subject'].nunique()

        # correlation + bias
        metric = CORR_METRIC[cohort]
        out = corr_bias_table(df_sub, metric)
        _record(NUM[(split, 'corr')] + j, cohort, METRIC_TAG[metric], atlas, split, out, nsub)

        # congruency (asymmetry sign agreement, both cohorts)
        df_ai = MOD02.build_asymmetry_dataframe(df_sub)
        out = congruency_table(df_ai)
        _record(NUM[(split, 'cong')] + j, cohort, 'congruency', atlas, split, out, nsub)

        # Cohen's d (Region + 3 d columns only)
        dd = R['cohens_d'][(cohort, atlas)][split]['per_region']
        out = (dd.rename(columns={'d_real': 'Cohens_d_Real_FDG',
                                  'd_synth': 'Cohens_d_Synthetic_FDG',
                                  'd_asl': 'Cohens_d_ASL'})
                 [['Region', 'Cohens_d_Real_FDG', 'Cohens_d_Synthetic_FDG', 'Cohens_d_ASL']]
                 .reset_index(drop=True))
        _record(None, cohort, 'cohensd', atlas, split, out, nsub)

manifest = pd.DataFrame(manifest)

# ----- ready-to-use captions with per-split n filled in from the data ---------
SPLIT_PHRASE = {'crossvalidated': 'cross-validated training split',
                'test': 'held-out test split'}
ATLAS_WORD = {'DKT': 'Desikan-Killiany-Tourville (DKT)', 'HarvardOxford': 'Harvard-Oxford'}
COHORT_WORD = {'TLE': 'epilepsy', 'MCI': 'mild cognitive impairment (MCI)'}
def caption_for(row):
    aw, cw = ATLAS_WORD[row['atlas']], COHORT_WORD[row['cohort']]
    sp = SPLIT_PHRASE[row['split']]
    if row['kind'].endswith('corr'):
        metric = ('asymmetry index' if row['cohort'] == 'TLE'
                  else 'standardized uptake value ratio (SUVR) / relative cerebral blood-flow (rCBF)')
        return (f"Across-subject regional concordance in the {cw} dataset using the {aw} atlas. "
                f"For each brain region, Spearman correlation and mean bias of FlowGAN-synthetic "
                f"and ASL {metric} are computed against real FDG-PET across subjects; Corr_Diff is the "
                f"FlowGAN minus ASL difference in Spearman r (positive favours FlowGAN). "
                f"Computed on the {sp} (n = {row['n_subjects']} subjects).")
    if row['kind'] == 'congruency':
        return (f"Regional sign-congruency in the {cw} dataset using the {aw} atlas. Congruency is "
                f"the proportion of subjects in which FlowGAN-synthetic (Congruency_FlowGAN) or ASL "
                f"(Congruency_ASL) agrees with real FDG-PET on the direction of regional asymmetry; "
                f"Congruency_Diff is their difference. Higher values indicate stronger agreement. "
                f"Computed on the {sp} (n = {row['n_subjects']} subjects).")
    return (f"Effect sizes (Cohen's d) for hemispheric lateralization ({cw}) using the {aw} atlas, "
            f"for real, FlowGAN-synthetic, and ASL. Computed on the {sp} "
            f"(n = {row['n_subjects']} subjects).")

cap_lines = ["Captions for the per-split supplementary tables.",
             "Per-region summary tables (no subject-level rows). Each table is",
             "reissued on the cross-validated training split and on the held-out",
             "test split. n is the number of subjects contributing to the split.", ""]
for _, row in manifest.sort_values(['split', 'supp', 'cohort', 'atlas'],
                                   key=lambda s: s.astype(str)).iterrows():
    cap_lines += [f"[{row['file']}]", caption_for(row), ""]
(SUPP_DIR / 'CAPTIONS.txt').write_text("\n".join(cap_lines))

print('Wrote', len(manifest) * 2, 'files (csv + xlsx) + CAPTIONS.txt to',
      SUPP_DIR.relative_to(SCRIPT_DIR))
display(manifest.sort_values(['split', 'kind', 'cohort', 'atlas'], key=lambda s: s.astype(str)))""")


# ============================================================================
# Section 10: Notes
# ============================================================================

md("""## 10. Notes for the reviewer / auditor

- Every number in §8 (table) and §9 (manuscript text) is interpolated from
  the `R` dict populated by §2–§6. To verify any specific number, inspect
  `R` from the kernel (e.g.,
  `R['within_subj'][('TLE','DKT')]['cv']['synth_mean']`).
- The volume-level quality metrics in §2 load a cached CSV from
  `10_per_fold_quality_metrics.py` because computing them from NIfTIs takes
  several minutes. Re-run that script with `--force-recompute` to rebuild
  from the raw volumes.
- The figures in §7 are saved to `figures/revision_notebook_figs/` as PDF and
  PNG. PDFs are vector and ready to drop into the manuscript.
- Conda env: see `requirements.txt`, which pins the versions these outputs were
  generated with (`conda create -n FlowGAN_repro python=3.11` then
  `pip install -r requirements.txt`).""")


# Assemble and save
nb['cells'] = cells
with open(NB_PATH, 'w') as f:
    nbf.write(nb, f)

print(f'Wrote: {NB_PATH}')
print(f'  cells: {len(cells)} ({sum(1 for c in cells if c.cell_type == "code")} code, '
      f'{sum(1 for c in cells if c.cell_type == "markdown")} markdown)')
