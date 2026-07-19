#!/usr/bin/env python3
"""
Build the per-figure Supplementary Data (source data) workbooks.

Each main manuscript figure that plots data gets one .xlsx file, with one sheet
per panel plus a data_dictionary sheet:

    source_data/SourceData_Fig3.xlsx   -> Figure 3 (quality metrics)
    source_data/SourceData_Fig4.xlsx   -> Figure 4 (TLE: 4B, 4C, 4D)
    source_data/SourceData_Fig5.xlsx   -> Figure 5 (MCI: 5C and per-region delta r)
    source_data/SourceData_Fig6.xlsx   -> Figure 6 (Cohen's d quadrants + shaded maps)

Figures 1 and 2 are schematics / representative images and plot no data, so they
have no source-data file.

The values written here are the same values the figures are drawn from: this
script repackages the already-validated analysis outputs (the per-subject
quality CSVs, the per-region supplementary tables, and the Cohen's d exports)
rather than recomputing them, so the source data cannot drift from the figures.

Run 04_lateralization_cohens_d.py and the notebook (run_all.py) first, so the
inputs below exist.

Usage:
    python 14_source_data.py
"""
import os
import json
import argparse

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES = os.path.join(SCRIPT_DIR, 'tables')
SUPP = os.path.join(TABLES, 'revision_supplementary_tables')
COHD = os.path.join(TABLES, 'revision_cohens_d')
QUAL = os.path.join(TABLES, '10_per_fold_quality_metrics')
OUT = os.path.join(SCRIPT_DIR, 'source_data')

DEV_FOLDS = [f'fold_{i}' for i in range(10)]
HOLDOUT_FOLDS = ['fold_10', 'fold_11']

METRICS = ['ssim', 'psnr', 'rmse', 'ncc']


def load_fold_map(cohort):
    with open(os.path.join(SCRIPT_DIR, 'data', f'subjects_in_each_fold_{cohort}.json')) as f:
        d = json.load(f)
    return {s: fold for fold, info in d.items() for s in info.get('test', [])}


def read(path):
    if not os.path.exists(path):
        raise SystemExit(
            f'Missing input: {path}\n'
            'Run `python run_all.py` first so the analysis outputs exist.')
    return pd.read_csv(path)


def write_workbook(name, sheets, dictionary):
    """sheets: list of (sheet_name, DataFrame). dictionary: list of (sheet, column, meaning)."""
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    dd = pd.DataFrame(dictionary, columns=['sheet', 'column', 'meaning'])
    with pd.ExcelWriter(path, engine='openpyxl') as xl:
        dd.to_excel(xl, sheet_name='data_dictionary', index=False)
        for sheet, df in sheets:
            df.to_excel(xl, sheet_name=sheet[:31], index=False)
    print(f'  wrote {os.path.relpath(path, SCRIPT_DIR)}  '
          f'({len(sheets)} panel sheets + data_dictionary)')


# ---------------------------------------------------------------- Figure 3
def build_fig3():
    """Quality metrics. Panels plot per-fold means (cross-validated) and
    per-subject values (held-out test), Synthetic PET vs ASL, for each metric."""
    sheets, dd = [], []
    for cohort in ['TLE', 'MCI']:
        fm = load_fold_map(cohort)
        df = read(os.path.join(QUAL, f'per_subject_quality_{cohort}.csv')).copy()
        df['fold'] = df['subject'].map(fm)
        df['split'] = df['fold'].apply(
            lambda f: 'test' if f in HOLDOUT_FOLDS
            else ('cv' if f in DEV_FOLDS else 'unknown'))
        df = df[df['split'] != 'unknown']

        # long format: one row per subject x metric x modality
        rows = []
        for _, r in df.iterrows():
            for m in METRICS:
                for mod, col in [('Synthetic PET', f'{m}_recon'), ('ASL', f'{m}_asl')]:
                    rows.append({'subject': r['subject'], 'fold': r['fold'],
                                 'split': r['split'], 'metric': m.upper(),
                                 'modality': mod, 'value': r[col]})
        per_sub = pd.DataFrame(rows).sort_values(
            ['split', 'metric', 'modality', 'subject']).reset_index(drop=True)
        sheets.append((f'Fig3_{cohort}_per_subject', per_sub))

        # the cross-validated panel plots one point per fold (the fold mean)
        fold_means = (per_sub[per_sub['split'] == 'cv']
                      .groupby(['metric', 'modality', 'fold'], as_index=False)['value']
                      .mean()
                      .sort_values(['metric', 'modality', 'fold'])
                      .reset_index(drop=True))
        sheets.append((f'Fig3_{cohort}_cv_fold_means', fold_means))

    dd += [
        ('Fig3_<cohort>_per_subject', 'subject', 'Subject identifier'),
        ('Fig3_<cohort>_per_subject', 'fold', 'Cross-validation fold the subject was tested in'),
        ('Fig3_<cohort>_per_subject', 'split', 'cv = folds 0-9; test = held-out folds 10-11'),
        ('Fig3_<cohort>_per_subject', 'metric', 'SSIM, PSNR, RMSE or NCC'),
        ('Fig3_<cohort>_per_subject', 'modality', 'Synthetic PET (FlowGAN) or ASL'),
        ('Fig3_<cohort>_per_subject', 'value', 'Metric value vs real FDG-PET for that subject'),
        ('Fig3_<cohort>_cv_fold_means', 'fold', 'Cross-validation fold (0-9)'),
        ('Fig3_<cohort>_cv_fold_means', 'value',
         'Mean metric across subjects in that fold; these are the points plotted '
         'in the cross-validated panel of Figure 3'),
    ]
    write_workbook('SourceData_Fig3.xlsx', sheets, dd)


# ------------------------------------------------- Figures 4 and 5 (per-region)
# Source the per-region values from 11_per_fold_regional_analysis/, which stores
# them at full precision. The equivalent supplementary tables are rounded to 3
# decimals, which is fine for reading but would not redraw the figures exactly.
# In those files the split suffixes are: _dev = cross-validated, _ho = held-out test.
PERFOLD = os.path.join(TABLES, '11_per_fold_regional_analysis')

SPLIT_SUFFIX = {'cv': 'dev', 'test': 'ho'}
ATLAS_TAG = {'DKT': 'DKT', 'HarvardOxford': 'HO'}


def _corr_panel(cohort, atlas, split):
    """Per-region across-subject Spearman r, Synthetic PET vs ASL (Fig 4B / 5C),
    and their difference (the forest-bar height in Fig 4C)."""
    sfx = SPLIT_SUFFIX[split]
    df = read(os.path.join(PERFOLD, f'across_subject_{cohort}_{atlas}.csv'))
    out = pd.DataFrame({
        'Region': df['Region'],
        'cohort': cohort,
        'atlas': atlas,
        'split': split,
        'n_subjects': df[f'n_{sfx}'],
        'Spearman_r_FlowGAN': df[f'r_synth_{sfx}'],
        'Spearman_r_ASL': df[f'r_asl_{sfx}'],
    })
    out['Delta_r_FlowGAN_minus_ASL'] = out['Spearman_r_FlowGAN'] - out['Spearman_r_ASL']
    return out


def _cong_panel(cohort, atlas, split):
    """Per-region sign congruency vs real FDG-PET (Fig 4D)."""
    sfx = SPLIT_SUFFIX[split]
    df = read(os.path.join(PERFOLD, f'congruency_{cohort}_{atlas}.csv'))
    out = pd.DataFrame({
        'Region': df['Region'],
        'cohort': cohort,
        'atlas': atlas,
        'split': split,
        'n_subjects': df[f'n_{sfx}'],
        'Congruency_FlowGAN': df[f'cong_synth_{sfx}'],
        'Congruency_ASL': df[f'cong_asl_{sfx}'],
    })
    out['Congruency_Diff'] = out['Congruency_FlowGAN'] - out['Congruency_ASL']
    return out


def build_fig4():
    """TLE. 4B: per-region asymmetry Spearman r (Synthetic vs ASL).
    4C: per-region delta r forest. 4D: per-region sign congruency."""
    sheets, dd = [], []
    for atlas in ['DKT', 'HarvardOxford']:
        for split in ['cv', 'test']:
            tag = ATLAS_TAG[atlas]
            sheets.append((f'Fig4BC_{tag}_{split}', _corr_panel('TLE', atlas, split)))
            sheets.append((f'Fig4D_{tag}_{split}', _cong_panel('TLE', atlas, split)))
    dd += [
        ('Fig4BC_<atlas>_<split>', 'Region', 'Brain region (atlas parcel)'),
        ('Fig4BC_<atlas>_<split>', 'split', 'cv = folds 0-9; test = held-out folds 10-11'),
        ('Fig4BC_<atlas>_<split>', 'n_subjects', 'Number of subjects contributing to this split'),
        ('Fig4BC_<atlas>_<split>', 'Spearman_r_FlowGAN',
         'Across-subject Spearman r between FlowGAN and real FDG-PET asymmetry index. '
         'One point in the Synthetic PET box of Fig 4B'),
        ('Fig4BC_<atlas>_<split>', 'Spearman_r_ASL',
         'Across-subject Spearman r between ASL and real FDG-PET asymmetry index. '
         'One point in the ASL box of Fig 4B'),
        ('Fig4BC_<atlas>_<split>', 'Delta_r_FlowGAN_minus_ASL',
         'Spearman_r_FlowGAN minus Spearman_r_ASL; the bar height plotted in Fig 4C '
         '(positive favours FlowGAN)'),
        ('Fig4D_<atlas>_<split>', 'Congruency_FlowGAN',
         'Proportion of subjects where FlowGAN agrees with real FDG-PET on the '
         'direction of regional asymmetry (plotted in Fig 4D)'),
        ('Fig4D_<atlas>_<split>', 'Congruency_ASL',
         'Same, for ASL (plotted in Fig 4D)'),
        ('Fig4D_<atlas>_<split>', 'Congruency_Diff', 'Congruency_FlowGAN minus Congruency_ASL'),
    ]
    write_workbook('SourceData_Fig4.xlsx', sheets, dd)


def build_fig5():
    """MCI. 5C: per-region SUVR Spearman r (Synthetic vs ASL), plus delta r forest."""
    sheets, dd = [], []
    for atlas in ['DKT', 'HarvardOxford']:
        for split in ['cv', 'test']:
            tag = ATLAS_TAG[atlas]
            sheets.append((f'Fig5C_{tag}_{split}', _corr_panel('MCI', atlas, split)))
    dd += [
        ('Fig5C_<atlas>_<split>', 'Region', 'Brain region (atlas parcel)'),
        ('Fig5C_<atlas>_<split>', 'split', 'cv = folds 0-9; test = held-out folds 10-11'),
        ('Fig5C_<atlas>_<split>', 'n_subjects', 'Number of subjects contributing to this split'),
        ('Fig5C_<atlas>_<split>', 'Spearman_r_FlowGAN',
         'Across-subject Spearman r between FlowGAN and real FDG-PET SUVR. '
         'One point in the Synthetic PET box of Fig 5C'),
        ('Fig5C_<atlas>_<split>', 'Spearman_r_ASL',
         'Across-subject Spearman r between ASL rCBF and real FDG-PET SUVR. '
         'One point in the ASL box of Fig 5C'),
        ('Fig5C_<atlas>_<split>', 'Delta_r_FlowGAN_minus_ASL',
         'Spearman_r_FlowGAN minus Spearman_r_ASL; the bar height plotted in the '
         'per-region delta r forest (positive favours FlowGAN)'),
    ]
    write_workbook('SourceData_Fig5.xlsx', sheets, dd)


# ---------------------------------------------------------------- Figure 6
def build_fig6():
    """Cohen's d. Quadrant scatterplots (Fig 6 / 6B) and the cortical-surface
    shaded maps, which are rendered from atlas_index + Cohen's d."""
    sheets, dd = [], []
    for cohort in ['TLE', 'MCI']:
        for tag in ['DKT', 'HO']:
            for split in ['cv', 'test']:
                df = read(os.path.join(COHD, f'cohens_d_{cohort}_{tag}_{split}.csv')).copy()
                df.insert(1, 'split', split)
                df.insert(1, 'atlas', 'DKT' if tag == 'DKT' else 'HarvardOxford')
                df.insert(1, 'cohort', cohort)
                sheets.append((f'Fig6_{cohort}_{tag}_{split}', df))
    dd += [
        ('Fig6_<cohort>_<atlas>_<split>', 'region', 'Brain region (atlas parcel)'),
        ('Fig6_<cohort>_<atlas>_<split>', 'atlas_index_left',
         'Numeric atlas index, left hemisphere; used to render the shaded surface maps'),
        ('Fig6_<cohort>_<atlas>_<split>', 'atlas_index_right',
         'Numeric atlas index, right hemisphere; used to render the shaded surface maps'),
        ('Fig6_<cohort>_<atlas>_<split>', 'cohens_d_real',
         "Cohen's d from real FDG-PET (TLE: L-TLE vs R-TLE lateralization; "
         'MCI: MCI vs control discrimination)'),
        ('Fig6_<cohort>_<atlas>_<split>', 'cohens_d_synthetic', "Cohen's d from FlowGAN synthetic PET"),
        ('Fig6_<cohort>_<atlas>_<split>', 'cohens_d_asl', "Cohen's d from ASL"),
        ('Fig6_<cohort>_<atlas>_<split>', 'comp_x_synth_minus_asl',
         "x-axis of the Fig 6B comparison quadrant scatter (synthetic minus ASL Cohen's d)"),
        ('Fig6_<cohort>_<atlas>_<split>', 'comp_y_real_minus_asl',
         "y-axis of the Fig 6B comparison quadrant scatter (real minus ASL Cohen's d)"),
        ('Fig6_<cohort>_<atlas>_<split>', 'comparison_quadrant',
         'Which Fig 6B quadrant the region falls in'),
        ('Fig6_<cohort>_<atlas>_<split>', 'comparison_color', 'Plot colour for comparison_quadrant'),
        ('Fig6_<cohort>_<atlas>_<split>', 'quadrant_category',
         'asl_gt_synth / similar / synth_gt_asl; the category shaded on the surface maps'),
        ('Fig6_<cohort>_<atlas>_<split>', 'quadrant_color', 'Plot colour for quadrant_category'),
    ]
    write_workbook('SourceData_Fig6.xlsx', sheets, dd)


def main():
    argparse.ArgumentParser(description=__doc__,
                            formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print('Building per-figure source data workbooks ->', os.path.relpath(OUT, SCRIPT_DIR))
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    print('Done.')


if __name__ == '__main__':
    main()
