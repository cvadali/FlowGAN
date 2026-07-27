# FlowGAN — Manuscript Reproduction Package

Code and extracted data to reproduce every figure, statistic, and table in the
revised manuscript on **FlowGAN**, a model that synthesizes FDG-PET images from
ASL and T1-weighted MRI. Results cover two cohorts: temporal lobe epilepsy (TLE,
asymmetry-based analysis) and mild cognitive impairment (MCI, bilateral SUVR
analysis).

Raw imaging is **not** shared. Instead, we share the regional values extracted
from the images (`regional_data.xlsx` and the matching `.pkl` files); every
downstream number, figure, and table in the manuscript is reproducible from
those values alone.

---

## 1. What reproduces the manuscript

The authoritative artifact is the Jupyter notebook **`revision_report.ipynb`**.
It recomputes **every numerical value** in the revised manuscript end-to-end from
the shared regional values and the fold-assignment files, and regenerates the
manuscript figures. No value is hard-coded — each number shown in the notebook's
text is computed in a code cell and rendered from that computation. The shipped
copy is already executed, so it can be read as-is; it can also be re-run to
confirm reproducibility.

The notebook is organized to mirror the manuscript:

| Notebook section | Manuscript content |
|---|---|
| 2. Volume-level quality metrics | SSIM, PSNR, RMSE, NCC (Fig. 3) |
| 3. Within-subject correlations | Spearman r across regions, per subject (Suppl. Fig. 3A) |
| 4. Across-subject correlations + bias | Regional correlation, Bland–Altman bias (Fig. 4B / 5C) |
| 5. Sign congruency + McNemar | Asymmetry-direction agreement (Fig. 4D) |
| 6. Cohen's d | Lateralization (TLE) / discrimination (MCI) effect sizes (Fig. 6) |
| 7. Figures | All manuscript-replacement figures → `figures/revision_notebook_figs/` |
| 8. Consolidated table | Before → after comparison |
| 9. Results text | Full Results section with values filled in |
| 9b. Supplementary tables | Cross-validated and held-out test-set tables |

**Cross-validated vs. held-out split.** Following the reviewer request, every
analysis is reported separately on a cross-validated sample (cross-validation
folds 0–9) and a held-out test set (folds 10–11). The fold membership of each
subject is defined in `data/subjects_in_each_fold_{TLE,MCI}.json`.

Alongside the notebook, standalone scripts regenerate the individual figure and
table files (see the table in section 4).

---

## 2. Setup

Verified with Python 3.11. `requirements.txt` pins the exact versions the shipped
outputs were generated with, so a fresh environment reproduces them.

```bash
conda create -n FlowGAN_repro python=3.11
conda activate FlowGAN_repro
pip install -r requirements.txt
```

---

## 3. How to run

From this directory:

```bash
# Everything: run the standalone scripts, then rebuild + execute the notebook
python run_all.py

# Only rebuild + re-execute the notebook (the full manuscript reproduction)
python run_all.py --notebook-only

# Only regenerate the standalone figure/table files
python run_all.py --skip-notebook
```

To read or re-run the notebook interactively:

```bash
jupyter notebook revision_report.ipynb    # then Kernel -> Restart & Run All
```

Outputs are written to `figures/` and `tables/` (see section 6). The package
already ships pre-generated outputs and an executed notebook; re-running
overwrites them with identical results.

---

## 4. Standalone scripts

| Script | Reproduces | Needs raw imaging? |
|---|---|---|
| `02_regional_analysis.py` | Within- and across-subject regional correlations, Bland–Altman | No |
| `03_congruency_analysis.py` | Sign-congruency rates, McNemar's test | No |
| `04_lateralization_cohens_d.py` | Cohen's d effect sizes, quadrant scatterplots | No |
| `11_per_fold_regional_analysis.py` | Per-fold (CV vs held-out) regional analysis | No |
| `13_followup_values.py` | Additional follow-up values from the revision | No |
| `14_source_data.py` | Per-figure source-data workbooks for the main figures | No |
| `10_per_fold_quality_metrics.py` | SSIM/PSNR/RMSE/NCC per fold | **Yes** (output pre-computed, see below) |
| `01_quality_metrics.py` | Volume-level quality metrics (helper for `10`) | **Yes** |
| `utils.py` | Shared helper functions | — |
| `build_revision_notebook.py` | Rebuilds `revision_report.ipynb` from source | No |

`14_source_data.py` runs last, because it repackages the outputs of the other
scripts and of the notebook. See section 8.

Scripts `04` and `11` also produce Harvard-Oxford-atlas results; pass
`--include-ho` (already the default in `run_all.py` where the manuscript reports
them).

### Volume-level quality metrics

`01` and `10` compute SSIM/PSNR/RMSE/NCC directly from NIfTI volumes, which are
not included in this package. Their per-subject output is shipped pre-computed in

```
tables/10_per_fold_quality_metrics/per_subject_quality_TLE.csv
tables/10_per_fold_quality_metrics/per_subject_quality_MCI.csv
```

and the notebook reads these CSVs to reproduce the quality-metric figures and
statistics (Fig. 3, Suppl. Tables S1–S2). `01` and `10` are included only for
transparency about how those cached numbers were produced.

To run them against your own imaging, point these environment variables at your
directories (they have no defaults, and the scripts fail with a clear message if
unset):

| Variable | Contents |
|---|---|
| `FLOWGAN_TLE_RECON_DIR` | TLE FlowGAN reconstructions, registered to the original PET |
| `FLOWGAN_TLE_BIDS_DIR` | TLE source BIDS directory (real PET and ASL) |
| `FLOWGAN_MCI_RECON_DIR` | MCI FlowGAN reconstructions, registered to the original PET |
| `FLOWGAN_MCI_SOURCE_DIR` | MCI source directory (real PET and ASL under `derivatives/pet_registration_ants/`) |

---

## 5. Shared data

Raw imaging is not shared. The regional values extracted from the images are
provided in two equivalent forms:

- **`regional_data.xlsx`** — one sheet per cohort × atlas, plus a
  `data_dictionary` sheet. Human-readable; from these values every result can be
  regenerated.
- **`df_pet_merged*.pkl`** — the identical data as pandas pickles. The code loads
  these directly, so they are kept alongside the spreadsheet to guarantee the
  scripts run unchanged.

| File / sheet | Cohort | Atlas | Subjects |
|---|---|---|---|
| `df_pet_merged.pkl` / `TLE_DKT` | TLE | DKT (native) | 68 |
| `df_pet_merged_ho.pkl` / `TLE_HarvardOxford` | TLE | Harvard-Oxford (MNI) | 68 |
| `df_pet_merged_mci.pkl` / `MCI_DKT` | MCI | DKT (native) | 85 |
| `df_pet_merged_mci_ho.pkl` / `MCI_HarvardOxford` | MCI | Harvard-Oxford (MNI) | 85 |

**Columns** (one row per subject × region × hemisphere):

| Column | Meaning |
|---|---|
| `subject` | Subject ID (`sub-RID*` for TLE, `sub-MCI*` for MCI) |
| `region_name`, `atlas_region` | Brain region (atlas parcel) |
| `side` | Hemisphere: Left, Right, or Mid |
| `atlas_index` | Numeric atlas index for the region |
| `value_pet_original` | Real FDG-PET value (putamen-normalized SUVR) |
| `value_pet_recon` | FlowGAN synthetic FDG-PET value |
| `value_pet_recon_t1_only` | Synthetic PET from T1 only (TLE/DKT only; ablation) |
| `value_asl` | ASL cerebral blood flow value |

All values are normalized by the putamen (sum of left + right), standard for
FDG-PET analysis since the putamen is relatively spared in both TLE and early MCI.

Supporting metadata in `data/`:

| File | Purpose |
|---|---|
| `subjects_in_each_fold_TLE.json`, `subjects_in_each_fold_MCI.json` | Cross-validation fold membership (folds 0–9 = CV, 10–11 = held-out test) |
| `clinical_metadata.xlsx` | TLE lateralization labels (L-TLE / R-TLE) |
| `list_of_control_subjects.txt`, `list_of_MCI_subjects.txt` | MCI healthy-control vs. patient group labels |
| `dkt.csv`, `ho.csv` | Atlas region definitions |

---

## 6. Outputs

- `revision_report.ipynb` — executed notebook: all manuscript numbers, text, and figures.
- `figures/revision_notebook_figs/` — manuscript-replacement figures (PDF + PNG).
- `figures/0X_*/` — figures from the standalone scripts.
- `tables/revision_cohens_d/` — per-region Cohen's d (one CSV per cohort × atlas × split).
- `tables/revision_supplementary_tables/` — supplementary tables (CSV + XLSX) with captions.
- `tables/0X_*/`, `tables/1X_*/` — tables from the standalone scripts.
- `source_data/` — per-figure source data for the main figures (see section 8).

---

## 8. Source data for the main figures

`14_source_data.py` writes one workbook per main figure to `source_data/`:

| Figure | File |
|---|---|
| Figure 3 — image quality metrics | `source_data/MainFigure3_SourceData.xlsx` |
| Figure 4 — TLE regional concordance (4B, 4C, 4D) | `source_data/MainFigure4_SourceData.xlsx` |
| Figure 5 — MCI regional concordance | `source_data/MainFigure5_SourceData.xlsx` |
| Figure 6 — Cohen's d quadrants and shaded surface maps | `source_data/MainFigure6_SourceData.xlsx` |

Each workbook has one sheet per figure panel plus a `data_dictionary` sheet
defining every column. Figures 1 and 2 are a schematic and representative images
and have no underlying source data. The same per-region summary values (rounded)
are also provided as the Supplementary Tables in
`tables/revision_supplementary_tables/`.

See **`SOURCE_DATA.md`** for the figure → file mapping and what each sheet
contains.

---

## 9. Excluded regions

These regions are excluded from the analyses (unreliable in one or both atlases,
or used for normalization): `unknown`, `bankssts`, `vessel`, `VentralDC`,
`temporalpole`, `frontalpole`, `corpuscallosum`, and `Putamen` (normalization
reference).
