#!/usr/bin/env python3
"""
Master script for the FlowGAN manuscript reproduction package.

Runs every analysis whose outputs appear in the revised manuscript, working
entirely from the shipped regional-value tables (df_pet_merged*.pkl /
regional_data.xlsx) and fold-assignment JSONs. No raw imaging is required.

The single authoritative artifact is revision_report.ipynb, which reproduces
*every number* in the revised manuscript and regenerates the manuscript
figures. This script (a) runs the standalone figure/table scripts and then
(b) rebuilds and re-executes that notebook.

Usage:
    python run_all.py                # run scripts, then rebuild + execute notebook
    python run_all.py --skip-notebook # only run the standalone scripts
    python run_all.py --notebook-only # only rebuild + execute the notebook

Note: 01_quality_metrics.py and 10_per_fold_quality_metrics.py compute the
volume-level SSIM/PSNR/RMSE/NCC metrics from NIfTI files and therefore require
the original imaging, which is NOT included. Their output is shipped pre-computed
in tables/10_per_fold_quality_metrics/per_subject_quality_{TLE,MCI}.csv, which the
notebook reads directly. These two scripts are included for transparency only and
are intentionally not run here.
"""
import os
import sys
import time
import argparse
import subprocess
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Standalone scripts (figures + tables). DKT atlas by default; each accepts
# --include-ho where Harvard-Oxford results are also reported in the manuscript.
STEPS = [
    ("02_regional_analysis.py",        None,             "Regional correlation & Bland-Altman"),
    ("03_congruency_analysis.py",      None,             "Sign congruency & McNemar's test"),
    ("04_lateralization_cohens_d.py",  ["--include-ho"], "Cohen's d (TLE lateralization / MCI discrimination)"),
    ("11_per_fold_regional_analysis.py", None,           "Per-fold (cross-validated vs held-out) regional analysis"),
    ("13_followup_values.py",          None,             "Follow-up values requested during revision"),
]

# Runs last: it repackages outputs of the steps above *and* of the notebook
# (tables/revision_cohens_d/), so it must come after both.
SOURCE_DATA_STEP = ("14_source_data.py", None,
                    "Per-figure source-data workbooks for the main figures")


def run(cmd, label):
    print(f"\n{'='*72}\n {label}\n{'='*72}")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=SCRIPT_DIR)
    if r.returncode != 0:
        print(f"  WARNING: exited with code {r.returncode}")
    print(f"  done in {time.time()-t0:.1f}s")
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-notebook", action="store_true",
                    help="Run the standalone scripts only; do not touch the notebook.")
    ap.add_argument("--notebook-only", action="store_true",
                    help="Rebuild and execute the notebook only; skip standalone scripts.")
    args = ap.parse_args()

    print(f"FlowGAN manuscript reproduction — start {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"Working directory: {SCRIPT_DIR}")

    if not args.notebook_only:
        for script, extra, label in STEPS:
            cmd = [sys.executable, os.path.join(SCRIPT_DIR, script)]
            if extra:
                cmd += extra
            run(cmd, label)

    if not args.skip_notebook:
        run([sys.executable, os.path.join(SCRIPT_DIR, "build_revision_notebook.py")],
            "Rebuild revision_report.ipynb")
        run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
             "--execute", "--inplace",
             "--ExecutePreprocessor.timeout=1800",
             os.path.join(SCRIPT_DIR, "revision_report.ipynb")],
            "Execute revision_report.ipynb (reproduces every manuscript number)")

    script, extra, label = SOURCE_DATA_STEP
    run([sys.executable, os.path.join(SCRIPT_DIR, script)] + (extra or []), label)

    print(f"\nAll done — {datetime.now():%Y-%m-%d %H:%M:%S}")
    print("Figures -> figures/    Tables -> tables/    Notebook -> revision_report.ipynb")
    print("Main-figure source data -> source_data/")


if __name__ == "__main__":
    main()
