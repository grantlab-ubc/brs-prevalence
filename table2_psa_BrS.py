#!/usr/bin/env python3
"""
Brugada Syndrome Prevalence Reappraisal
Probabilistic Sensitivity Analysis (Monte Carlo) — Table 2 Reproduction

This script reproduces the primary two-factor correction model and all
complementary scenarios reported in Table 2 of the manuscript:

"Potential Underestimation of Brugada Syndrome Prevalence:
 A Quantitative Reappraisal of Diagnostic Sensitivity Limitations"

Core model:
    P_corrected = P_observed / (S1 × S2)

Where:
- S1 (intermittency)   : probability a confirmed carrier shows spontaneous Type 1
                         on a single ECG (~0.20 from systematic ajmaline cohorts)
- S2 (lead sensitivity): probability standard leads capture a spontaneous Type 1
                         visible with optimized high right precordial placement
                         (primary = 0.66 from BHF-RASE; alternate provocation-based
                         scenarios use ~0.58)

Distributions (chosen to reflect published data spread):
- P_observed ~ Normal(0.50, SD=0.102)          # meta-analytic 95% CI 0.3–0.7
- S1         ~ Beta(200, 800)                  # mean 0.20 (San Donato 1-3, FINGER, BHF-RASE)
- S2_primary ~ Beta(66, 34)                    # mean 0.66 (BHF-RASE spontaneous sub-study)
- S3         ~ Beta(40, 10)                    # mean 0.80 (exploratory circadian)

All results use 100,000 iterations by default (seeded for reproducibility).

Usage:
    python reproduce_table2.py
    python reproduce_table2.py --n-iter 50000 --seed 123

Outputs:
- Clean console summary of all six scenarios
- CSV file: output/table_2_psa.csv (relative to this script)

Author: Luke Melo
Manuscript version: v31 (May 2026)
"""

import argparse
import os
import sys

import numpy as np
from scipy.stats import norm, beta
import pandas as pd


def compute_corrected(p_obs, s1, s2=1.0, s3=1.0):
    """Return corrected prevalence per 1,000."""
    return p_obs / (s1 * s2 * s3)


def summarize_scenario(name, corrected, fold_increase):
    """Compute and return summary statistics for one scenario."""
    med = np.median(corrected)
    ci_low, ci_high = np.percentile(corrected, [2.5, 97.5])
    iqr_low, iqr_high = np.percentile(corrected, [25, 75])
    fold_med = np.median(fold_increase)
    fold_ci_low, fold_ci_high = np.percentile(fold_increase, [2.5, 97.5])

    return {
        "scenario": name,
        "median_corrected": round(med, 2),
        "ci_2.5": round(ci_low, 2),
        "ci_97.5": round(ci_high, 2),
        "iqr_25": round(iqr_low, 2),
        "iqr_75": round(iqr_high, 2),
        "median_fold": round(fold_med, 1),
        "fold_ci_2.5": round(fold_ci_low, 1),
        "fold_ci_97.5": round(fold_ci_high, 1),
    }


def print_scenario(name, corrected, fold_increase, inputs_str):
    """Pretty-print one scenario to console."""
    med = np.median(corrected)
    ci_low, ci_high = np.percentile(corrected, [2.5, 97.5])
    iqr_low, iqr_high = np.percentile(corrected, [25, 75])
    fold_med = np.median(fold_increase)
    fold_ci_low, fold_ci_high = np.percentile(fold_increase, [2.5, 97.5])

    print(f"\n{name}")
    print(f"  Inputs: {inputs_str}")
    print(f"  Median corrected prevalence: {med:.2f} per 1,000")
    print(f"  95% CrI: ({ci_low:.2f} – {ci_high:.2f})")
    print(f"  IQR: ({iqr_low:.2f} – {iqr_high:.2f})")
    print(f"  Median fold-increase vs published (0.5): {fold_med:.1f}-fold")
    print(f"  95% CrI fold-increase: ({fold_ci_low:.1f} – {fold_ci_high:.1f})")


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Table 2 (Monte Carlo PSA) for BrS prevalence reappraisal manuscript"
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=100_000,
        help="Number of Monte Carlo iterations (default: 100000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    n_iter = args.n_iter
    seed = args.seed

    # Determine output directory relative to this script (os-independent)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("BrS PREVALENCE REAPPRAISAL — PROBABILISTIC SENSITIVITY ANALYSIS")
    print(f"Monte Carlo with {n_iter:,} iterations  |  seed = {seed}")
    print("Reproduces Table 2 of the manuscript")
    print("=" * 80)

    # ==================== SAMPLE PARAMETERS ====================
    np.random.seed(seed)

    p_obs_samples = np.random.normal(0.50, 0.102, n_iter)
    s1_samples    = np.random.beta(200, 800, n_iter)      # S1 ~ mean 0.20
    s2_samples    = np.random.beta(66, 34, n_iter)        # Primary S2 ~ mean 0.66
    s3_samples    = np.random.beta(40, 10, n_iter)        # Exploratory S3 ~ mean 0.80

    # Clip P_observed to positive values (rare edge case)
    p_obs_samples = np.maximum(p_obs_samples, 0.01)

    # ==================== SCENARIO CALCULATIONS ====================

    # 1. Baseline (published estimate, no correction)
    baseline_corrected = p_obs_samples.copy()
    baseline_fold = baseline_corrected / 0.5

    # 2. Single-factor (S1 only)
    single_corrected = compute_corrected(p_obs_samples, s1_samples, s2=1.0, s3=1.0)
    single_fold = single_corrected / 0.5

    # 3. Primary two-factor (preferred central scenario)
    primary_corrected = compute_corrected(p_obs_samples, s1_samples, s2=s2_samples, s3=1.0)
    primary_fold = primary_corrected / 0.5

    # 4. Alternate two-factor (provocation-based S2 = 0.58)
    alt_corrected = compute_corrected(p_obs_samples, s1_samples, s2=0.58, s3=1.0)
    alt_fold = alt_corrected / 0.5

    # 5. Exploratory three-factor (S1 + S2 + S3)
    three_corrected = compute_corrected(p_obs_samples, s1_samples, s2=s2_samples, s3=s3_samples)
    three_fold = three_corrected / 0.5

    # 6. Exploratory upper (fixed lower S2 + fixed S3)
    upper_corrected = compute_corrected(p_obs_samples, s1_samples, s2=0.58, s3=0.80)
    upper_fold = upper_corrected / 0.5

    # ==================== PRINT RESULTS ====================
    print("\n" + "=" * 80)
    print("RESULTS — TABLE 2 (Monte Carlo Probabilistic Sensitivity Analysis)")
    print("=" * 80)

    print_scenario(
        "Baseline (Published estimate)",
        baseline_corrected, baseline_fold,
        "P_observed = 0.5 (no correction)"
    )

    print_scenario(
        "Single-factor (Single-ECG underdetection alone)",
        single_corrected, single_fold,
        "S1 ~ Beta(200,800)  [S2=1, S3=1 fixed]"
    )

    print_scenario(
        "Primary two-factor (Preferred central scenario)",
        primary_corrected, primary_fold,
        "S1 ~ Beta(200,800), S2 ~ Beta(66,34)"
    )

    print_scenario(
        "Alternate two-factor (Provocation-based lead scenario)",
        alt_corrected, alt_fold,
        "S1 ~ Beta(200,800), S2 = 0.58 (fixed)"
    )

    print_scenario(
        "Exploratory three-factor",
        three_corrected, three_fold,
        "S1 ~ Beta(200,800), S2 ~ Beta(66,34), S3 ~ Beta(40,10)"
    )

    print_scenario(
        "Exploratory upper",
        upper_corrected, upper_fold,
        "S1 ~ Beta(200,800), S2 = 0.58 (fixed), S3 = 0.80 (fixed)"
    )

    # ==================== SAVE RESULTS TO CSV ====================
    results = [
        summarize_scenario("Baseline", baseline_corrected, baseline_fold),
        summarize_scenario("Single-factor", single_corrected, single_fold),
        summarize_scenario("Primary two-factor", primary_corrected, primary_fold),
        summarize_scenario("Alternate two-factor", alt_corrected, alt_fold),
        summarize_scenario("Exploratory three-factor", three_corrected, three_fold),
        summarize_scenario("Exploratory upper", upper_corrected, upper_fold),
    ]

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, "table_2_psa.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("CSV saved to:", csv_path)
    print("Prevalence reported in number per thousand population.")
    print("All credible intervals from Monte Carlo probabilistic sensitivity analysis.")
    print("=" * 80)

    # Also show a compact console table
    print("\nCompact summary table:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
