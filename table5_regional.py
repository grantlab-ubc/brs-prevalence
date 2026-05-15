#!/usr/bin/env python3
"""
Brugada Syndrome — Regionally Stratified Prevalence (Table 5)

Reproduces regional corrected prevalence using:
- Observed prevalences from Vutthikraivit et al. 2018 meta-analysis
- Two S2 scenarios consistent with the main analysis:
    - Primary:   S2 = 0.66 (BHF-RASE spontaneous data)
    - Alternate: S2 = 0.58 (provocation studies)

This approach avoids strong assumptions about region-specific lead sensitivity.

Row order follows the manuscript presentation.

Author: Luke Melo
"""

import argparse
import os

import numpy as np
import pandas as pd


# =============================================================================
# DATA
# =============================================================================

REGIONS = [
    "Southeast Asia",
    "Middle East",
    "South Asia",
    "East Asia",
    "Europe",
    "North America",
]

# Observed prevalence per 1,000 (Vutthikraivit et al. 2018)
OBSERVED = {
    "Southeast Asia": 3.7,
    "Middle East":    1.8,
    "South Asia":     1.8,
    "East Asia":      1.6,
    "Europe":         0.1,
    "North America":  0.05,
}


def run_monte_carlo(n_iter=100_000, seed=42):
    np.random.seed(seed)

    # Global S1 (same as main analysis)
    s1_samples = np.random.beta(200, 800, n_iter)      # mean ≈ 0.20

    # Primary S2 (BHF-RASE)
    s2_primary = np.random.beta(66, 34, n_iter)        # mean ≈ 0.66

    # Alternate S2 (fixed at 0.58, as in Table 2)
    s2_alt = 0.58

    results = {}

    for region in REGIONS:
        observed = OBSERVED[region]

        # Primary model
        corr_primary = observed / (s1_samples * s2_primary)

        # Alternate model
        corr_alt = observed / (s1_samples * s2_alt)

        def get_stats(arr):
            med = np.median(arr)
            ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
            return round(med, 2), round(ci_low, 2), round(ci_high, 2)

        p_med, p_low, p_high = get_stats(corr_primary)
        a_med, a_low, a_high = get_stats(corr_alt)

        results[region] = {
            "observed": observed,
            "primary_median": p_med,
            "primary_2.5": p_low,
            "primary_97.5": p_high,
            "alternate_median": a_med,
            "alternate_2.5": a_low,
            "alternate_97.5": a_high,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Reproduce Table 5 - Regional BrS prevalence")
    parser.add_argument("--n-iter", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 95)
    print("BrS REGIONALLY STRATIFIED PREVALENCE — TABLE 5 REPRODUCTION")
    print(f"Monte Carlo: {args.n_iter:,} iterations | seed = {args.seed}")
    print("Observed data: Vutthikraivit et al. 2018")
    print("S2 scenarios: Primary (0.66) and Alternate (0.58) — consistent with main analysis")
    print("=" * 95)

    results = run_monte_carlo(n_iter=args.n_iter, seed=args.seed)

    print("\n" + "=" * 95)
    print("TABLE 5 — REGIONALLY STRATIFIED BrS PREVALENCE")
    print("=" * 95)

    header = f"{'Region':<18} {'Observed':>9} {'Primary (S2=0.66) Corrected (95% CrI)':<38} {'Alternate (S2=0.58) Corrected (95% CrI)':<38}"
    print(f"\n{header}")
    print("-" * 95)

    for region in REGIONS:
        r = results[region]
        print(f"{region:<18} {r['observed']:>9.2f} "
              f"{r['primary_median']:>6.2f} ({r['primary_2.5']:>5.2f}–{r['primary_97.5']:>5.2f})            "
              f"{r['alternate_median']:>6.2f} ({r['alternate_2.5']:>5.2f}–{r['alternate_97.5']:>5.2f})")

    # Save CSV
    rows = []
    for region in REGIONS:
        r = results[region]
        rows.append({
            "region": region,
            "observed_per_1000": r["observed"],
            "primary_corrected_median": r["primary_median"],
            "primary_corrected_2.5": r["primary_2.5"],
            "primary_corrected_97.5": r["primary_97.5"],
            "alternate_corrected_median": r["alternate_median"],
            "alternate_corrected_2.5": r["alternate_2.5"],
            "alternate_corrected_97.5": r["alternate_97.5"],
        })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, "table_5_regional.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 95)
    print(f"CSV saved to: {csv_path}")
    print("95% CrIs from Monte Carlo using global S1 + Primary/Alternate S2 scenarios.")
    print("=" * 95)


if __name__ == "__main__":
    main()
