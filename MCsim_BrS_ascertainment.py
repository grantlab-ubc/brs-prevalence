# -*- coding: utf-8 -*-
"""
Hierarchical Monte Carlo Simulation of BrS Ascertainment
(50,000,000 simulated carriers)

This script reproduces the key metrics for Supplemental Table 5 in the manuscript using a
heterogeneous probability model for Brugada Type 1 ECG detection.

Model Overview
--------------
- Base per-patient detection probability (p_i) is drawn from a Beta(0.27, 1.08)
  distribution (informed by Daw et al. 2022 and the manuscript).
- Different simulation scenarios apply multiplicative correction factors to p_i
  to reflect:
    - High precordial leads (1/S₂)
    - Sex-specific spontaneous detection rates (S₁ᵐ / S₁ and S₁ᶠ / S₁)
    - Combined effects (Male/Female + High Leads)
    - Asia-specific lead sensitivity (S₂_Asia / S₂)

Key Outputs per Scenario
------------------------
- P_detection: Mean and SD of the (scaled) per-patient detection probability
- % of ECGs with Type 1
- Patient classification:
    • Always T1   = Detected on the first ECG
    • Dynamic     = Detected on a later ECG (but within observed window)
    • Never T1    = Not detected within the simulated number of ECGs
- Median ECGs until first Type 1 detection (uncensored)
- Ascertainment at a fixed 30 ECGs (theoretical)
- Censored Ascertainment: Uses realistic per-patient ECG counts (n_i) drawn
  from a Negative Binomial distribution (mean=8, dispersion=3, capped at 30).
  This reflects that most patients do not receive 30+ serial ECGs in practice.
- An alternative high-frequency setting (MAX_N = 2_880) is available (commented 
  out) for simulating Holter or intensive monitoring scenarios. 
Parameters
----------
- N_PATIENTS     = 50,000,000
- SEED           = 42
- Beta(α=0.27, β=1.08) for base p_i
- Negative Binomial for realistic n_i (used in Censored column)
- S₂ (standard)  = 0.66 (BHF-RASE)
- S₂ (Asia)      = 0.564 (Tseng et al.)
- S₁_male        = 0.31, S₁_female = 0.115 (BHF-RASE)

Output
------
- Clean summary table printed to console
- CSV saved to output/simulation.csv
"""

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

#%% User inputs

# Basics
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
EXCEL_FILE_NAME = "HMCsim_BrS_ascertainment.xlsx"
SEED = 42
N_PATIENTS = 50_000_000

# Set up ECG modelling
N_MODEL = "negative_binomial"
MIN_N = 1
MAX_N = 30
# MAX_N = 2_880 # For simulating Holter
POISSON_LAMBDA = 8.0
NB_MEAN = 8.0
NB_DISPERSION = 3.0

# Set up probability ranges
P_DISTRIBUTION = "normal"
P_RANGE_SIGMAS = 2.0

# ==================== PARAMETERS ====================
n_sims = N_PATIENTS
alpha, beta_param = 0.27, 1.08          # Overall (Daw + manuscript)
s2_std = 0.66                           # BHF-RASE standard vs high
s2_asia = 0.564                         # Tseng et al. (43.6% missed by standard in Asian cohort)

# Sex-specific means from BHF-RASE (n=358 carriers)
s1_male = 0.31
s1_female = 0.115

# ==================== SIMULATE BASE p (overall) ====================
p_base = np.clip(np.random.beta(alpha, beta_param, n_sims), 1e-8, 1.0)

# === STRATIFIED p DISTRIBUTIONS ===
p_std      = p_base.copy()                              # overall standard
p_high     = np.clip(p_base / s2_std, 0.0, 1.0)         # high precordial
p_male     = np.clip(p_base * (s1_male   / 0.20), 0.0, 1.0)   # male (scaled)
p_female   = np.clip(p_base * (s1_female / 0.20), 0.0, 1.0)   # female (scaled)
p_male_high   = np.clip(p_base * (s1_male   / 0.20 / s2_std), 0.0, 1.0)   # male (scaled)
p_female_high = np.clip(p_base * (s1_female / 0.20 / s2_std), 0.0, 1.0)   # female (scaled)
p_asia     = np.clip(p_base * (s2_asia   / s2_std), 0.0, 1.0) # Asia standard (lower S2)

# Comment these out as necessary
P_SCENARIOS = [
    {"label": "Standard"},
    {"label": "High"},
    {"label": "Male Standard"},
    {"label": "Male High"},
    {"label": "Female Standard"},
    {"label": "Female High"},
    {"label": "Asia"},
]

# Do not comment out!
p_map = {
    "Standard": p_std,
    "Global": p_std,
    "High": p_high,
    "Male": p_male,
    "Female": p_female,
    "Asia": p_asia, 
    "Male Standard": p_male,
    "Female Standard": p_female,
    "Male High": p_male_high,
    "Female High": p_female_high,
}

CORRECTION_FACTOR_MAP = {
    "Standard": "–",
    "Global": "–",
    "High": "1.515 (1/S₂)",
    "Male": "1.55 (S₁ᵐ / S₁)",
    "Male High": "2.35 (S₁ᵐ / (S₁ × S₂))",
    "Male Standard": "1.55 (S₁ᵐ / S₁)",
    "Female": "0.575 (S₁ᶠ / S₁)",
    "Female High": "0.87 (S₁ᶠ / (S₁ × S₂))",
    "Female Standard": "0.575 (S₁ᶠ / S₁)",
    "Asia": "0.85 (S₂ᴀsia / S₂)",
}

# Summary options
PRIMARY_ASCERTAINMENT_MODE = "observed_window"
SECONDARY_ASCERTAINMENT_MODE = "uncensored"
WRITE_PATIENT_LEVEL_SHEET = False # Disable when lots of scenarios are being examined... 

#%% Functions


def model_counts(rng: np.random.Generator, size: int, min_n: int, max_n: int | None) -> np.ndarray:
    """ Model the number of ECGs for each patient using a truncated Poisson or Negative Binomial distribution, as appropriate """
    
    # Model
    if N_MODEL == "poisson":
        values = rng.poisson(lam=POISSON_LAMBDA, size=size)
    elif N_MODEL == "negative_binomial":
        if NB_DISPERSION <= 0:
            raise ValueError("NB_DISPERSION must be > 0")
        lam = rng.gamma(shape=NB_DISPERSION, scale=NB_MEAN / NB_DISPERSION, size=size)
        values = rng.poisson(lam=lam)
    else:
        raise ValueError(f"Unsupported N_MODEL: {N_MODEL}")
    
    # Truncate
    mask = values < min_n
    if max_n is not None:
        mask |= values > max_n
    while np.any(mask):
        redraw_n = int(mask.sum())
        if N_MODEL == "poisson":
            values[mask] = rng.poisson(lam=POISSON_LAMBDA, size=redraw_n)
        else:
            lam = rng.gamma(shape=NB_DISPERSION, scale=NB_MEAN / NB_DISPERSION, size=redraw_n)
            values[mask] = rng.poisson(lam=lam)
        mask = values < min_n
        if max_n is not None:
            mask |= values > max_n
            
    return values.astype(int)


def summarize_probabilities(probabilities: np.ndarray) -> dict:
    """ Statistical summary """
    arr = np.asarray(probabilities, dtype=float)
    return {
        "mean": float(arr.mean()) if arr.size else float("nan"),
        "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()) if arr.size else float("nan"),
        "q1": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "q3": float(np.percentile(arr, 75)) if arr.size else float("nan"),
        "max": float(arr.max()) if arr.size else float("nan"),
    }


def summarize_counts(counts: np.ndarray) -> dict:
    """ Statistical summary """
    arr = np.asarray(counts)
    mu = float(arr.mean()) if arr.size else float("nan")
    var = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    vmr = var / mu if mu > 0 else float("inf")
    return {
        "n_patients": int(arr.size),
        "total_ecgs": int(arr.sum()) if arr.size else 0,
        "mean": mu,
        "median": float(np.median(arr)) if arr.size else float("nan"),
        "q1": float(np.percentile(arr, 25)) if arr.size else float("nan"),
        "q3": float(np.percentile(arr, 75)) if arr.size else float("nan"),
        "min": int(arr.min()) if arr.size else 0,
        "max": int(arr.max()) if arr.size else 0,
        "variance": var,
        "variance_to_mean_ratio": vmr,
    }


def assess_overdispersion(counts: np.ndarray) -> dict:
    """ Compare variance and mean to determine which model is preferable """
    summary = summarize_counts(counts)
    mu = summary["mean"]
    var = summary["variance"]
    vmr = summary["variance_to_mean_ratio"]
    if mu <= 0:
        recommendation = "insufficient"
        rationale = "Mean count is zero or undefined."
    elif vmr < 1.2:
        recommendation = "poisson"
        rationale = "Variance is close to mean; little evidence of overdispersion."
    elif vmr < 1.5:
        recommendation = "either"
        rationale = "Mild overdispersion; compare Poisson and Negative Binomial fits."
    else:
        recommendation = "negative_binomial"
        rationale = "Variance materially exceeds mean; Negative Binomial is preferred."
    nb_dispersion_mom = (mu * mu) / (var - mu) if var > mu and mu > 0 else np.nan
    out = {"recommendation": recommendation, "rationale": rationale, "nb_dispersion_mom": nb_dispersion_mom}
    out.update(summary)
    return out


def simulate_first_positive_uncensored(rng: np.random.Generator, p_i: np.ndarray) -> np.ndarray:
    """ Geometric sampling for unlimited followup """
    p = np.asarray(p_i, dtype=float)
    if np.any((p <= 0) | (p > 1)):
        raise ValueError("All p_i values must satisfy 0 < p_i <= 1")
    return rng.geometric(p).astype(int)


def simulate_patient_paths(
        rng: np.random.Generator, 
        p_i: np.ndarray, 
        n_i: np.ndarray, 
        t_first_true: np.ndarray
    ):
    """Generate observed-window ECG outcomes coherent with an uncensored first-positive time.

    Parameters
    ----------
    p_i : per-patient ECG positivity probabilities
    n_i : observed ECG counts
    t_first_true : uncensored first positive ECG index from geometric sampling

    Returns
    -------
    k_i, ever_type1, always_type1, dynamic
    """
    p_i = np.asarray(p_i, dtype=float)
    n_i = np.asarray(n_i, dtype=int)
    t_first_true = np.asarray(t_first_true, dtype=int)

    # Safety check
    n_patients = len(p_i)
    max_n = int(np.max(n_i)) if n_patients else 0
    if max_n == 0:
        empty = np.array([], dtype=int)
        return empty, empty, empty, empty, empty

    # Calculate observations
    observed = np.zeros((n_patients, max_n), dtype=bool)
    for j in range(n_patients):
        n_obs = int(n_i[j])
        if n_obs <= 0:
            continue
        first = int(t_first_true[j])
        
        # Reject if outside sampled range
        if first > n_obs:
            continue
        
        # Convert to boolean
        if first > 1:
            observed[j,:first-1] = False
        observed[j,first-1] = True
        
        # Simulate obersvations past first hit
        if first < n_obs:
            tail_len = n_obs - first
            observed[j, first:n_obs] = rng.random(tail_len) < p_i[j]

    # Summarize and calculate "class"
    k_i = observed.sum(axis=1).astype(int)
    ever_type1 = (t_first_true <= n_i).astype(int)
    always_type1 = (k_i == n_i).astype(int)
    dynamic = ((k_i > 0) & (k_i < n_i)).astype(int)

    return k_i, ever_type1, always_type1, dynamic


def simulate_cohort(
        scenario: dict,
        scenario_index: int,
        p_i_external: np.ndarray | None = None,
    ) -> pd.DataFrame:
    """ Main simulation """
    rng = np.random.default_rng(SEED + scenario_index)
    label = scenario["label"]

    # If supplied, use beta probabilities
    if p_i_external is not None:
        p_i = np.asarray(p_i_external, dtype=float)
        if p_i.size != N_PATIENTS:
            raise ValueError(f"Expected p_i length {N_PATIENTS}, got {p_i.size}")
        p_min = float(p_i.min())
        p_max = float(p_i.max())
        p_mean = float(p_i.mean())
        p_sd = float(p_i.std(ddof=1)) if p_i.size > 1 else 0.0
    else:
        raise ValueError("No probabilities supplied")

    # Generate number of ECGs
    n_i = model_counts(rng, N_PATIENTS, MIN_N, MAX_N)

    # Basic first Type 1 score
    t_first_true = simulate_first_positive_uncensored(rng, p_i)
    
    # Simulate 
    k_i, ever_type1, always_type1, dynamic = simulate_patient_paths(rng, p_i, n_i, t_first_true)

    # Summarize
    df = pd.DataFrame({
        "patient_id": np.arange(1, N_PATIENTS + 1, dtype=int),
        "p_i": p_i,
        "n_i": n_i,
        "k_i": k_i,
        "t_first": t_first_true.astype(float),
        "ever_type1": ever_type1,
        "always_type1": always_type1,
        "dynamic": dynamic,
        "scenario": label,
        "p_min": p_min,
        "p_max": p_max,
        "derived_p_mean": p_mean,
        "derived_p_sd": p_sd,
    })
    return df


def cumulative_ascertainment_by_n(df_patients: pd.DataFrame, max_n: int, mode: str) -> pd.DataFrame:
    """Calculate cumulative ascertainment at each ECG count n."""
    n_values = np.arange(1, max_n + 1, dtype=int)
    p = df_patients["p_i"].to_numpy(dtype=float)
    t = df_patients["t_first"].to_numpy(dtype=float)
    n_i = df_patients["n_i"].to_numpy(dtype=int)

    if mode == "analytic":
        ascertainment = np.array([np.mean(1.0 - (1.0 - p) ** n) for n in n_values], dtype=float)
    elif mode == "uncensored":
        ascertainment = np.array([np.mean(t <= n) for n in n_values], dtype=float)
    elif mode == "observed_window":
        ascertainment = np.array([np.mean((t <= n) & (t <= n_i)) for n in n_values], dtype=float)
    else:
        raise ValueError("Mode must be 'analytic', 'uncensored', or 'observed_window'")

    return pd.DataFrame({
        "n": n_values,
        "cumulative_ascertainment": ascertainment,
        "cumulative_nondetection": 1.0 - ascertainment,
        "mode": mode,
    })


def build_summary_dataframe(
        df_patients: pd.DataFrame, 
        df_asc_primary: pd.DataFrame,
        df_asc_secondary: pd.DataFrame, 
        scenario: dict
    ) -> pd.DataFrame:
    """ Finalize for output """
    counts = df_patients["n_i"].to_numpy(dtype=int)
    probabilities = df_patients["p_i"].to_numpy(dtype=float)
    t_first = df_patients["t_first"].to_numpy(dtype=int)
    label = scenario["label"]
    summary_rows = [
        {"scenario": label, "metric": "seed", "value": SEED},
        {"scenario": label, "metric": "n_patients", "value": N_PATIENTS},
        {"scenario": label, "metric": "p_distribution", "value": P_DISTRIBUTION},
        {"scenario": label, "metric": "p_range_sigmas", "value": P_RANGE_SIGMAS},
        {"scenario": label, "metric": "n_model", "value": N_MODEL},
        {"scenario": label, "metric": "min_n", "value": MIN_N},
        {"scenario": label, "metric": "max_n", "value": MAX_N},
        {"scenario": label, "metric": "poisson_lambda", "value": POISSON_LAMBDA},
        {"scenario": label, "metric": "nb_mean", "value": NB_MEAN},
        {"scenario": label, "metric": "nb_dispersion", "value": NB_DISPERSION},
        {"scenario": label, "metric": "primary_ascertainment_mode", "value": PRIMARY_ASCERTAINMENT_MODE},
        {"scenario": label, "metric": "secondary_ascertainment_mode", "value": SECONDARY_ASCERTAINMENT_MODE},
        {"scenario": label, "metric": "total_ecgs", "value": int(df_patients["n_i"].sum())},
        {"scenario": label, "metric": "total_type1_ecgs", "value": int(df_patients["k_i"].sum())},
        {"scenario": label, "metric": "overall_type1_ecg_fraction", "value": float(df_patients["k_i"].sum() / df_patients["n_i"].sum())},
        {"scenario": label, "metric": "ever_type1_fraction", "value": float(df_patients["ever_type1"].mean())},
        {"scenario": label, "metric": "always_type1_fraction", "value": float(df_patients["always_type1"].mean())},
        {"scenario": label, "metric": "dynamic_fraction", "value": float(df_patients["dynamic"].mean())},
        {"scenario": label, "metric": "E_trials_to_first_positive_uncensored", "value": float(np.mean(t_first))},
        {"scenario": label, "metric": "median_trials_to_first_positive_uncensored", "value": float(np.median(t_first))},
        {"scenario": label, "metric": "ascertainment_primary_at_max_n", "value": float(df_asc_primary["cumulative_ascertainment"].iloc[-1])},
        {"scenario": label, "metric": "ascertainment_secondary_at_max_n", "value": float(df_asc_secondary["cumulative_ascertainment"].iloc[-1])},
    ]
    for key, value in summarize_probabilities(probabilities).items():
        summary_rows.append({"scenario": label, "metric": f"p_i_{key}", "value": value})
    for key, value in summarize_counts(counts).items():
        summary_rows.append({"scenario": label, "metric": f"n_i_{key}", "value": value})
    for key, value in assess_overdispersion(counts).items():
        summary_rows.append({"scenario": label, "metric": f"overdispersion_{key}", "value": value})
    
    # add raw uncensored t statistics
    for key, value in summarize_probabilities(t_first).items():
        summary_rows.append({"scenario": label, "metric": f"t_inf_{key}", "value": value})
    
    # add capped / finite-window t statistics for comparison to observed-window behavior
    cap_n = MAX_N if MAX_N is not None else int(np.max(t_first))
    t_first_capped = np.minimum(t_first, cap_n)
    for key, value in summarize_probabilities(t_first_capped).items():
        summary_rows.append({"scenario": label, "metric": f"t_{cap_n}_{key}", "value": value})
    
    # optional extra summaries that are often useful
    summary_rows.append({
        "scenario": label,
        "metric": f"t_first_gt_{cap_n}_fraction",
        "value": float(np.mean(t_first > cap_n)),
    })
    summary_rows.append({
        "scenario": label,
        "metric": f"t_first_leq_{cap_n}_fraction",
        "value": float(np.mean(t_first <= cap_n)),
    })

    return pd.DataFrame(summary_rows)


def build_reptable_style_dataframe(
        df_patients: pd.DataFrame,
        scenario_label: str,
        correction_label: str = "–",
        ascertainment_n: int = 30,
    ) -> pd.DataFrame:
    """
    Create a one-row summary for a single scenario.

    Expected columns in df_patients
    ------------------------------
    p_i, n_i, k_i, t_first, ever_type1, always_type1, dynamic

    Returns
    -------
    pd.DataFrame
        One-row dataframe with Table-6 style columns.
    """
    required = {
        "p_i", "n_i", "k_i", "t_first",
        "ever_type1", "always_type1", "dynamic"
    }
    missing = required.difference(df_patients.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    p_i = df_patients["p_i"].to_numpy(dtype=float)
    n_i = df_patients["n_i"].to_numpy(dtype=int)
    t_first = df_patients["t_first"].to_numpy(dtype=int)

    # - Always T1: first ECG positive
    # - Dynamic: later first positive, but detected within observed window
    # - Never T1: not detected within observed window
    always_t1 = df_patients["always_type1"]
    dynamic_t1 = df_patients["dynamic"]
    never_t1 = (t_first > n_i)

    row = {
        "Scenario": scenario_label,
        "Correction_Factor": correction_label,
        "P_detection_Mean": round(float(np.mean(p_i)), 3),
        "P_detection_SD": round(float(np.std(p_i, ddof=1)), 3),
        "Pct_ECGs_Type1": round(float(np.mean(p_i) * 100.0), 1),
        "Pct_Always_T1": round(float(np.mean(always_t1) * 100.0), 1),
        "Pct_Dynamic": round(float(np.mean(dynamic_t1) * 100.0), 1),
        "Pct_Never_T1": round(float(np.mean(never_t1) * 100.0), 1),
        "Median_ECGs_to_First_T1": int(round(float(np.median(t_first)))),
        "Ascertainment_at_30": round(float(np.mean(t_first <= ascertainment_n) * 100.0), 1),
        "Ascertainment_Censored": round(float(np.mean(t_first <= n_i) * 100.0), 1),
    }

    return pd.DataFrame([row])


def get_median_t_by_scenario(df_patients: pd.DataFrame) -> pd.DataFrame:
    """ Calculate median t """
    out = df_patients.groupby("scenario", as_index=False)["t_first"].median().rename(columns={"t_first": "median_t"})
    out["median_t"] = out["median_t"].astype(float)
    return out


def write_excel(df_reptable: pd.DataFrame, df_summary: pd.DataFrame, df_ascertainment: pd.DataFrame, df_patients: pd.DataFrame, output_path: str) -> None:
    """ Output """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_reptable.to_excel(writer, sheet_name="report_table", index=False)
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_ascertainment.to_excel(writer, sheet_name="ascertainment", index=False)
        if WRITE_PATIENT_LEVEL_SHEET:
            df_patients.to_excel(writer, sheet_name="patients", index=False)


def simulate_scenario_wrapper(scenario: dict, scenario_index: int, p_i_external: np.ndarray) -> tuple:
    """Wrapper function for parallel processing of a single scenario on CPU."""
    df_patients = simulate_cohort(scenario, scenario_index, p_i_external=p_i_external)

    max_curve_n = MAX_N if MAX_N is not None else 40
    df_asc_primary = cumulative_ascertainment_by_n(
        df_patients, max_curve_n, PRIMARY_ASCERTAINMENT_MODE
    )
    df_asc_secondary = cumulative_ascertainment_by_n(
        df_patients, max_curve_n, SECONDARY_ASCERTAINMENT_MODE
    )

    df_asc_primary["scenario"] = scenario["label"]
    df_asc_secondary["scenario"] = scenario["label"]

    df_summary = build_summary_dataframe(df_patients, df_asc_primary, df_asc_secondary, scenario)
    df_reptable = build_reptable_style_dataframe(
        df_patients=df_patients,
        scenario_label=scenario["label"],
        correction_label=CORRECTION_FACTOR_MAP.get(scenario["label"], "–"),
        ascertainment_n=MAX_N,
    )

    return df_patients, df_asc_primary, df_asc_secondary, df_summary, df_reptable, scenario["label"]


#%%


def run_simulation() -> dict[str, pd.DataFrame]:
    """ Main loop with nice console output (now parallelized across scenarios) """
    print("=" * 90)
    print("Reproducing Table 6: Hierarchical Monte Carlo Simulation of BrS Ascertainment")
    print("=" * 90)

    print(f"\nParameters:")
    print(f"  • Simulated carriers (N)     : {N_PATIENTS:,}")
    print(f"  • Random seed                : {SEED}")
    print(f"  • Base p_i distribution      : Beta(α=0.27, β=1.08)")
    print(f"  • Realistic n_i distribution : Negative Binomial (mean={NB_MEAN}, dispersion={NB_DISPERSION}, capped at {MAX_N})")
    print(f"  • S₂ (standard leads)        : {s2_std}")
    print(f"  • S₂ (Asia)                  : {s2_asia}")
    print(f"  • S₁_male / S₁_female        : {s1_male} / {s1_female}")

    print("\nScenarios being simulated:")
    for s in P_SCENARIOS:
        cf = CORRECTION_FACTOR_MAP.get(s["label"], "–")
        print(f"  • {s['label']:<18} → Correction factor: {cf}")

    print("\nNote on 'Censored Ascertainment':")
    print("  Each simulated patient is assigned a realistic number of ECGs (n_i)")
    print("  drawn from a Negative Binomial distribution. Ascertainment is then")
    print("  calculated only within that patient's observed window (reflecting")
    print("  real-world follow-up patterns where most patients do not receive")
    print("  30+ serial ECGs).\n")

    print("Running simulation in parallel across scenarios...")
    print("-" * 90)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_patients, all_ascertainment, all_summary, all_reptable = [], [], [], []
    max_curve_n = MAX_N if MAX_N is not None else 40

    n_workers = min(len(P_SCENARIOS), os.cpu_count() or 4)
    print(f"  Using up to {n_workers} parallel CPU workers")

    tasks = [(scenario, scenario_index, p_map[scenario["label"]]) 
             for scenario_index, scenario in enumerate(P_SCENARIOS)]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(
            simulate_scenario_wrapper,
            [t[0] for t in tasks],   # scenarios (in P_SCENARIOS order)
            [t[1] for t in tasks],   # scenario_index
            [t[2] for t in tasks]    # p_i_external
        )
        for res in results:
            df_patients, df_asc_primary, df_asc_secondary, df_summary, df_reptable, label = res
            all_patients.append(df_patients)
            all_ascertainment.extend([df_asc_primary, df_asc_secondary])
            all_summary.append(df_summary)
            all_reptable.append(df_reptable)
            print(f"  ✓ Completed: {label}")

    # Combine results
    df_patients_all = pd.concat(all_patients, ignore_index=True)
    df_ascertainment_all = pd.concat(all_ascertainment, ignore_index=True)
    df_summary_all = pd.concat(all_summary, ignore_index=True)
    df_reptable_all = pd.concat(all_reptable, ignore_index=True)

    write_excel(df_reptable_all, df_summary_all, df_ascertainment_all, df_patients_all,
                os.path.join(OUTPUT_DIR, EXCEL_FILE_NAME))

    print("\n" + "=" * 90)
    print("Table 6 Simulation Complete (parallelized)")
    print("=" * 90)

    # Print the main report table
    print(df_reptable_all.to_string(index=False))

    print(f"\nCSV/Excel saved to: {os.path.join(OUTPUT_DIR, EXCEL_FILE_NAME)}")
    print("=" * 90)

    return {
        "reptable": df_reptable_all,
        "summary": df_summary_all,
        "ascertainment": df_ascertainment_all,
        "patients": df_patients_all if WRITE_PATIENT_LEVEL_SHEET else pd.DataFrame()
    }


if __name__ == "__main__":
    simulation_results = run_simulation()
