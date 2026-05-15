# Code and Reproduction Scripts

This repository contains the Python scripts used to generate the results presented in the manuscript:

**"Potential Underestimation of Brugada Syndrome Prevalence: A Quantitative Reappraisal of Diagnostic Sensitivity Limitations"**

All scripts are designed for full reproducibility of the tables and figures in the paper.

## Overview of Scripts

| Script | Table | Description | Key Inputs | Outputs |
|--------|-------|-------------|------------|---------|
| `table2_gender.py` | **Table 2** | Probabilistic Sensitivity Analysis (Monte Carlo) of the two-factor correction model (`P_corrected = P_observed / (S₁ × S₂)`) across six scenarios (Baseline, Single-factor, Primary two-factor, Alternate two-factor, Exploratory three-factor, and Upper bound). | Meta-analytic prevalence, Beta distributions for S₁ and S₂ (from BHF-RASE and systematic ajmaline cohorts) | Console summary + `output/table_2_psa.csv` |
| `table4_sex_stratified.py` | **Table 4** | Sex-stratified underascertainment analysis. Derives male and female spontaneous Type 1 detection rates (S₁) from BHF-RASE and applies Monte Carlo to quantify the male vs female diagnostic gap. | Scrocco et al. 2024 (BHF-RASE) Table 1 | Console output + `output/table_4_sex_stratified.csv` |
| `table5_regional.py` | **Table 5** | Regionally stratified prevalence correction using observed prevalences from Vutthikraivit et al. 2018. Applies Primary (S₂ = 0.66) and Alternate (S₂ = 0.58) lead sensitivity scenarios consistently with the main analysis. | Vutthikraivit et al. 2018 meta-analysis | Console table + `output/table_5_regional.csv` |
| `table6_MCsim_BrS_ascertainment.py` | **Table 6** | Hierarchical Monte Carlo simulation of BrS ascertainment (150,000 simulated carriers). Models heterogeneous detection probabilities and realistic ECG follow-up patterns. Includes patient classification (Always T1 / Dynamic / Never T1) and both uncensored and censored ascertainment. | Daw et al. 2022 + manuscript parameters | Console summary + Excel file (`table_6_HMCsim_BrS_ascertainment.xlsx`) |

## Requirements

- Python ≥ 3.10
- numpy
- pandas
- scipy

You can install the dependencies with:

```bash
pip install numpy pandas scipy
```
## How to Run

All scripts are standalone and can be run with:

```bash
python table2_gender.py
python table4_sex_stratified.py
python table5_regional.py
python table6_MCsim_BrS_ascertainment.py
```
Optional arguments (where supported):

--n-iter — Number of Monte Carlo iterations

--seed   — Random seed for reproducibility

## Reproducibility 

All scripts use fixed random seeds (default = 42).
Output files are saved in an output/ directory created automatically by each script.
Scripts are designed to be run independently.

## License

This project is licensed under the MIT License.
