# Code and Reproduction Scripts

This repository contains the Python scripts used to generate the results presented in the manuscript:

**"Potential Underestimation of Brugada Syndrome Prevalence: A Quantitative Reappraisal of Diagnostic Sensitivity Limitations"**

## Authors

**Ashton Christy**<sup>1</sup>, **Luke Melo**<sup>1</sup>, **Giuseppe Ciconte**<sup>2</sup>, **Luigi Anastasia**<sup>3</sup>, **Edward Grant**<sup>1</sup>, **Carlo Pappone**<sup>2,4</sup>,

<sup>1</sup> Department of Chemistry, University of British Columbia, Vancouver, BC, Canada  
<sup>2</sup> Arrhythmia and Electrophysiology Center, IRCCS Policlinico San Donato, Milan, Italy  
<sup>3</sup> Stem Cell Laboratory for Tissue Engineering, Università Vita-Salute San Raffaele, Milan, Italy  
<sup>4</sup> Department of Cardiology, Università Vita-Salute San Raffaele, Milan, Italy

## Overview of Scripts

All scripts are designed for full reproducibility of the tables and figures in the paper.

| Script | Table | Description | Key Inputs | Outputs |
|--------|-------|-------------|------------|---------|
| `psa_BrS.py` | **Table S9** | Probabilistic Sensitivity Analysis (Monte Carlo) of the two-factor correction model (`P_corrected = P_observed / (S₁ × S₂)`) across six scenarios (Baseline, Single-factor, Primary two-factor, Alternate two-factor, Exploratory three-factor, and Upper bound). | Meta-analytic prevalence, Beta distributions for S₁ and S₂ (from BHF-RASE and systematic ajmaline cohorts) | Console summary + `output/psa.csv` |
| `sex_stratified.py` | **Table S3** | Sex-stratified underascertainment analysis. Derives male and female spontaneous Type 1 detection rates (S₁) from BHF-RASE and applies Monte Carlo to quantify the male vs female diagnostic gap. | Scrocco et al. 2024 (BHF-RASE) Table 1 | Console output + `output/sex_stratified.csv` |
| `regional.py` | **Table S4** | Regionally stratified prevalence correction using observed prevalences from Vutthikraivit et al. 2018. Applies Primary (S₂ = 0.66) and Alternate (S₂ = 0.58) lead sensitivity scenarios consistently with the main analysis. | Vutthikraivit et al. 2018 meta-analysis | Console table + `output/regional.csv` |
| `MCsim_BrS_ascertainment.py` | **Table S5** | Hierarchical Monte Carlo simulation of BrS ascertainment (150,000 simulated carriers). Models heterogeneous detection probabilities and realistic ECG follow-up patterns. Includes patient classification (Always T1 / Dynamic / Never T1) and both uncensored and censored ascertainment. | Daw et al. 2022 + manuscript parameters | Console summary + Excel file (`HMCsim_BrS_ascertainment.xlsx`) |

> **Note:** Running the scripts will automatically generate output files (CSV/Excel) inside an `output/` folder. These files are not included in the repository.

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
python psa_BrS.py
python sex_stratified.py
python regional.py
python MCsim_BrS_ascertainment.py
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
