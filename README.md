# HDP-Final
Project submission for Ariel University’s High-Dimensional Probability (HDP) course (2025).

Course site: https://aigner-horev.wixsite.com/eigen/hdp

# Project Description:
This project studies the geometry, spectra, and structure of high-dimensional data across two contrasting datasets:

**Covertype** (54D, 7 classes, 581,012 samples): mixed continuous + one-hot features. We quantify concentration effects, compare distance metrics (Euclidean, Cosine, Mahalanobis), test PCA on continuous features, and evaluate OOD detection using MLP embeddings + KDE.

**Tox21** (2048D binary Morgan fingerprints, ~8k molecules): we estimate intrinsic dimensionality, apply Random Matrix Theory (RMT) to separate signal/noise in the feature correlation spectrum, and validate reduced spaces via downstream classification.

# Repository Layout
```
HDP-Final/
├─ organized/                  # All source code (pipeline, models, analysis)
│  └─ figures/                 # All generated figures (ready for the report)
├─ runner.py                   # Single entry point; runs the full pipeline
├─ final_report.pdf            # Concise technical report with results
├─ environment.yml             # Full conda environment (recommended)
└─ requirements.txt            # pip-only subset (fallback)
```
In concise terms, `runner.py` will produce all figures and artifacts for the report. 


# Setup

## 1. Clone 
```bash
git clone https://github.com/ItayMeiri/HDP-Final
cd HDP-Final
```

## 2. Environment (recommended: conda)
```bash
# Create environment from YAML (reproducible)
conda env create -f environment.yml
# Activate (use the name defined in environment.yml)
conda activate final_project
```

Alternatively, you can use pip - this method might have limitations in some functionality.
```bash
pip install -r requirements.txt
```

## 3. Running the project
```bash
# From the repository root, with the environment activated:
python runner.py
```
This runs all the analysis modules and regenerates all figures into `organized/figures/`.
Produces artifacts mentioned by `final_report.pdf`. 

# Notes
* Python 3.13 was used for this project. It is possible some functions will work differently on older versions.
