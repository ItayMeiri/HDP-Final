# HDP-Final
This is the repository for the project submission in Ariel University's HDP course. 
Course site: https://aigner-horev.wixsite.com/eigen/hdp
Year: 2025. 

# Project Description:


# Environment Setup

This project uses a conda environment. Two types of files are provided for reproducibility:

- **`requirements.txt`** – captures pip-installed packages.
- **`environment.yml`** – captures the full conda environment (channels, dependencies, versions).

---

## 1. Reproducing the Environment (Recommended)

The most reliable method is to recreate the environment from the YAML file:

```bash
# Create environment from environment.yml
conda env create -f environment.yml
```


## 2. Using pip with `requirements.txt`

If you cannot use conda (e.g., deployment to a system without conda), you can still install the pip-only subset of dependencies.

```bash
# Install into current environment (virtualenv, venv, or base)
pip install -r requirements.txt
```
**Important caveats:**

- `requirements.txt` only captures pip-installed packages.  
- Packages installed exclusively through conda (e.g., `numpy` linked against MKL, `cudatoolkit`, etc.) will not be included.  
- For full reproducibility, prefer using `environment.yml`.  

