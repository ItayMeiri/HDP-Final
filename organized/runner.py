# runner.py
from organized.combined_covertype import main as run_combined_covertype
from organized.tox21_pipeline import main as run_tox21
from organized.ood_covtype import main as run_ood_covtype


def main():
    # Runs everything sequentially, preserving each script's original behavior.
    run_combined_covertype()
    run_tox21()
    run_ood_covtype()


if __name__ == "__main__":
    main()
