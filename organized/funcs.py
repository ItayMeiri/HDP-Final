import os
import numpy as np
from typing import Dict, Tuple

from sklearn.covariance import EmpiricalCovariance
from sklearn.datasets import fetch_covtype
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from utils import (
    sample_off_diagonal_pairs,
    compute_pairwise_distances_for_pairs,
    histogram_overlap_coefficient,
    bootstrap_ci_mean_difference,
    subset_silhouette_score,
    plot_intra_inter_distance_histograms,
)

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def pca_reduce_continuous(
        X: np.ndarray, cont_idx: slice, bin_idx: slice, var_threshold: float = 0.95
) -> Tuple[np.ndarray, PCA]:
    """
    Reduce continuous block with PCA (retain enough components for given variance),
    keep one-hots unchanged, concatenate back.
    """
    X_cont = X[:, cont_idx]
    X_bin = X[:, bin_idx]

    pca = PCA(n_components=var_threshold, svd_solver="full")
    X_cont_red = pca.fit_transform(X_cont)

    X_mixed = np.hstack([X_cont_red, X_bin])
    return X_mixed, pca


def analyze_feature_types(X: np.ndarray):
    """
    Analyze each feature in X and print whether it is likely continuous or binary.
    Heuristics:
      - Binary: only two unique values, both in {0, 1}
      - Continuous: many unique values and wide range
    """
    n_features = X.shape[1]
    binary_features = []
    continuous_features = []
    unknown_features = []
    for i in range(n_features):
        vals = np.unique(X[:, i])
        n_unique = len(vals)
        vmin, vmax = vals.min(), vals.max()
        if n_unique == 2 and set(vals).issubset({0, 1}):
            binary_features.append(i)
            feature_type = "binary"
        elif n_unique > 20 and vmax - vmin > 1:
            continuous_features.append(i)
            feature_type = "continuous"
        else:
            unknown_features.append(i)
            feature_type = "unknown"
        print(f"Feature {i}: unique={n_unique}, min={vmin}, max={vmax}, type={feature_type}")
    print(f"\nSummary:")
    print(f"  Total features: {n_features}")
    print(f"  Continuous features: {len(continuous_features)} at indices {continuous_features}")
    print(f"  Binary features: {len(binary_features)} at indices {binary_features}")
    print(f"  Unknown/other features: {len(unknown_features)} at indices {unknown_features}")


def load_and_sanitize_covertype(
        random_state: int = 42,
        test_size: float = 0.2,
        val_size_within_train: float = 0.25,
) -> Dict[str, np.ndarray]:
    """
    Load Covertype and perform minimal, correct sanitization.

    - Covertype has 54 features: first 10 are continuous; remaining 44 are one-hots.
    - Standardize only the first 10 columns using training statistics.
    - Preserve binary one-hots as-is.

    Returns
    -------
    dict containing:
        X_train, y_train, X_val, y_val, X_test, y_test
        scaler (fitted StandardScaler for the continuous dims)
        continuous_idx (slice for the continuous dims)
        binary_idx (slice for the binary dims)
    """
    data = fetch_covtype(as_frame=False)
    X = data.data.astype(np.float64)
    y = data.target.astype(np.int64)  # labels 1..7

    # Analyze and print feature types
    analyze_feature_types(X)

    # Print statements for feature info
    n_features = X.shape[1]
    print(f"Covertype has {n_features} features.")

    # Deduce continuous features: those with many unique values and wide range
    unique_counts = np.array([len(np.unique(X[:, i])) for i in range(n_features)])
    value_ranges = np.array([np.ptp(X[:, i]) for i in range(n_features)])
    # Heuristic: continuous if unique count > 20 and range > 1
    continuous_mask = (unique_counts > 20) & (value_ranges > 1)
    continuous_indices = np.where(continuous_mask)[0]
    binary_indices = np.where(~continuous_mask)[0]

    print(f"Detected {len(continuous_indices)} continuous features at indices: {continuous_indices.tolist()}")
    print(f"First 10 features analysis:")
    for i in range(min(10, n_features)):
        print(f"  Feature {i}: unique={unique_counts[i]}, range={value_ranges[i]}")

    # Use slices if possible, else use index arrays
    if (continuous_indices[-1] - continuous_indices[0] + 1 == len(continuous_indices)):
        continuous_idx = slice(continuous_indices[0], continuous_indices[-1] + 1)
    else:
        continuous_idx = continuous_indices
    if (binary_indices[-1] - binary_indices[0] + 1 == len(binary_indices)):
        binary_idx = slice(binary_indices[0], binary_indices[-1] + 1)
    else:
        binary_idx = binary_indices

    # Stratified splits: train / (val within train) / test.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size_within_train,
        stratify=y_train_full,
        random_state=random_state,
    )

    # Standardize only the continuous block with training stats.
    scaler = StandardScaler()
    X_train_cont = scaler.fit_transform(X_train[:, continuous_idx])
    X_val_cont = scaler.transform(X_val[:, continuous_idx])
    X_test_cont = scaler.transform(X_test[:, continuous_idx])

    # Reassemble: standardized continuous + original binaries.
    X_train_std = np.hstack([X_train_cont, X_train[:, binary_idx]])
    X_val_std = np.hstack([X_val_cont, X_val[:, binary_idx]])
    X_test_std = np.hstack([X_test_cont, X_test[:, binary_idx]])

    return {
        "X_train": X_train_std,
        "y_train": y_train,
        "X_val": X_val_std,
        "y_val": y_val,
        "X_test": X_test_std,
        "y_test": y_test,
        "scaler": scaler,
        "continuous_idx": continuous_idx,
        "binary_idx": binary_idx,
    }


def implement_steps_1_and_2(
        random_state: int = 42,
        sample_size: int = 10000,
        n_pairs: int = 300000,
        silhouette_subset: int = 5000,
        output_dir: str = "figures",
) -> None:
    """
    Implements:
      (1) Load & sanitize Covertype.
      (2) Intraclass vs interclass structure with Euclidean and cosine distances.

    Produces:
      - Printed summary stats (means/variances, gap Δ, overlap coeffs, 95% CI of Δ).
      - Silhouette scores (subset).
      - Optional figures saved to 'output_dir' if matplotlib is available.
    """
    # (1) Load & sanitize
    ds = load_and_sanitize_covertype(random_state=random_state)
    X_train, y_train = ds["X_train"], ds["y_train"]

    # Draw a working subset from TRAIN for unbiased exploration (no leakage).
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    m = min(sample_size, n_train)
    sub_idx = rng.choice(n_train, size=m, replace=False)
    Xs = X_train[sub_idx]
    ys = y_train[sub_idx]

    # (2) Intraclass vs interclass via pair sampling
    pairs = sample_off_diagonal_pairs(m, n_pairs=n_pairs, rng=rng)
    same_class = ys[pairs[0]] == ys[pairs[1]]

    for metric in ("euclidean", "cosine"):
        d = compute_pairwise_distances_for_pairs(Xs, pairs, metric=metric)
        dintra = d[same_class]
        dinter = d[~same_class]

        mu_intra, var_intra = float(np.mean(dintra)), float(np.var(dintra))
        mu_inter, var_inter = float(np.mean(dinter)), float(np.var(dinter))
        delta = mu_inter - mu_intra
        ovl = histogram_overlap_coefficient(dintra, dinter, bins=256)
        ci_lo, ci_hi = bootstrap_ci_mean_difference(
            dintra, dinter, n_boot=1000, alpha=0.05, rng=rng
        )
        sil = subset_silhouette_score(
            Xs, ys, metric=metric, n_samples=silhouette_subset,
            random_state=random_state
        )

        print(f"\n=== Metric: {metric} ===")
        print(f"intra:  mean={mu_intra:.6f}, var={var_intra:.6f}, n={dintra.size}")
        print(f"inter:  mean={mu_inter:.6f}, var={var_inter:.6f}, n={dinter.size}")
        print(f"Δ = E[inter] - E[intra] = {delta:.6f} "
              f"(95% CI [{ci_lo:.6f}, {ci_hi:.6f}])")
        print(f"overlap coefficient (hist-based): {ovl:.6f}")
        print(f"silhouette (subset={silhouette_subset}): {sil:.6f}")

        plot_intra_inter_distance_histograms(
            dintra, dinter, metric=metric, outdir=output_dir,
            dataset_tag="covertype_train"
        )


def compute_mahalanobis_pairs(
        X: np.ndarray, y: np.ndarray, n_pairs: int = 200000, random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute intra- and inter-class Mahalanobis distances by random pair sampling.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Fit global covariance inverse (for whitening).
    cov_est = EmpiricalCovariance().fit(X)
    VI = cov_est.precision_

    # Sample pairs
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]

    Xi, Xj = X[i], X[j]
    diff = Xi - Xj

    # Mahalanobis squared distance = diff * VI * diff^T
    d2 = np.einsum("ij,jk,ik->i", diff, VI, diff)
    d = np.sqrt(np.maximum(d2, 0.0))

    same_class = y[i] == y[j]
    return d[same_class], d[~same_class]


def run_step5_extensions(ds: Dict[str, np.ndarray], random_state: int = 42) -> None:
    """
    Implements:
      (1) Mahalanobis intra/inter distances.
      (4) Mixed dimensionality reduction: PCA on continuous + one-hots unchanged.
    Reports summary stats analogous to Euclidean/Cosine case.
    """
    rng = np.random.default_rng(random_state)
    X_train, y_train = ds["X_train"], ds["y_train"]
    cont_idx, bin_idx = ds["continuous_idx"], ds["binary_idx"]

    # Subsample for tractability
    m = min(10000, X_train.shape[0])
    sub_idx = rng.choice(X_train.shape[0], size=m, replace=False)
    Xs, ys = X_train[sub_idx], y_train[sub_idx]

    # (1) Mahalanobis distances
    d_intra, d_inter = compute_mahalanobis_pairs(Xs, ys, n_pairs=200000, random_state=random_state)
    print("\n=== Mahalanobis ===")
    print(f"intra: mean={np.mean(d_intra):.6f}, var={np.var(d_intra):.6f}, n={d_intra.size}")
    print(f"inter: mean={np.mean(d_inter):.6f}, var={np.var(d_inter):.6f}, n={d_inter.size}")
    print(f"Δ = {np.mean(d_inter) - np.mean(d_intra):.6f}")

    # (4) PCA-reduce continuous, keep one-hots
    X_mixed, pca = pca_reduce_continuous(Xs, cont_idx, bin_idx, var_threshold=0.95)
    print("\n=== PCA on continuous only ===")
    print(f"Continuous 10 → {pca.n_components_} PCs for 95% variance")
    print(f"New total dimension = {X_mixed.shape[1]} "
          f"(was {Xs.shape[1]})")

    # Optional: compute Euclidean intra/inter again in reduced space
    i, j = sample_off_diagonal_pairs(m, n_pairs=200000, rng=rng)
    same_class = ys[i] == ys[j]
    # Efficiently compute Euclidean distances for each pair
    diff = X_mixed[i] - X_mixed[j]
    d = np.linalg.norm(diff, axis=1)
    dintra, dinter = d[same_class], d[~same_class]
    print(f"Euclidean in reduced space: Δ={np.mean(dinter) - np.mean(dintra):.6f}")
