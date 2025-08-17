# combined_covertype.py
import numpy as np

from organized.funcs import (
    load_and_sanitize_covertype,
    run_step5_extensions,
)
from organized.utils import (
    sample_off_diagonal_pairs,
    compute_pairwise_distances_for_pairs,
    histogram_overlap_coefficient,
    bootstrap_ci_mean_difference,
    subset_silhouette_score,
    plot_intra_inter_distance_histograms,
)


def _run_euclidean_and_cosine_analysis(
    ds,
    random_state: int = 42,
    sample_size: int = 10000,
    n_pairs: int = 300000,
    silhouette_subset: int = 5000,
    output_dir: str = "figures",
    dataset_tag: str = "covertype_train_combined",
) -> None:
    X_train, y_train = ds["X_train"], ds["y_train"]
    rng = np.random.default_rng(random_state)
    n_train = X_train.shape[0]
    m = min(sample_size, n_train)
    sub_idx = rng.choice(n_train, size=m, replace=False)
    Xs = X_train[sub_idx]
    ys = y_train[sub_idx]
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
        print(f"\n=== Metric: {metric} (combined) ===")
        print(f"intra:  mean={mu_intra:.6f}, var={var_intra:.6f}, n={dintra.size}")
        print(f"inter:  mean={mu_inter:.6f}, var={var_inter:.6f}, n={dinter.size}")
        print(f"Δ = E[inter] - E[intra] = {delta:.6f} (95% CI [{ci_lo:.6f}, {ci_hi:.6f}])")
        print(f"overlap coefficient (hist-based): {ovl:.6f}")
        print(f"silhouette (subset={silhouette_subset}): {sil:.6f}")
        plot_intra_inter_distance_histograms(
            dintra, dinter, metric=metric, outdir=output_dir, dataset_tag=dataset_tag
        )


def run_combined(
    random_state: int = 42,
    sample_size: int = 10000,
    n_pairs: int = 300000,
    silhouette_subset: int = 5000,
    output_dir: str = "figures",
) -> None:
    ds = load_and_sanitize_covertype(random_state=random_state)
    _run_euclidean_and_cosine_analysis(
        ds,
        random_state=random_state,
        sample_size=sample_size,
        n_pairs=n_pairs,
        silhouette_subset=silhouette_subset,
        output_dir=output_dir,
        dataset_tag="covertype_train_combined",
    )
    run_step5_extensions(ds, random_state=random_state)


def main():
    random_state = 42
    sample_size = 10000
    n_pairs = 300000
    silhouette_subset = 5000
    output_dir = "figures"
    run_combined(
        random_state=random_state,
        sample_size=sample_size,
        n_pairs=n_pairs,
        silhouette_subset=silhouette_subset,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
