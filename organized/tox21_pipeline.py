# tox21_pipeline.py
# ==============================================================================
#  Project: High-Dimensional Analysis of the Tox21 Dataset
#  Author: Gemini AI + Additions for RMT→Motif Mapping
# ==============================================================================

import os
import csv
import collections
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator

from torch_geometric.datasets import MoleculeNet

import skdim

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from models import SimpleClassifier, Autoencoder
from utils import (
    compile_motif_definitions,
    build_mols_from_smiles,
    compute_motif_presence,
    representative_fragment_smiles_for_bit,
    enrich_motifs_for_bit,
    train_and_evaluate,
)


def main():
    FIGURES_DIR = 'figures'
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(f"All plots will be saved in the '{FIGURES_DIR}/' directory.")
    MORGAN_RADIUS = 2
    MORGAN_BITS = 2048

    print("\n--- PART 1: Loading Data and Generating Fingerprints ---")
    print("Loading Tox21 dataset from PyTorch Geometric...")
    dataset = MoleculeNet(root='.', name='Tox21')
    print("Dataset loaded successfully.")

    def generate_morgan_fingerprints(smiles_strings, radius=2, n_bits=2048):
        fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
        fingerprints = []
        for smiles in tqdm(smiles_strings, desc="Generating Fingerprints"):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = fp_generator.GetFingerprintAsNumPy(mol)
                fingerprints.append(fp)
            else:
                fingerprints.append(np.zeros(n_bits, dtype=int))
        return np.array(fingerprints)

    smiles_list = [data.smiles for data in dataset]
    fingerprint_data = generate_morgan_fingerprints(
        smiles_list, radius=MORGAN_RADIUS, n_bits=MORGAN_BITS
    )

    labels = np.array([data.y[0, 0] for data in dataset])

    print(f"\nShape of the final fingerprint data matrix: {fingerprint_data.shape}")
    print(f"We have {fingerprint_data.shape[0]} molecules, each represented by a {fingerprint_data.shape[1]}-D vector.")

    print("\n--- PART 2: Estimating Intrinsic Dimensionality ---")
    subset_size = 3000
    np.random.seed(42)
    subset_indices = np.random.choice(fingerprint_data.shape[0], subset_size, replace=False)
    data_subset = fingerprint_data[subset_indices]
    smiles_subset = [smiles_list[i] for i in subset_indices]
    print(f"Performing ID estimation on a random subset of {subset_size} data points.")
    print("Calculating pairwise Jaccard distance matrix (this may take a moment)...")
    distance_matrix = pairwise_distances(data_subset, metric='jaccard', n_jobs=-1)
    print("Distance matrix calculated.")

    print("\n--- Intrinsic Dimension Estimation Results ---")
    print("Running k-NN estimator (on raw data matrix)...")
    knn_estimator = skdim.id.KNN()
    knn_estimator.fit(data_subset)
    print(f"  - k-NN Estimate: {knn_estimator.dimension_:.2f}")

    print("Running TwoNN estimator (on raw data; no precomputed distances)...")
    X_twonn = data_subset.astype(float, copy=True)
    uniq_X, uniq_idx = np.unique(X_twonn, axis=0, return_index=True)
    if uniq_X.shape[0] < X_twonn.shape[0]:
        removed = X_twonn.shape[0] - uniq_X.shape[0]
        print(f"  - Removed {removed} exact duplicates before TwoNN.")
    X_twonn = uniq_X
    rng = np.random.default_rng(42)
    eps = 1e-6
    X_twonn += eps * rng.standard_normal(size=X_twonn.shape)
    twonn_estimator = skdim.id.TwoNN(dist=False, discard_fraction=0.1)
    twonn_estimator.fit(X_twonn)
    print(f"  - TwoNN Estimate (no precomputed dist): {twonn_estimator.dimension_:.2f}")

    print("Running MOM estimator (on raw data matrix; no precomputed kNN)...")
    mom_estimator = skdim.id.MOM()
    mom_estimator.fit(data_subset)
    print(f"  - MOM Estimate: {mom_estimator.dimension_:.2f}")

    print("\n--- Analyzing the Distribution of Neighbor Distances ---")
    print("Analyzing the Jaccard distance distribution to understand data sparsity.")
    nn_distances = np.min(distance_matrix + np.eye(distance_matrix.shape[0]) * np.inf, axis=1)
    fn_distances = np.max(distance_matrix, axis=1)
    distance_ratio = np.zeros_like(fn_distances, dtype=float)
    valid_indices = fn_distances > 0
    distance_ratio[valid_indices] = nn_distances[valid_indices] / fn_distances[valid_indices]
    distance_ratio = np.nan_to_num(distance_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    ratio_plot_path = os.path.join(FIGURES_DIR, 'neighbor_distance_ratio_distribution.png')
    plt.figure(figsize=(12, 7))
    plt.hist(distance_ratio, bins=50, density=True)
    plt.title('Distribution of (Nearest / Farthest) Neighbor Jaccard Distance Ratios', fontsize=16)
    plt.xlabel('Ratio (d_min / d_max)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.grid(alpha=0.5)
    plt.savefig(ratio_plot_path)
    plt.close()
    mean_ratio = np.mean(distance_ratio)
    print(f"  - Mean ratio of (d_min / d_max): {mean_ratio:.4f}")
    print(f"  - Visualization saved to: '{ratio_plot_path}'")

    print("\n--- PART 3: Performing Standard Random Matrix Theory Analysis ---")
    N, p = data_subset.shape
    print(f"Analyzing feature correlations for N={N} samples and p={p} features.")
    print("Calculating feature correlation matrix...")
    correlation_matrix = np.corrcoef(data_subset.T)
    correlation_matrix = np.nan_to_num(correlation_matrix)
    print("Calculating eigenvalues...")
    eigenvalues = np.linalg.eigvalsh(correlation_matrix)
    Q = p / N
    lambda_plus = (1 + np.sqrt(1 / Q)) ** 2
    lambda_minus = (1 - np.sqrt(1 / Q)) ** 2
    signal_eigenvalues = eigenvalues[eigenvalues > lambda_plus]
    num_signal_eigenvalues = len(signal_eigenvalues)
    print("\n--- Random Matrix Theory Results ---")
    print(f"  - Aspect Ratio Q (p/N): {Q:.4f}")
    print(f"  - Marchenko-Pastur Upper Bound (λ+): {lambda_plus:.4f}")
    print(f"  - Number of 'Signal' Eigenvalues (outside RMT bounds): {num_signal_eigenvalues}")
    rmt_plot_path = os.path.join(FIGURES_DIR, 'rmt_eigenvalue_spectrum.png')
    plt.figure(figsize=(12, 7))
    plt.hist(eigenvalues, bins=100, density=True, label='Empirical Eigenvalue Distribution')
    plt.axvline(x=lambda_plus, linestyle='--', lw=2, label=f'RMT Upper Bound (λ+ = {lambda_plus:.2f})')
    plt.axvline(x=lambda_minus, linestyle='--', lw=2, label=f'RMT Lower Bound (λ- = {lambda_minus:.2f})')
    plt.title('Eigenvalue Spectrum vs. Random Matrix Theory Prediction', fontsize=16)
    plt.xlabel('Eigenvalue', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.5)
    plt.savefig(rmt_plot_path)
    print(f"  - RMT plot saved to: '{rmt_plot_path}'")
    plt.close()

    print("\n--- PART 4: Advanced RMT - Analyzing Signal Eigenvectors ---")
    print("Calculating eigenvectors...")
    eigenvalues_full, eigenvectors_full = np.linalg.eigh(correlation_matrix)
    sorted_indices = np.argsort(eigenvalues_full)[::-1]
    sorted_eigenvalues = eigenvalues_full[sorted_indices]
    sorted_eigenvectors = eigenvectors_full[:, sorted_indices]
    signal_mask = sorted_eigenvalues > lambda_plus
    signal_count = int(signal_mask.sum())
    signal_eigenvectors = sorted_eigenvectors[:, :signal_count]
    print(f"Isolated {signal_count} signal eigenvectors for analysis.")

    if signal_count > 0:
        principal_eigenvector = signal_eigenvectors[:, 0]
        n_top_features = 15
        top_feature_indices = np.argsort(np.abs(principal_eigenvector))[::-1][:n_top_features]
        top_feature_weights = principal_eigenvector[top_feature_indices]
        print("\n--- Analysis of the Principal Eigenvector (Largest Signal) ---")
        print("Top feature indices and absolute loadings:")
        for i, (idx, weight) in enumerate(zip(top_feature_indices, top_feature_weights)):
            print(f"  - Rank {i + 1}: Bit {idx}, |Loading|: {abs(weight):.4f}")

    print("\n--- RMT Interpretation ---")
    print(f"RMT identified {signal_count} significant eigenvalues.")
    print("This suggests the presence of strong correlation structures in the feature space.")

    print("\n--- Connecting RMT to Intrinsic Dimension ---")
    print(f"RMT identified {signal_count} dimensions of non-random linear correlation.")
    print(
        f"ID estimators: TwoNN (raw data) ≈ {twonn_estimator.dimension_:.1f}; "
        f"MOM (raw data) ≈ {mom_estimator.dimension_:.1f}."
    )
    print(f"Both analyses indicate complexity far below the ambient dimension {p}.")

    print("\n--- PART 4B: Mapping RMT Eigenvectors to Chemical Motifs ---")
    subset_mols = build_mols_from_smiles(smiles_subset)
    motif_funcs = compile_motif_definitions()
    motif_names, motif_matrix = compute_motif_presence(subset_mols, motif_funcs)
    print(f"Computed presence for {len(motif_names)} motifs over {len(subset_mols)} molecules.")
    top_k_eigenvectors = min(5, signal_count)
    n_top_bits_each_side = 25
    min_support = 20

    if signal_count == 0:
        print("No signal eigenvectors found above the MP bound; skipping PART 4B motif mapping.")
    else:
        for ev_idx in range(top_k_eigenvectors):
            vec = signal_eigenvectors[:, ev_idx]
            pos_idx = np.argsort(vec)[::-1][:n_top_bits_each_side]
            neg_idx = np.argsort(vec)[:n_top_bits_each_side]
            chosen_bits = np.unique(np.concatenate([pos_idx, neg_idx]))
            csv_path = os.path.join(FIGURES_DIR, f"motif_enrichment_evec_{ev_idx + 1}.csv")
            rows_for_csv = []
            print(f"\nEigenvector {ev_idx + 1}: analyzing {len(chosen_bits)} top-loading bits ...")
            for bit in chosen_bits:
                bit_vec = data_subset[:, bit].astype(bool)
                enrich_rows = enrich_motifs_for_bit(bit_vec, motif_matrix, motif_names, min_support=min_support)
                frag_smiles = representative_fragment_smiles_for_bit(
                    bit_idx=int(bit),
                    mols=subset_mols,
                    radius=MORGAN_RADIUS,
                    n_bits=MORGAN_BITS,
                    max_molecules=300,
                    max_envs_per_mol=3,
                )
                loading = vec[bit]
                for er in enrich_rows:
                    er_row = {
                        "eigenvector_rank": ev_idx + 1,
                        "bit_index": int(bit),
                        "bit_loading": float(loading),
                        "fragment_smiles": frag_smiles if frag_smiles is not None else "",
                        **er,
                    }
                    rows_for_csv.append(er_row)
            rows_for_csv.sort(key=lambda r: (abs(r["z"]), -r["log2_enrichment"]), reverse=True)
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "eigenvector_rank", "bit_index", "bit_loading", "fragment_smiles",
                        "motif", "n_bit1", "n_bit0", "x_motif_bit1", "x_motif_bit0",
                        "p_motif_bit1", "p_motif_bit0", "log2_enrichment", "z", "p_two_sided",
                    ],
                )
                writer.writeheader()
                for r in rows_for_csv:
                    writer.writerow(r)
            print(f"  - Wrote motif enrichment table: {csv_path}")
            for r in rows_for_csv[:10]:
                print(
                    f"    motif={r['motif']:<18} bit={r['bit_index']:>4} "
                    f"load={r['bit_loading']:+.3f} log2_enr={r['log2_enrichment']:+.2f} "
                    f"z={r['z']:+.2f} p={r['p_two_sided']:.2e} frag='{r['fragment_smiles']}'"
                )
            print(f"\n  --- Summary for Eigenvector {ev_idx + 1} ---")
            significant_hits = [r for r in rows_for_csv if r['p_two_sided'] < 0.05 and r['log2_enrichment'] > 0]
            if not significant_hits:
                print("    No statistically significant motif enrichments found for this eigenvector's top bits.")
                continue
            motif_summary = collections.defaultdict(lambda: {'pos_loadings': 0, 'neg_loadings': 0})
            for hit in significant_hits:
                motif = hit['motif']
                if hit['bit_loading'] > 0:
                    motif_summary[motif]['pos_loadings'] += 1
                else:
                    motif_summary[motif]['neg_loadings'] += 1
            pos_motifs = sorted([m for m, v in motif_summary.items() if v['pos_loadings'] > 0])
            neg_motifs = sorted([m for m, v in motif_summary.items() if v['neg_loadings'] > 0])
            if pos_motifs or neg_motifs:
                summary_str = f"    This eigenvector appears to separate molecules based on chemical features.\n"
                if pos_motifs:
                    summary_str += f"    - Positive loadings are associated with bits enriched for: {', '.join(pos_motifs)}.\n"
                if neg_motifs:
                    summary_str += f"    - Negative loadings are associated with bits enriched for: {', '.join(neg_motifs)}."
                print(summary_str)
            else:
                print("    Could not determine a clear chemical basis for this eigenvector from motif enrichment.")

    print("\n--- PART 5: Training Models on Original and Reduced Data ---")
    print("\n--- Preparing data for supervised learning ---")
    valid_indices = ~np.isnan(labels)
    X_clean = fingerprint_data[valid_indices].astype(np.float32)
    y_clean = labels[valid_indices].astype(np.float32).reshape(-1, 1)
    print(f"Removed samples with missing labels. Usable data shape: {X_clean.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
    X_train_full_tensor = torch.from_numpy(X_train)
    X_test_full_tensor = torch.from_numpy(X_test)
    y_train_tensor = torch.from_numpy(y_train)

    print("\n--- Training Baseline Model on Full 2048-D Data ---")
    model_baseline = SimpleClassifier(input_dim=X_train.shape[1])
    auc_baseline, acc_baseline = train_and_evaluate(
        model_baseline, X_train_full_tensor, y_train_tensor, X_test_full_tensor, y_test
    )
    print(f"  - Baseline Performance: AUC = {auc_baseline:.4f}, Accuracy = {acc_baseline:.4f}")

    print(f"\n--- Training RMT-Reduced Model ({signal_count}-D Data) ---")
    X_train_rmt = X_train @ signal_eigenvectors
    X_test_rmt = X_test @ signal_eigenvectors
    scaler = StandardScaler()
    X_train_rmt_scaled = scaler.fit_transform(X_train_rmt)
    X_test_rmt_scaled = scaler.transform(X_test_rmt)
    X_train_rmt_tensor = torch.from_numpy(X_train_rmt_scaled.astype(np.float32))
    X_test_rmt_tensor = torch.from_numpy(X_test_rmt_scaled.astype(np.float32))
    model_rmt = SimpleClassifier(input_dim=X_train_rmt.shape[1])
    auc_rmt, acc_rmt = train_and_evaluate(
        model_rmt, X_train_rmt_tensor, y_train_tensor, X_test_rmt_tensor, y_test
    )
    print(f"  - RMT Performance: AUC = {auc_rmt:.4f}, Accuracy = {acc_rmt:.4f}")

    print("\n--- Training Autoencoder-Reduced Model (32-D Latent Space) ---")
    LATENT_DIM = 32
    autoencoder = Autoencoder(input_dim=X_train.shape[1], latent_dim=LATENT_DIM)
    ae_criterion = nn.MSELoss()
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    ae_train_loader = DataLoader(TensorDataset(X_train_full_tensor), batch_size=64, shuffle=True)
    autoencoder.train()
    for _ in range(20):
        for (data,) in ae_train_loader:
            recon = autoencoder(data)
            loss = ae_criterion(recon, data)
            ae_optimizer.zero_grad()
            loss.backward()
            ae_optimizer.step()
    autoencoder.eval()
    with torch.no_grad():
        X_train_ae = autoencoder.encoder(X_train_full_tensor).numpy()
        X_test_ae = autoencoder.encoder(X_test_full_tensor).numpy()
    X_train_ae_tensor = torch.from_numpy(X_train_ae)
    X_test_ae_tensor = torch.from_numpy(X_test_ae)
    model_ae = SimpleClassifier(input_dim=LATENT_DIM)
    auc_ae, acc_ae = train_and_evaluate(
        model_ae, X_train_ae_tensor, y_train_tensor, X_test_ae_tensor, y_test
    )
    print(f"  - Autoencoder Performance: AUC = {auc_ae:.4f}, Accuracy = {acc_ae:.4f}")

    print("\n\n--- Final Model Performance Comparison ---")
    print(f"{'Model':<25} | {'Feature Dim':<15} | {'Test AUC':<12} | {'Test Accuracy'}")
    print("-" * 75)
    print(f"{'Baseline (Full Data)':<25} | {X_train.shape[1]:<15} | {auc_baseline:<12.4f} | {acc_baseline:.4f}")
    print(f"{'RMT-Reduced':<25} | {X_train_rmt.shape[1]:<15} | {auc_rmt:<12.4f} | {acc_rmt:.4f}")
    print(f"{'Autoencoder-Reduced':<25} | {LATENT_DIM:<15} | {auc_ae:<12.4f} | {acc_ae:.4f}")

    results_plot_path = os.path.join(FIGURES_DIR, 'model_performance_comparison.png')
    bar_labels = ['Baseline (2048-D)', f'RMT ({signal_count}-D)', f'Autoencoder ({LATENT_DIM}-D)']
    auc_scores = [auc_baseline, auc_rmt, auc_ae]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(bar_labels, auc_scores)
    plt.ylabel('Test AUC-ROC Score')
    plt.title('Model Performance with Different Feature Sets')
    plt.ylim(min(auc_scores) - 0.05, max(auc_scores) + 0.05)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
    plt.savefig(results_plot_path)
    print(f"\nComparison plot saved to: '{results_plot_path}'")
    plt.close()
    print("\nAnalysis and modeling script finished successfully.")


if __name__ == "__main__":
    main()
