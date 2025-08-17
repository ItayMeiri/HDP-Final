# utils.py
import math
import collections
from typing import Callable, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, silhouette_score
from sklearn.neighbors import KernelDensity

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# RDKit (used by motif utilities)
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFingerprintGenerator


# ---------------------------- Printing / Reporting -----------------------------
def print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f"--- {title} ---")
    print("=" * 80)


def calculate_ood_metrics(
    scores_in: np.ndarray,
    scores_out: np.ndarray,
    label: str,
    tpr_level: float = 0.95,
) -> Tuple[float, float]:
    labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    scores = np.concatenate([scores_in, scores_out])
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, tpr_level)
    fpr_at_tpr = fpr[idx] if idx < len(fpr) else 1.0
    print(f"OOD Detection Performance ({label}):")
    print(f"  - AUROC: {auroc:.4f}")
    print(f"  - FPR at {tpr_level * 100:.0f}% TPR: {fpr_at_tpr:.4f}")
    return auroc, fpr_at_tpr


def plot_training_summary(history: dict, final_acc: float, filepath: str) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color)
    ax1.plot(history['loss'], color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color)
    ax2.plot(history['acc'], color=color, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title(f'Training Summary (Final Val Acc: {final_acc:.2%})')
    fig.tight_layout()
    plt.savefig(filepath)
    plt.close()
    print(f"Training summary plot saved to {filepath}")


def plot_density_distributions(
    scores_in: np.ndarray,
    scores_hard: np.ndarray,
    scores_easy: np.ndarray,
    title: str,
    filepath: str
) -> None:
    plt.figure(figsize=(10, 6))
    sns.kdeplot(scores_in, fill=True, label='In-Distribution (Test)')
    sns.kdeplot(scores_hard, fill=True, label='OOD (Hard - Held-out Classes)')
    sns.kdeplot(scores_easy, fill=True, label='OOD (Easy - Synthetic Noise)')
    plt.title(title)
    plt.xlabel("Log-Likelihood Score (Higher is more In-Distribution)")
    plt.ylabel("Density")
    if scores_in.size > 0 and scores_hard.size > 0:
        lower_bound = np.percentile(scores_hard, 0.5)
        upper_bound = np.percentile(scores_in, 99.5)
        if lower_bound < upper_bound:
            plt.xlim(lower_bound, upper_bound)
    plt.legend()
    plt.savefig(filepath)
    plt.close()
    print(f"Density plot saved to {filepath}")


# ---------------------------- Embedding utilities ------------------------------
def extract_embeddings(model, loader) -> np.ndarray:
    model.eval()
    embeddings = []
    with torch.no_grad():
        for inputs, _ in loader:
            inputs = inputs.to(next(model.parameters()).device)
            emb = model.get_embedding(inputs)
            embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)


def run_ood_analysis(
    title: str,
    embeddings_in: np.ndarray,
    embeddings_hard: np.ndarray,
    embeddings_easy: np.ndarray,
    kde_subset_size: int = 8000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print_header(title)
    subset_indices = np.random.choice(len(embeddings_in), kde_subset_size, replace=False)
    kde = KernelDensity(kernel='gaussian', bandwidth='scott').fit(embeddings_in[subset_indices])
    scores_in = kde.score_samples(embeddings_in)
    scores_hard = kde.score_samples(embeddings_hard)
    scores_easy = kde.score_samples(embeddings_easy)
    print("\n--- OOD Detection on 'Easy' (Synthetic) Data ---")
    calculate_ood_metrics(scores_in, scores_easy, "Easy OOD")
    print("\n--- OOD Detection on 'Hard' (Held-out Classes) Data ---")
    calculate_ood_metrics(scores_in, scores_hard, "Hard OOD")
    return scores_in, scores_hard, scores_easy


# ------------------------------ Training helper --------------------------------
def train_and_evaluate(
    model: torch.nn.Module,
    X_train_tensor: torch.Tensor,
    y_train_tensor: torch.Tensor,
    X_test_tensor: torch.Tensor,
    y_test_np: np.ndarray
) -> Tuple[float, float]:
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for _ in range(15):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        y_pred_proba = model(X_test_tensor).numpy()
    auc = roc_auc_score(y_test_np, y_pred_proba)
    acc = accuracy_score(y_test_np, (y_pred_proba > 0.5).astype(int))
    return auc, acc


# --------------------------- Motif/morgan utilities ----------------------------
def compile_motif_definitions() -> Dict[str, Callable[[Chem.Mol], bool]]:
    smarts = {
        "halogen": "[F,Cl,Br,I]",
        "nitro": "[N+](=O)[O-]",
        "carbonyl": "[CX3]=O",
        "amide": "C(=O)N",
        "ester": "[CX3](=O)O[#6]",
        "sulfonamide": "S(=O)(=O)N",
        "hydroxyl": "[OX2H]",
    }
    smarts_mols = {k: Chem.MolFromSmarts(v) for k, v in smarts.items()}

    def has_aromatic_ring(m: Chem.Mol) -> bool:
        return rdMolDescriptors.CalcNumAromaticRings(m) > 0

    def has_heteroaromatic(m: Chem.Mol) -> bool:
        ri = m.GetRingInfo()
        atom_rings = ri.AtomRings()
        for ring in atom_rings:
            if all(m.GetAtomWithIdx(a).GetIsAromatic() for a in ring):
                if any(m.GetAtomWithIdx(a).GetAtomicNum() in (7, 8, 16) for a in ring):
                    return True
        return False

    motif_funcs: Dict[str, Callable[[Chem.Mol], bool]] = {
        "aromatic_ring": has_aromatic_ring,
        "heteroaromatic_ring": has_heteroaromatic,
    }

    for name, patt in smarts_mols.items():
        if patt is None:
            continue

        def make_f(mol_pattern):
            def f(m):
                return m.HasSubstructMatch(mol_pattern)
            return f

        motif_funcs[name] = make_f(patt)

    return motif_funcs


def build_mols_from_smiles(smiles_list: List[str]) -> List[Chem.Mol]:
    mols: List[Chem.Mol] = []
    for s in smiles_list:
        try:
            mols.append(Chem.MolFromSmiles(s))
        except Exception:
            mols.append(None)
    return mols


def compute_motif_presence(
    mols: List[Chem.Mol],
    motif_funcs: Dict[str, Callable[[Chem.Mol], bool]]
) -> Tuple[List[str], np.ndarray]:
    motif_names = list(motif_funcs.keys())
    n = len(mols)
    m = len(motif_names)
    out = np.zeros((n, m), dtype=bool)
    for i, mol in enumerate(mols):
        if mol is None:
            continue
        for j, name in enumerate(motif_names):
            try:
                out[i, j] = bool(motif_funcs[name](mol))
            except Exception:
                out[i, j] = False
    return motif_names, out


def two_prop_z_test(x1: int, n1: int, x2: int, n2: int, eps: float = 1e-9) -> Tuple[float, float]:
    p1 = x1 / max(n1, eps)
    p2 = x2 / max(n2, eps)
    p_pool = (x1 + x2) / max(n1 + n2, eps)
    denom = math.sqrt(max(p_pool * (1.0 - p_pool) * (1.0 / max(n1, eps) + 1.0 / max(n2, eps)), eps))
    z = (p1 - p2) / denom
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return z, p


def representative_fragment_smiles_for_bit(
    bit_idx: int,
    mols: List[Chem.Mol],
    radius: int,
    n_bits: int,
    max_molecules: int = 500,
    max_envs_per_mol: int = 3,
) -> str:
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    counts = collections.Counter()
    seen = 0
    for mol in mols:
        if mol is None:
            continue
        bit_info = {}
        try:
            _ = fp_generator.GetFingerprint(mol, bitInfo=bit_info)
        except Exception:
            continue
        if bit_idx not in bit_info:
            continue
        envs = bit_info.get(bit_idx, [])[:max_envs_per_mol]
        for (center, rad) in envs:
            try:
                if rad == 0:
                    frag_smiles = Chem.MolFragmentToSmiles(
                        mol, atomsToUse=[center], kekuleSmiles=True
                    )
                    counts[frag_smiles] += 1
                else:
                    bond_ids = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, center)
                    if not bond_ids:
                        frag_smiles = Chem.MolFragmentToSmiles(
                            mol, atomsToUse=[center], kekuleSmiles=True
                        )
                        counts[frag_smiles] += 1
                    else:
                        sub = Chem.PathToSubmol(mol, bond_ids)
                        frag_smiles = Chem.MolToSmiles(sub, kekuleSmiles=True)
                        counts[frag_smiles] += 1
            except Exception:
                continue
        seen += 1
        if seen >= max_molecules:
            break
    if not counts:
        return None
    frag, _ = counts.most_common(1)[0]
    return frag


def enrich_motifs_for_bit(
    bit_vector: np.ndarray,
    motif_matrix: np.ndarray,
    motif_names: List[str],
    min_support: int = 20
) -> List[dict]:
    bit_mask = bit_vector.astype(bool)
    n = bit_mask.shape[0]
    n1 = int(bit_mask.sum())
    n0 = int((~bit_mask).sum())
    rows: List[dict] = []
    if n1 < min_support or n0 < min_support:
        return rows
    for j, name in enumerate(motif_names):
        motif_col = motif_matrix[:, j]
        x1 = int((motif_col & bit_mask).sum())
        x0 = int((motif_col & (~bit_mask)).sum())
        p1 = (x1 + 1e-9) / (n1 + 1e-9)
        p0 = (x0 + 1e-9) / (n0 + 1e-9)
        log2_enr = math.log2(p1 / p0) if p0 > 0 else float("inf")
        z, p = two_prop_z_test(x1, n1, x0, n0)
        rows.append({
            "motif": name,
            "n_bit1": n1,
            "n_bit0": n0,
            "x_motif_bit1": x1,
            "x_motif_bit0": x0,
            "p_motif_bit1": p1,
            "p_motif_bit0": p0,
            "log2_enrichment": log2_enr,
            "z": z,
            "p_two_sided": p,
        })
    rows.sort(key=lambda r: (abs(r["z"]), -r["log2_enrichment"]), reverse=True)
    return rows


# ---------------------- General statistical/ML utilities -----------------------
def sample_off_diagonal_pairs(
    n: int, n_pairs: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample (i, j) pairs uniformly with i != j."""
    i = rng.integers(0, n, size=n_pairs, endpoint=False)
    j = rng.integers(0, n, size=n_pairs, endpoint=False)
    mask = i != j
    while not np.all(mask):
        n_bad = np.count_nonzero(~mask)
        j[~mask] = rng.integers(0, n, size=n_bad, endpoint=False)
        mask = i != j
    return i, j


def compute_pairwise_distances_for_pairs(
    X: np.ndarray,
    pairs: Tuple[np.ndarray, np.ndarray],
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Compute distances only for the requested pairs, avoiding O(n^2) memory.
    Supported metrics: 'euclidean', 'cosine'.
    """
    i, j = pairs
    Xi = X[i]
    Xj = X[j]
    if metric == "euclidean":
        return np.linalg.norm(Xi - Xj, axis=1)
    if metric == "cosine":
        eps = 1e-12
        ni = np.linalg.norm(Xi, axis=1) + eps
        nj = np.linalg.norm(Xj, axis=1) + eps
        sim = np.sum(Xi * Xj, axis=1) / (ni * nj)
        return 1.0 - sim
    raise ValueError(f"Unsupported metric: {metric}")


def histogram_overlap_coefficient(
    x: np.ndarray, y: np.ndarray, bins: int = 256
) -> float:
    """
    Estimate the overlap coefficient (Szymkiewicz–Simpson) between two 1D distributions
    via shared histogram mass (density histograms).
    """
    lo = min(float(np.min(x)), float(np.min(y)))
    hi = max(float(np.max(x)), float(np.max(y)))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return float("nan")
    hist_x, edges = np.histogram(x, bins=bins, range=(lo, hi), density=True)
    hist_y, _ = np.histogram(y, bins=edges, density=True)
    dx = np.diff(edges)
    return float(np.sum(np.minimum(hist_x, hist_y) * dx))


def bootstrap_ci_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator = np.random.default_rng(0),
) -> Tuple[float, float]:
    """Bootstrap CI for difference of means: mean(b) - mean(a)."""
    na, nb = a.shape[0], b.shape[0]
    diffs = np.empty(n_boot, dtype=np.float64)
    for k in range(n_boot):
        sa = a[rng.integers(0, na, size=na, endpoint=False)]
        sb = b[rng.integers(0, nb, size=nb, endpoint=False)]
        diffs[k] = float(np.mean(sb) - np.mean(sa))
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return lo, hi


def subset_silhouette_score(
    X: np.ndarray,
    y: np.ndarray,
    metric: str,
    n_samples: int,
    random_state: int,
) -> float:
    """Compute silhouette score on a manageable subset to avoid O(n^2) memory."""
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    m = min(n_samples, n)
    idx = rng.choice(n, size=m, replace=False)
    return float(silhouette_score(X[idx], y[idx], metric=metric))


def plot_intra_inter_distance_histograms(
    d_intra: np.ndarray,
    d_inter: np.ndarray,
    metric: str,
    outdir: str,
    dataset_tag: str = "covertype",
) -> None:
    """Layered density-style histograms for intra vs inter distances."""
    if plt is None:
        return
    import os
    os.makedirs(outdir, exist_ok=True)
    lo = min(d_intra.min(), d_inter.min())
    hi = max(d_intra.max(), d_inter.max())
    bins = 128
    plt.figure(figsize=(6, 4))
    plt.hist(
        d_intra, bins=bins, range=(lo, hi), density=True, alpha=0.5,
        label="intra-class", edgecolor="none",
    )
    plt.hist(
        d_inter, bins=bins, range=(lo, hi), density=True, alpha=0.5,
        label="inter-class", edgecolor="none",
    )
    plt.xlabel(f"{metric} distance")
    plt.ylabel("density")
    plt.title(f"Distance distributions ({dataset_tag}, {metric})")
    plt.legend()
    for ext in ("png", "pdf"):
        plt.tight_layout()
        plt.savefig(
            os.path.join(outdir, f"{dataset_tag}_intra_inter_{metric}.{ext}"),
            dpi=300,
        )
    plt.close()

