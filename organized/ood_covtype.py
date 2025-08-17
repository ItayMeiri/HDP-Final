# ood_covtype.py
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from models import MLP
from utils import (
    print_header,
    calculate_ood_metrics,
    plot_training_summary,
    plot_density_distributions,
    extract_embeddings,
    run_ood_analysis,
)

warnings.filterwarnings("ignore", category=UserWarning)


class Config:
    OOD_CLASSES = [6, 7]
    N_CONTINUOUS_FEATURES = 10
    TEST_SIZE = 0.2
    EMBEDDING_DIM = 32
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-3
    EPOCHS = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    KDE_SUBSET_SIZE = 8000
    PCA_N_COMPONENTS = 6
    FPR_AT_TPR_LEVEL = 0.95
    RANDOM_STATE = 42
    FIGURE_DIR = "figures"


os.makedirs(Config.FIGURE_DIR, exist_ok=True)


def load_and_prepare_data():
    print_header("1. Data Loading and Preprocessing")
    covtype = fetch_covtype()
    X, y = covtype.data, covtype.target
    print(f"Original dataset shape: X={X.shape}, y={y.shape}")
    ood_mask = np.isin(y, Config.OOD_CLASSES)
    X_in, y_in = X[~ood_mask], y[~ood_mask]
    X_ood_hard, y_ood_hard = X[ood_mask], y[ood_mask]
    unique_labels = np.unique(y_in)
    label_map = {label: i for i, label in enumerate(unique_labels)}
    y_in = np.array([label_map[label] for label in y_in])
    print(f"In-distribution classes: {unique_labels}")
    print(f"Out-of-distribution classes: {Config.OOD_CLASSES}")
    print(f"In-distribution samples: {len(X_in)}")
    print(f"Hard OOD samples: {len(X_ood_hard)}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_in, y_in, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y_in
    )
    scaler = StandardScaler()
    X_train[:, :Config.N_CONTINUOUS_FEATURES] = scaler.fit_transform(
        X_train[:, :Config.N_CONTINUOUS_FEATURES]
    )
    X_test[:, :Config.N_CONTINUOUS_FEATURES] = scaler.transform(
        X_test[:, :Config.N_CONTINUOUS_FEATURES]
    )
    X_ood_hard[:, :Config.N_CONTINUOUS_FEATURES] = scaler.transform(
        X_ood_hard[:, :Config.N_CONTINUOUS_FEATURES]
    )

    n_ood_easy = len(X_ood_hard)
    ood_easy_indices = np.random.choice(len(X_train), n_ood_easy, replace=True)
    X_ood_easy = X_train[ood_easy_indices].copy()

    n_features = X_train.shape[1]
    n_features_to_corrupt = int(n_features * 0.5)
    min_val = X_train.min(axis=0)
    max_val = X_train.max(axis=0)
    for i in range(n_ood_easy):
        features_to_corrupt = np.random.choice(n_features, n_features_to_corrupt, replace=False)
        for feat_idx in features_to_corrupt:
            X_ood_easy[i, feat_idx] = np.random.uniform(low=min_val[feat_idx], high=max_val[feat_idx])

    print(f"Created easy OOD (perturbed in-distribution) samples: {X_ood_easy.shape}")

    datasets = {
        "train": TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)),
        "test": TensorDataset(torch.Tensor(X_test), torch.LongTensor(y_test)),
        "ood_hard": TensorDataset(torch.Tensor(X_ood_hard), torch.LongTensor(y_ood_hard)),
        "ood_easy": TensorDataset(torch.Tensor(X_ood_easy), torch.LongTensor(np.ones(len(X_ood_easy)))),
    }
    return datasets, X_train, X_test, X_ood_hard, X_ood_easy, len(unique_labels)


class MLPTrainer:
    """Encapsulates training to preserve original behavior and signatures."""
    def __init__(self, model: torch.nn.Module, num_classes: int):
        self.model = model
        self.num_classes = num_classes

    def train(self, train_loader, test_loader):
        print_header("2. Model Training")
        self.model.to(Config.DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)
        history = {'loss': [], 'acc': []}
        for epoch in range(Config.EPOCHS):
            self.model.train()
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")
            for inputs, labels in pbar:
                inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(Config.DEVICE), labels.to(Config.DEVICE)
                    outputs = self.model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            epoch_loss = running_loss / len(train_loader)
            epoch_acc = correct / total
            history['loss'].append(epoch_loss)
            history['acc'].append(epoch_acc)
            print(f"Epoch {epoch + 1} Summary - Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
        print("Training finished.")
        return history, history['acc'][-1]


def main():
    np.random.seed(Config.RANDOM_STATE)
    torch.manual_seed(Config.RANDOM_STATE)

    datasets, X_train_raw, X_test_raw, X_ood_hard_raw, X_ood_easy_raw, n_classes = load_and_prepare_data()
    train_loader = DataLoader(datasets['train'], batch_size=Config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=Config.BATCH_SIZE, shuffle=False)
    ood_hard_loader = DataLoader(datasets['ood_hard'], batch_size=Config.BATCH_SIZE, shuffle=False)
    ood_easy_loader = DataLoader(datasets['ood_easy'], batch_size=Config.BATCH_SIZE, shuffle=False)

    print_header("1b. Baseline: OOD Detection directly on raw data")
    print("This shows that simple density estimation on the raw feature space is ineffective.")

    def get_random_subset(arr, size):
        if len(arr) > size:
            indices = np.random.choice(len(arr), size, replace=False)
            return arr[indices]
        return arr

    X_train_subset = get_random_subset(X_train_raw, Config.KDE_SUBSET_SIZE)
    X_test_subset = get_random_subset(X_test_raw, Config.KDE_SUBSET_SIZE)
    X_ood_hard_subset = get_random_subset(X_ood_hard_raw, Config.KDE_SUBSET_SIZE)
    X_ood_easy_subset = get_random_subset(X_ood_easy_raw, Config.KDE_SUBSET_SIZE)

    print(f"Using random subset of size up to: {Config.KDE_SUBSET_SIZE} for KDE fitting on raw data.")
    kde_raw = KernelDensity(kernel='gaussian').fit(X_train_subset)
    scores_in_raw = kde_raw.score_samples(X_test_subset)
    scores_hard_raw = kde_raw.score_samples(X_ood_hard_subset)
    scores_easy_raw = kde_raw.score_samples(X_ood_easy_subset)
    print("\n--- Results on Raw Data ---")
    calculate_ood_metrics(scores_in_raw, scores_easy_raw, "Easy OOD (Raw)", Config.FPR_AT_TPR_LEVEL)
    calculate_ood_metrics(scores_in_raw, scores_hard_raw, "Hard OOD (Raw)", Config.FPR_AT_TPR_LEVEL)
    plot_density_distributions(
        scores_in_raw, scores_hard_raw, scores_easy_raw,
        title="Density of OOD Scores on Raw Data",
        filepath=os.path.join(Config.FIGURE_DIR, "ood_scores_raw_data.png")
    )

    input_dim = X_train_raw.shape[1]
    model = MLP(input_dim, n_classes, Config.EMBEDDING_DIM)
    trainer = MLPTrainer(model, n_classes)
    history, final_acc = trainer.train(train_loader, test_loader)
    plot_training_summary(history, final_acc, filepath=os.path.join(Config.FIGURE_DIR, "training_summary.png"))

    print_header("3. Extracting Embeddings from Trained Model")
    embeddings_in = extract_embeddings(model, test_loader)
    embeddings_hard = extract_embeddings(model, ood_hard_loader)
    embeddings_easy = extract_embeddings(model, ood_easy_loader)
    print(f"Shape of In-Distribution Embeddings: {embeddings_in.shape}")
    print(f"Shape of Hard OOD Embeddings: {embeddings_hard.shape}")
    print(f"Shape of Easy OOD Embeddings: {embeddings_easy.shape}")

    def cap(arr):
        return get_random_subset(arr, min(Config.KDE_SUBSET_SIZE, len(arr)))

    embeddings_in = cap(embeddings_in)
    embeddings_hard = cap(embeddings_hard)
    embeddings_easy = cap(embeddings_easy)

    scores_in_full, scores_hard_full, scores_easy_full = run_ood_analysis(
        "4. OOD Analysis on Full Embeddings",
        embeddings_in, embeddings_hard, embeddings_easy
    )
    plot_density_distributions(
        scores_in_full, scores_hard_full, scores_easy_full,
        title="Density of OOD Scores on Full Embeddings",
        filepath=os.path.join(Config.FIGURE_DIR, "ood_scores_full_embeddings.png")
    )

    print_header("5. OOD Analysis on PCA-Reduced Embeddings")
    pca = PCA(n_components=Config.PCA_N_COMPONENTS)
    embeddings_in_pca = pca.fit_transform(embeddings_in)
    embeddings_hard_pca = pca.transform(embeddings_hard)
    embeddings_easy_pca = pca.transform(embeddings_easy)
    print(f"Reduced In-Distribution Embedding Shape: {embeddings_in_pca.shape}")
    print(f"Explained variance by {Config.PCA_N_COMPONENTS} components: {np.sum(pca.explained_variance_ratio_):.4f}")

    scores_in_pca, scores_hard_pca, scores_easy_pca = run_ood_analysis(
        f"OOD Analysis on {Config.PCA_N_COMPONENTS}D PCA-Reduced Embeddings",
        embeddings_in_pca, embeddings_hard_pca, embeddings_easy_pca
    )
    plot_density_distributions(
        scores_in_pca, scores_hard_pca, scores_easy_pca,
        title=f"Density of OOD Scores on {Config.PCA_N_COMPONENTS}D PCA-Reduced Embeddings",
        filepath=os.path.join(Config.FIGURE_DIR, "ood_scores_pca_embeddings.png")
    )


if __name__ == "__main__":
    main()
