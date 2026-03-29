"""analyze_pca_vs_label.py

PCA-based diagnostic for feature informativeness versus router label.

Why this script exists:
- A low per-feature label correlation can still hide informative joint structure.
- PCA can reveal whether major shared feature-variance directions align with label.

Notes:
- PCA is unsupervised: it maximizes feature variance, not label correlation.
- If PC1 correlation is weak, features may still be predictive non-linearly or via
  combinations not aligned to top-variance directions.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def load_alpha_rows(csv_path):
    """Load alpha/feature rows from analysis CSV."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"Input CSV has no rows: {csv_path}")
    if not fieldnames:
        raise ValueError(f"Input CSV has no header: {csv_path}")

    return rows, fieldnames


def infer_feature_columns(fieldnames, label_column):
    """Infer numeric feature columns from alpha_analysis schema."""
    excluded = {"query_id", "dataset", "alpha", "soft_label", label_column}
    feature_cols = [c for c in fieldnames if c not in excluded]
    if not feature_cols:
        raise ValueError("No feature columns inferred from CSV header.")
    return feature_cols


def rows_to_matrices(rows, feature_cols, label_column):
    """Convert CSV rows to feature matrix X and label vector y."""
    qids = []
    datasets = []
    X = np.zeros((len(rows), len(feature_cols)), dtype=np.float64)
    y = np.zeros(len(rows), dtype=np.float64)

    for i, row in enumerate(rows):
        qids.append(row.get("query_id", f"row_{i}"))
        datasets.append(row.get("dataset", ""))

        try:
            y[i] = float(row[label_column])
        except KeyError as exc:
            raise KeyError(
                f"Label column {label_column!r} is missing. "
                "Use --label-column to select an existing column."
            ) from exc

        for j, col in enumerate(feature_cols):
            try:
                X[i, j] = float(row[col])
            except Exception as exc:
                raise ValueError(f"Non-numeric value in feature column {col!r}, row {i}.") from exc

    return qids, datasets, X, y


def zscore_standardize(X):
    """Z-score features with zero-variance guard."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std <= 1.0e-12] = 1.0
    return (X - mean) / std


def pca_svd(X_std):
    """Compute PCA scores/components using SVD.

    Returns:
      scores: (n_samples, n_components)
      components: (n_components, n_features)
      explained_ratio: (n_components,)
    """
    n_samples = X_std.shape[0]
    if n_samples < 2:
        raise ValueError("Need at least 2 rows for PCA.")

    U, S, Vt = np.linalg.svd(X_std, full_matrices=False)
    scores = U * S

    # Explained variance for centered standardized matrix.
    explained_var = (S ** 2) / max(1, (n_samples - 1))
    total_var = explained_var.sum()
    if total_var <= 1.0e-15:
        explained_ratio = np.zeros_like(explained_var)
    else:
        explained_ratio = explained_var / total_var

    return scores, Vt, explained_ratio


def pearson_corr(x, y):
    """Compute Pearson correlation with zero-variance guard."""
    if np.std(x) <= 1.0e-12 or np.std(y) <= 1.0e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def save_component_correlation_csv(out_csv, explained_ratio, corr_by_pc):
    """Save per-PC explained variance and correlation with label."""
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pc_index", "explained_variance_ratio", "pearson_corr_with_label"])
        for idx, (evr, corr) in enumerate(zip(explained_ratio, corr_by_pc), start=1):
            writer.writerow([idx, f"{float(evr):.12f}", f"{float(corr):.12f}"])


def save_scores_csv(out_csv, qids, datasets, y, scores, n_components):
    """Save query-level PCA scores for reproducibility and inspection."""
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        header = ["query_id", "dataset", "label"] + [f"pc{i}" for i in range(1, n_components + 1)]
        writer.writerow(header)
        for i in range(len(qids)):
            row = [qids[i], datasets[i], f"{float(y[i]):.12f}"]
            row.extend([f"{float(scores[i, j]):.12f}" for j in range(n_components)])
            writer.writerow(row)


def plot_pc1_vs_label(out_png, pc1, y, label_name):
    """Scatter plot of PC1 vs label with least-squares trend line."""
    plt.figure(figsize=(8, 6))
    plt.scatter(pc1, y, s=12, alpha=0.35)

    if np.std(pc1) > 1.0e-12:
        slope, intercept = np.polyfit(pc1, y, deg=1)
        xs = np.linspace(float(pc1.min()), float(pc1.max()), 200)
        ys = slope * xs + intercept
        plt.plot(xs, ys, linewidth=2)

    corr = pearson_corr(pc1, y)
    plt.title(f"PC1 vs {label_name} (r={corr:.3f})")
    plt.xlabel("PC1 score")
    plt.ylabel(label_name)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def plot_explained_variance(out_png, explained_ratio, max_components):
    """Plot explained variance ratio and cumulative curve."""
    n = min(max_components, len(explained_ratio))
    pcs = np.arange(1, n + 1)
    evr = explained_ratio[:n]
    cum = np.cumsum(evr)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(pcs, evr, alpha=0.85)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_xticks(pcs)
    ax1.grid(axis="y", alpha=0.2)

    ax2 = ax1.twinx()
    ax2.plot(pcs, cum, marker="o", linewidth=2)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0.0, 1.02)

    plt.title("PCA explained variance")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pc1_pc2_colored_label(out_png, scores, y, label_name):
    """Scatter PC1 vs PC2 with label as color for quick structure check."""
    if scores.shape[1] < 2:
        return

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(scores[:, 0], scores[:, 1], c=y, s=14, alpha=0.5)
    cbar = plt.colorbar(sc)
    cbar.set_label(label_name)
    plt.xlabel("PC1 score")
    plt.ylabel("PC2 score")
    plt.title("PC1 vs PC2 (colored by label)")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze PCA components versus label from alpha_analysis.csv."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional path to alpha_analysis.csv. Defaults to current analysis output folder.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory. Defaults to <analysis>/pca_label_diagnostics.",
    )
    parser.add_argument(
        "--label-column",
        default="soft_label",
        help="Label column name from input CSV (default: soft_label).",
    )
    parser.add_argument(
        "--max-components",
        type=int,
        default=5,
        help="Max number of principal components to report in CSV/plots.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    results_root = get_config_path(cfg, "results_folder", "data/results")
    analysis_dir = os.path.join(results_root, short_model, "analysis")

    input_csv = args.input_csv or os.path.join(analysis_dir, "alpha_analysis.csv")
    out_dir = args.output_dir or os.path.join(analysis_dir, "pca_label_diagnostics")
    ensure_dir(out_dir)

    rows, fieldnames = load_alpha_rows(input_csv)
    feature_cols = infer_feature_columns(fieldnames, args.label_column)

    qids, datasets, X, y = rows_to_matrices(rows, feature_cols, args.label_column)
    X_std = zscore_standardize(X)
    scores, components, explained_ratio = pca_svd(X_std)

    n_components = min(
        max(1, int(args.max_components)),
        scores.shape[1],
    )

    corr_by_pc = [pearson_corr(scores[:, i], y) for i in range(n_components)]

    corr_csv = os.path.join(out_dir, "pc_label_correlation.csv")
    scores_csv = os.path.join(out_dir, "pca_scores.csv")
    pc1_plot = os.path.join(out_dir, "pc1_vs_label.png")
    evr_plot = os.path.join(out_dir, "pca_explained_variance.png")
    pc12_plot = os.path.join(out_dir, "pc1_pc2_colored_by_label.png")

    save_component_correlation_csv(corr_csv, explained_ratio[:n_components], corr_by_pc)
    save_scores_csv(scores_csv, qids, datasets, y, scores, n_components)
    plot_pc1_vs_label(pc1_plot, scores[:, 0], y, args.label_column)
    plot_explained_variance(evr_plot, explained_ratio, n_components)
    plot_pc1_pc2_colored_label(pc12_plot, scores[:, :max(2, n_components)], y, args.label_column)

    # Optional: loadings for interpretability.
    loadings_csv = os.path.join(out_dir, "pc_loadings.csv")
    with open(loadings_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name"] + [f"pc{i}" for i in range(1, n_components + 1)])
        for feat_idx, feat_name in enumerate(feature_cols):
            writer.writerow(
                [feat_name] + [f"{float(components[pc_idx, feat_idx]):.12f}" for pc_idx in range(n_components)]
            )

    print("PCA label-diagnostic completed.")
    print(f"Input CSV           : {input_csv}")
    print(f"Rows                : {len(rows)}")
    print(f"Features            : {len(feature_cols)}")
    print(f"Label column        : {args.label_column}")
    print(f"Output directory    : {out_dir}")
    print(f"- {corr_csv}")
    print(f"- {scores_csv}")
    print(f"- {loadings_csv}")
    print(f"- {pc1_plot}")
    print(f"- {evr_plot}")
    if n_components >= 2:
        print(f"- {pc12_plot}")


if __name__ == "__main__":
    main()
