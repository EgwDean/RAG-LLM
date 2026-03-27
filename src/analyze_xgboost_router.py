"""analyze_xgboost_router.py

Interpretability and alpha behavior analysis for the XGBoost query router.

Outputs are written to:
  data/results/<model_short_name>/analysis/

This script reuses existing cache/training utilities and does not alter
training/evaluation logic.
"""

import argparse
import csv
import os
import re
import sys
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.retrieve_and_evaluate import (
    apply_zscore,
    build_or_load_query_feature_cache,
    compute_zscore_stats,
    ensure_retrieval_results_cached,
    get_selected_feature_names,
    get_router_model_type,
    get_xgboost_config,
    predict_router_alpha,
    resolve_training_feature_names,
    set_global_seed,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def parse_mode(cli_mode, cfg):
    """Resolve analysis mode from CLI override or benchmark config."""
    if cli_mode:
        mode = cli_mode.strip().lower()
    else:
        mode = str(cfg.get("benchmark", {}).get("evaluation_mode", "within_dataset")).strip().lower()

    if mode not in {"within_dataset", "loodo"}:
        raise ValueError(f"Unsupported mode={mode!r}. Use 'within_dataset' or 'loodo'.")
    return mode


def rows_to_matrix(rows, feature_names):
    """Convert cached feature rows to X, y, qids using explicit feature order."""
    qids = [r["query_id"] for r in rows]
    X = np.asarray(
        [[float(r["features"][name]) for name in feature_names] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(r["soft_label"]) for r in rows], dtype=np.float32)
    return X, y, qids


def sanitize_filename(value):
    """Create a filesystem-safe suffix for plot file names."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value).strip())
    return safe or "feature"


def force_xgboost_cfg(cfg):
    """Return a copy of cfg with router model forced to xgboost."""
    out = deepcopy(cfg)
    out.setdefault("supervised_routing", {})
    out["supervised_routing"]["model_type"] = "xgboost"
    return out


def build_dataset_rows(all_rows, datasets):
    """Group rows by dataset with deterministic query ordering."""
    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])
    return rows_by_dataset


def resolve_feature_names_for_xgboost(cfg):
    """Resolve and validate the exact feature list used by XGBoost training."""
    cfg_xgb = force_xgboost_cfg(cfg)
    feature_names = resolve_training_feature_names(cfg_xgb)
    expected = get_selected_feature_names()

    if feature_names != expected:
        raise ValueError(
            "Feature mismatch detected for XGBoost routing. "
            f"resolve_training_feature_names={feature_names}, expected={expected}"
        )
    return feature_names


def load_shap_or_fail():
    """Import shap with a clear install message if missing."""
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError(
            "shap is required for XGBoost interpretability. Install it with: pip install shap"
        ) from exc
    return shap


def build_model_for_dataset(base_cfg, dataset_name, optimization_mode):
    """Build cfg copy for one dataset using tuned mode-specific xgboost params."""
    cfg_ds = force_xgboost_cfg(base_cfg)
    cfg_ds["xgboost"] = get_xgboost_config(
        base_cfg,
        dataset_name=dataset_name,
        optimization_mode=optimization_mode,
    )
    return cfg_ds


def run_within_dataset_analysis(cfg, datasets, rows_by_dataset, feature_names, device, shap_module):
    """Train one model per dataset on all its queries and analyze all queries."""
    shap_blocks = []
    feature_blocks = []
    shap_input_blocks = []
    alpha_rows = []
    shap_records = []

    for dataset_name in datasets:
        ds_rows = rows_by_dataset[dataset_name]
        if not ds_rows:
            continue

        X_raw, y, qids = rows_to_matrix(ds_rows, feature_names)
        train_mean, train_std = compute_zscore_stats(X_raw)
        X_norm = apply_zscore(X_raw, train_mean, train_std)

        cfg_ds = build_model_for_dataset(cfg, dataset_name, "within_dataset")
        model_bundle = train_router_model(X_norm, y, cfg_ds, device)
        alphas = predict_router_alpha(model_bundle, X_norm, cfg_ds, device)

        if model_bundle["model_type"] != "xgboost":
            raise ValueError("Expected xgboost model bundle in analysis path.")

        explainer = shap_module.TreeExplainer(model_bundle["model"])
        shap_values = explainer.shap_values(X_norm)
        shap_values = np.asarray(shap_values, dtype=np.float32)

        shap_blocks.append(shap_values)
        feature_blocks.append(X_raw)
        shap_input_blocks.append(X_norm)

        for row_idx, qid in enumerate(qids):
            out_row = {
                "query_id": qid,
                "dataset": dataset_name,
                "alpha": float(alphas[row_idx]),
                "soft_label": float(y[row_idx]),
            }
            for feat_idx, feat_name in enumerate(feature_names):
                out_row[feat_name] = float(X_raw[row_idx, feat_idx])
                shap_records.append(
                    {
                        "query_id": qid,
                        "feature_name": feat_name,
                        "shap_value": float(shap_values[row_idx, feat_idx]),
                    }
                )
            alpha_rows.append(out_row)

    if not alpha_rows:
        raise RuntimeError("No rows were analyzed in within_dataset mode.")

    return {
        "shap_values": np.vstack(shap_blocks),
        "feature_matrix": np.vstack(feature_blocks),
        "shap_input_matrix": np.vstack(shap_input_blocks),
        "alpha_rows": alpha_rows,
        "shap_records": shap_records,
    }


def run_loodo_analysis(cfg, datasets, rows_by_dataset, feature_names, device, shap_module):
    """Train LOODO models and analyze each held-out dataset's queries exactly once."""
    if len(datasets) < 2:
        raise ValueError("LOODO analysis requires at least two datasets.")

    shap_blocks = []
    feature_blocks = []
    shap_input_blocks = []
    alpha_rows = []
    shap_records = []

    for heldout in datasets:
        train_rows = [row for ds in datasets if ds != heldout for row in rows_by_dataset[ds]]
        test_rows = list(rows_by_dataset[heldout])

        if not train_rows or not test_rows:
            raise ValueError(f"Fold {heldout}: train/test rows are empty.")

        X_train_raw, y_train, _ = rows_to_matrix(train_rows, feature_names)
        X_test_raw, y_test, test_qids = rows_to_matrix(test_rows, feature_names)

        train_mean, train_std = compute_zscore_stats(X_train_raw)
        X_train = apply_zscore(X_train_raw, train_mean, train_std)
        X_test = apply_zscore(X_test_raw, train_mean, train_std)

        cfg_ds = build_model_for_dataset(cfg, heldout, "loodo")
        model_bundle = train_router_model(X_train, y_train, cfg_ds, device)
        alphas = predict_router_alpha(model_bundle, X_test, cfg_ds, device)

        if model_bundle["model_type"] != "xgboost":
            raise ValueError("Expected xgboost model bundle in analysis path.")

        explainer = shap_module.TreeExplainer(model_bundle["model"])
        shap_values = explainer.shap_values(X_test)
        shap_values = np.asarray(shap_values, dtype=np.float32)

        shap_blocks.append(shap_values)
        feature_blocks.append(X_test_raw)
        shap_input_blocks.append(X_test)

        for row_idx, qid in enumerate(test_qids):
            out_row = {
                "query_id": qid,
                "dataset": heldout,
                "alpha": float(alphas[row_idx]),
                "soft_label": float(y_test[row_idx]),
            }
            for feat_idx, feat_name in enumerate(feature_names):
                out_row[feat_name] = float(X_test_raw[row_idx, feat_idx])
                shap_records.append(
                    {
                        "query_id": qid,
                        "feature_name": feat_name,
                        "shap_value": float(shap_values[row_idx, feat_idx]),
                    }
                )
            alpha_rows.append(out_row)

    return {
        "shap_values": np.vstack(shap_blocks),
        "feature_matrix": np.vstack(feature_blocks),
        "shap_input_matrix": np.vstack(shap_input_blocks),
        "alpha_rows": alpha_rows,
        "shap_records": shap_records,
    }


def write_shap_values_csv(records, output_csv):
    """Write long-form SHAP values for all query-feature pairs."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "feature_name", "shap_value"])
        for item in records:
            writer.writerow([item["query_id"], item["feature_name"], f"{item['shap_value']:.12f}"])


def write_alpha_analysis_csv(alpha_rows, feature_names, output_csv):
    """Write alpha/label-per-query table including all selected feature values."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "dataset", "alpha", "soft_label", *feature_names])
        for row in alpha_rows:
            writer.writerow(
                [
                    row["query_id"],
                    row["dataset"],
                    f"{row['alpha']:.12f}",
                    f"{row['soft_label']:.12f}",
                    *[f"{row[name]:.12f}" for name in feature_names],
                ]
            )


def write_feature_alpha_correlation_csv(alpha_rows, feature_names, output_csv):
    """Compute and write Pearson correlation between alpha and each feature."""
    alphas = np.asarray([float(r["alpha"]) for r in alpha_rows], dtype=np.float64)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "pearson_correlation"])

        for feat in feature_names:
            x = np.asarray([float(r[feat]) for r in alpha_rows], dtype=np.float64)
            if np.std(alphas) <= 1.0e-12 or np.std(x) <= 1.0e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, alphas)[0, 1])
            writer.writerow([feat, f"{corr:.12f}"])


def write_feature_label_correlation_csv(alpha_rows, feature_names, output_csv):
    """Compute and write Pearson correlation between soft label and each feature."""
    labels = np.asarray([float(r["soft_label"]) for r in alpha_rows], dtype=np.float64)

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_name", "pearson_correlation"])

        for feat in feature_names:
            x = np.asarray([float(r[feat]) for r in alpha_rows], dtype=np.float64)
            if np.std(labels) <= 1.0e-12 or np.std(x) <= 1.0e-12:
                corr = float("nan")
            else:
                corr = float(np.corrcoef(x, labels)[0, 1])
            writer.writerow([feat, f"{corr:.12f}"])


def save_shap_ranking_plot(shap_module, shap_values, X_for_plots, feature_names, out_dir):
    """Save a single SHAP ranking plot with all features (mean |SHAP| bar ranking)."""
    shap_module.summary_plot(
        shap_values,
        X_for_plots,
        feature_names=feature_names,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "shap_feature_ranking.png"), dpi=200, bbox_inches="tight")
    plt.close()


def save_feature_vs_label_plots(alpha_rows, feature_names, out_dir):
    """Save one grid figure with all features plotted against soft label."""
    labels = np.asarray([float(r["soft_label"]) for r in alpha_rows], dtype=np.float64)
    n_features = len(feature_names)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 3.4 * n_rows), squeeze=False)
    flat_axes = axes.ravel()

    for idx, feat in enumerate(feature_names):
        ax = flat_axes[idx]
        x = np.asarray([float(r[feat]) for r in alpha_rows], dtype=np.float64)
        if np.std(labels) <= 1.0e-12 or np.std(x) <= 1.0e-12:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x, labels)[0, 1])

        ax.scatter(x, labels, s=10, alpha=0.35)
        ax.set_title(f"{feat}\nr={corr:.3f}")
        ax.set_xlabel(feat)
        ax.set_ylabel("soft_label")
        ax.grid(alpha=0.2)

    for idx in range(n_features, len(flat_axes)):
        flat_axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "all_features_vs_soft_label.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_alpha_plots(alpha_rows, feature_names, out_dir):
    """Save alpha distribution, per-dataset boxplot, and alpha-vs-key-features scatter plots."""
    alphas = np.asarray([float(r["alpha"]) for r in alpha_rows], dtype=np.float64)
    datasets = sorted(set(r["dataset"] for r in alpha_rows))

    plt.figure(figsize=(8, 5))
    plt.hist(alphas, bins=40)
    plt.title("Predicted alpha distribution")
    plt.xlabel("alpha")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close()

    grouped = []
    for ds in datasets:
        arr = np.asarray([float(r["alpha"]) for r in alpha_rows if r["dataset"] == ds], dtype=np.float64)
        grouped.append(arr)

    plt.figure(figsize=(10, 5))
    plt.boxplot(grouped, labels=datasets, showfliers=False)
    plt.title("Alpha per dataset")
    plt.xlabel("dataset")
    plt.ylabel("alpha")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "alpha_per_dataset.png"), dpi=200, bbox_inches="tight")
    plt.close()

    key_features = ["agreement", "confidence_gap", "query_length"]
    for feat in key_features:
        if feat not in feature_names:
            continue
        x = np.asarray([float(r[feat]) for r in alpha_rows], dtype=np.float64)
        plt.figure(figsize=(8, 5))
        plt.scatter(x, alphas, s=14, alpha=0.5)
        plt.title(f"Alpha vs {feat}")
        plt.xlabel(feat)
        plt.ylabel("alpha")
        plt.tight_layout()
        out_name = f"alpha_vs_feature_{sanitize_filename(feat)}.png"
        plt.savefig(os.path.join(out_dir, out_name), dpi=200, bbox_inches="tight")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run SHAP and alpha behavior analysis for the XGBoost query router."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--mode",
        choices=["within_dataset", "loodo"],
        default=None,
        help="Analysis mode override. Defaults to benchmark.evaluation_mode.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()
    mode = parse_mode(args.mode, cfg)

    print("Running SHAP analysis for XGBoost router")

    cfg = force_xgboost_cfg(cfg)
    if get_router_model_type(cfg) != "xgboost":
        raise ValueError("Analysis script requires supervised_routing.model_type='xgboost'.")

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    routing_cfg = cfg.get("supervised_routing", {})
    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_dir = os.path.join(results_root, short_model, "analysis")
    ensure_dir(out_dir)

    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Model: {cfg['embeddings']['model_name']}")

    feature_names = resolve_feature_names_for_xgboost(cfg)
    print(f"Feature list used (canonical thesis set): {feature_names}")

    print("\n[1/3] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/3] Building or loading per-query feature/label cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)
    rows_by_dataset = build_dataset_rows(all_rows, datasets)

    shap_module = load_shap_or_fail()

    print("\n[3/3] Training analysis models and computing SHAP/alpha outputs ...")
    if mode == "within_dataset":
        output = run_within_dataset_analysis(
            cfg=cfg,
            datasets=datasets,
            rows_by_dataset=rows_by_dataset,
            feature_names=feature_names,
            device=device,
            shap_module=shap_module,
        )
    else:
        output = run_loodo_analysis(
            cfg=cfg,
            datasets=datasets,
            rows_by_dataset=rows_by_dataset,
            feature_names=feature_names,
            device=device,
            shap_module=shap_module,
        )

    n_queries = len(output["alpha_rows"])
    print(f"Number of queries analyzed: {n_queries}")

    shap_csv = os.path.join(out_dir, "shap_values.csv")
    alpha_csv = os.path.join(out_dir, "alpha_analysis.csv")
    alpha_corr_csv = os.path.join(out_dir, "feature_alpha_correlation.csv")
    label_corr_csv = os.path.join(out_dir, "feature_label_correlation.csv")

    write_shap_values_csv(output["shap_records"], shap_csv)
    write_alpha_analysis_csv(output["alpha_rows"], feature_names, alpha_csv)
    write_feature_alpha_correlation_csv(output["alpha_rows"], feature_names, alpha_corr_csv)
    write_feature_label_correlation_csv(output["alpha_rows"], feature_names, label_corr_csv)

    save_shap_ranking_plot(
        shap_module,
        output["shap_values"],
        output["shap_input_matrix"],
        feature_names,
        out_dir,
    )
    save_alpha_plots(output["alpha_rows"], feature_names, out_dir)
    save_feature_vs_label_plots(output["alpha_rows"], feature_names, out_dir)

    print("\nAnalysis completed.")
    print(f"Output directory: {out_dir}")
    print(f"- {shap_csv}")
    print(f"- {alpha_csv}")
    print(f"- {alpha_corr_csv}")
    print(f"- {label_corr_csv}")
    print("- shap_feature_ranking.png")
    print("- all_features_vs_soft_label.png")
    print("- alpha_distribution.png")
    print("- alpha_per_dataset.png")
    print("- alpha_vs_feature_*.png")


if __name__ == "__main__":
    main()
