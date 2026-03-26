"""compare_negative_delta_features.py

Compare a compact routing model (features with negative LOFO macro deltas)
against the full-feature routing model.

Method:
- Within-dataset only
- Paired train/test splits
- Delta definition: compact_ndcg - full_ndcg
"""

import argparse
import csv
import os
import sys

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
    dataset_seed_offset,
    ensure_retrieval_results_cached,
    evaluate_benchmark_methods_for_qids,
    get_selected_feature_names,
    predict_router_alpha,
    rows_to_matrix_with_features,
    set_global_seed,
    split_rows_train_test,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


def load_negative_feature_rows(macro_delta_csv, threshold):
    """Load LOFO macro delta rows and keep features with delta < threshold."""
    rows = []
    with open(macro_delta_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            feature = str(row["feature_removed"]).strip()
            delta = float(row["macro_delta_ndcg"])
            if delta < threshold:
                rows.append({"feature_removed": feature, "macro_delta_ndcg": delta})
    return rows


def run_model_once(train_rows, test_rows, feature_names, cfg, dataset_name, ds_cache, device, ndcg_k, rrf_k):
    """Train and evaluate one model on one paired split."""
    X_train_raw, y_train, _ = rows_to_matrix_with_features(train_rows, feature_names)
    X_test_raw, _, test_qids = rows_to_matrix_with_features(test_rows, feature_names)

    train_mean, train_std = compute_zscore_stats(X_train_raw)
    X_train = apply_zscore(X_train_raw, train_mean, train_std)
    X_test = apply_zscore(X_test_raw, train_mean, train_std)

    model_bundle = train_router_model(
        X_train,
        y_train,
        cfg,
        device,
        dataset_name=dataset_name,
        optimization_mode="within_dataset",
    )
    alphas = predict_router_alpha(model_bundle, X_test, cfg, device)
    alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}

    metrics = evaluate_benchmark_methods_for_qids(
        bm25_results=ds_cache["bm25_results"],
        dense_results=ds_cache["dense_results"],
        qrels=ds_cache["qrels"],
        ndcg_k=ndcg_k,
        rrf_k=rrf_k,
        query_ids=test_qids,
        alpha_map=alpha_map,
    )
    return float(metrics["dynamic_wrrf_ndcg"])


def save_negative_feature_list(rows, output_csv):
    """Persist the chosen compact feature set with macro deltas."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_kept", "macro_delta_ndcg"])
        for row in sorted(rows, key=lambda r: r["macro_delta_ndcg"]):
            writer.writerow([row["feature_removed"], f"{row['macro_delta_ndcg']:.6f}"])


def save_per_dataset_comparison(rows, output_csv):
    """Save per-dataset full-vs-compact comparison."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "full_dynamic_ndcg",
                "compact_dynamic_ndcg",
                "delta_compact_minus_full",
                "compact_feature_count",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["dataset"],
                    f"{row['full_dynamic_ndcg']:.6f}",
                    f"{row['compact_dynamic_ndcg']:.6f}",
                    f"{row['delta_compact_minus_full']:.6f}",
                    row["compact_feature_count"],
                ]
            )


def save_macro_comparison(row, output_csv):
    """Save macro-average full-vs-compact comparison."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "macro_full_dynamic_ndcg",
                "macro_compact_dynamic_ndcg",
                "macro_delta_compact_minus_full",
                "full_feature_count",
                "compact_feature_count",
            ]
        )
        writer.writerow(
            [
                f"{row['macro_full_dynamic_ndcg']:.6f}",
                f"{row['macro_compact_dynamic_ndcg']:.6f}",
                f"{row['macro_delta_compact_minus_full']:.6f}",
                row["full_feature_count"],
                row["compact_feature_count"],
            ]
        )


def main():
    parser = argparse.ArgumentParser(
        description="Compare full router vs compact router built from negative LOFO macro deltas."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--macro-delta-csv",
        default=None,
        help="Optional path to lofo_macro_delta.csv. Defaults to model results folder.",
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.0,
        help="Keep features with macro_delta_ndcg < threshold (default: 0.0).",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    cfg.setdefault("supervised_routing", {})
    cfg["supervised_routing"]["model_type"] = "xgboost"
    routing_cfg = cfg.get("supervised_routing", {})

    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    within_cfg = cfg.get("within_dataset_evaluation", {})
    train_fraction = float(within_cfg.get("train_fraction", 0.8))
    n_repeats = int(within_cfg.get("n_repeats", 5))
    shuffle = bool(within_cfg.get("shuffle", True))

    if n_repeats <= 0:
        raise ValueError("within_dataset_evaluation.n_repeats must be > 0.")

    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))
    rrf_k = int(cfg["benchmark"].get("rrf", {}).get("k", 60))
    short_model = model_short_name(cfg["embeddings"]["model_name"])

    results_root = get_config_path(cfg, "results_folder", "data/results")
    ablation_dir = os.path.join(results_root, short_model, "ablation")
    macro_delta_csv = args.macro_delta_csv or os.path.join(ablation_dir, "lofo_macro_delta.csv")
    if not os.path.exists(macro_delta_csv):
        raise FileNotFoundError(
            f"Macro delta CSV not found: {macro_delta_csv}. Run LOFO ablation first."
        )

    full_features = get_selected_feature_names()
    negative_rows = load_negative_feature_rows(macro_delta_csv, args.delta_threshold)
    negative_feature_set = {r["feature_removed"] for r in negative_rows}
    compact_features = [f for f in full_features if f in negative_feature_set]

    if not compact_features:
        raise ValueError(
            "No compact features selected. Try increasing --delta-threshold or verify LOFO CSV values."
        )

    print("=" * 72)
    print("Full vs compact (negative-delta) feature comparison")
    print(f"Device                 : {device}")
    print(f"Datasets ({len(datasets)})         : {', '.join(datasets)}")
    print(f"Full feature count     : {len(full_features)}")
    print(f"Compact feature count  : {len(compact_features)}")
    print(f"Delta threshold        : {args.delta_threshold} (keep delta < threshold)")
    print(f"Macro delta CSV        : {macro_delta_csv}")
    print("Delta definition       : compact_ndcg - full_ndcg")
    print("=" * 72)

    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Loading or building query feature cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)
    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/4] Running paired within-dataset comparison ...")
    per_dataset_rows = []

    for dataset_name in datasets:
        ds_rows = list(rows_by_dataset[dataset_name])
        if len(ds_rows) < 2:
            raise ValueError(f"Dataset {dataset_name} has only {len(ds_rows)} rows; need at least 2.")

        ds_offset = dataset_seed_offset(dataset_name)
        full_scores = []
        compact_scores = []

        for repeat_idx in range(n_repeats):
            repeat_seed = seed + repeat_idx + ds_offset
            train_rows, test_rows = split_rows_train_test(
                ds_rows,
                train_fraction=train_fraction,
                repeat_seed=repeat_seed,
                shuffle=shuffle,
            )

            full_ndcg = run_model_once(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_names=full_features,
                cfg=cfg,
                dataset_name=dataset_name,
                ds_cache=dataset_cache_map[dataset_name],
                device=device,
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )
            compact_ndcg = run_model_once(
                train_rows=train_rows,
                test_rows=test_rows,
                feature_names=compact_features,
                cfg=cfg,
                dataset_name=dataset_name,
                ds_cache=dataset_cache_map[dataset_name],
                device=device,
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
            )

            full_scores.append(full_ndcg)
            compact_scores.append(compact_ndcg)

        full_mean = float(np.mean(full_scores))
        compact_mean = float(np.mean(compact_scores))
        delta = float(compact_mean - full_mean)

        per_dataset_rows.append(
            {
                "dataset": dataset_name,
                "full_dynamic_ndcg": full_mean,
                "compact_dynamic_ndcg": compact_mean,
                "delta_compact_minus_full": delta,
                "compact_feature_count": len(compact_features),
            }
        )
        print(
            f"  dataset={dataset_name:>10} | "
            f"full={full_mean:.4f} | compact={compact_mean:.4f} | delta={delta:.4f}"
        )

    macro_full = float(np.mean([r["full_dynamic_ndcg"] for r in per_dataset_rows]))
    macro_compact = float(np.mean([r["compact_dynamic_ndcg"] for r in per_dataset_rows]))
    macro_delta = float(macro_compact - macro_full)
    macro_row = {
        "macro_full_dynamic_ndcg": macro_full,
        "macro_compact_dynamic_ndcg": macro_compact,
        "macro_delta_compact_minus_full": macro_delta,
        "full_feature_count": len(full_features),
        "compact_feature_count": len(compact_features),
    }

    print("\n[4/4] Writing comparison outputs ...")
    out_dir = os.path.join(ablation_dir, "negative_feature_subset_comparison")
    ensure_dir(out_dir)

    kept_features_csv = os.path.join(out_dir, "kept_negative_delta_features.csv")
    per_dataset_csv = os.path.join(out_dir, "full_vs_compact_per_dataset.csv")
    macro_csv = os.path.join(out_dir, "full_vs_compact_macro.csv")

    save_negative_feature_list(negative_rows, kept_features_csv)
    save_per_dataset_comparison(per_dataset_rows, per_dataset_csv)
    save_macro_comparison(macro_row, macro_csv)

    print("\nComparison completed.")
    print(f"Kept features CSV   : {kept_features_csv}")
    print(f"Per-dataset CSV     : {per_dataset_csv}")
    print(f"Macro summary CSV   : {macro_csv}")
    print(
        "Macro result        : "
        f"full={macro_full:.6f}, compact={macro_compact:.6f}, delta={macro_delta:.6f}"
    )


if __name__ == "__main__":
    main()
