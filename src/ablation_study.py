"""ablation_study.py

Structured feature ablation study for supervised query routing.

This script reuses cached retrieval artifacts and cached query-level features,
then evaluates the same logistic router under different feature subsets.
"""

import argparse
import csv
import os
import sys
from copy import deepcopy

import numpy as np
import torch
from tqdm import tqdm

# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.retrieve_and_evaluate import (
    FEATURE_NAMES,
    apply_zscore,
    build_or_load_query_feature_cache,
    compute_zscore_stats,
    dataset_seed_offset,
    ensure_retrieval_results_cached,
    evaluate_benchmark_methods_for_qids,
    predict_alpha,
    set_global_seed,
    split_rows_train_test,
    train_logistic_regression_pytorch,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


FEATURE_GROUPS = {
    "overlap": ["agreement", "overlap_at_3"],
    "confidence": ["dense_confidence", "sparse_confidence", "confidence_gap"],
    "idf": ["average_idf", "max_idf", "idf_std", "rare_term_ratio"],
    "query": ["query_length", "stopword_ratio"],
    "entropy": ["cross_entropy"],
}
ALL_FEATURES = sum(FEATURE_GROUPS.values(), [])


def filter_rows_by_features(rows, selected_features):
    """Return rows with only selected features retained."""
    selected_set = set(selected_features)
    filtered = []
    for row in rows:
        copied = deepcopy(row)
        copied["features"] = {k: v for k, v in row["features"].items() if k in selected_set}
        filtered.append(copied)
    return filtered


def ordered_feature_subset(selected_features):
    """Return selected features ordered as they appear in FEATURE_NAMES."""
    ordered_features = [f for f in FEATURE_NAMES if f in selected_features]
    if not ordered_features:
        raise ValueError("No features selected for ablation.")
    return ordered_features


def rows_to_matrix_subset(rows, selected_features):
    """Convert rows to feature matrix using selected feature subset."""
    ordered_features = ordered_feature_subset(selected_features)
    qids = [r["query_id"] for r in rows]
    X = np.asarray(
        [[float(r["features"][name]) for name in ordered_features] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(r["soft_label"]) for r in rows], dtype=np.float32)
    return X, y, qids, ordered_features


def _validate_feature_setup():
    """Validate that ablation feature groups match active routing features."""
    if len(ALL_FEATURES) != len(set(ALL_FEATURES)):
        raise ValueError("Duplicate feature names detected in FEATURE_GROUPS.")

    expected = set(FEATURE_NAMES)
    actual = set(ALL_FEATURES)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            "Ablation feature definition mismatch with active FEATURE_NAMES. "
            f"Missing={missing}, Extra={extra}"
        )


def _run_loodo_for_features(
    rows_by_dataset,
    dataset_cache_map,
    selected_features,
    cfg,
    datasets,
    device,
    ndcg_k,
    rrf_k,
):
    """Run LOODO evaluation using a selected feature subset."""
    if len(datasets) < 2:
        raise ValueError("LOODO requires at least two configured datasets.")

    fold_metrics = []

    for heldout in datasets:
        train_datasets = [d for d in datasets if d != heldout]
        train_rows = [row for d in train_datasets for row in rows_by_dataset[d]]
        test_rows = list(rows_by_dataset[heldout])

        if not train_rows or not test_rows:
            raise ValueError(f"Fold {heldout}: train/test rows are empty.")

        X_train_raw, y_train, _, _ = rows_to_matrix_subset(train_rows, selected_features)
        X_test_raw, _, test_qids, _ = rows_to_matrix_subset(test_rows, selected_features)

        train_mean, train_std = compute_zscore_stats(X_train_raw)
        X_train = apply_zscore(X_train_raw, train_mean, train_std)

        X_test = apply_zscore(X_test_raw, train_mean, train_std)

        model = train_logistic_regression_pytorch(X_train, y_train, cfg, device)
        alphas = predict_alpha(model, X_test, device)
        alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}

        ds_cache = dataset_cache_map[heldout]
        metrics = evaluate_benchmark_methods_for_qids(
            bm25_results=ds_cache["bm25_results"],
            dense_results=ds_cache["dense_results"],
            qrels=ds_cache["qrels"],
            ndcg_k=ndcg_k,
            rrf_k=rrf_k,
            query_ids=test_qids,
            alpha_map=alpha_map,
        )
        fold_metrics.append(metrics)

    return {
        "dense": float(np.mean([m["dense_only_ndcg"] for m in fold_metrics])),
        "sparse": float(np.mean([m["sparse_only_ndcg"] for m in fold_metrics])),
        "static": float(np.mean([m["static_rrf_ndcg"] for m in fold_metrics])),
        "dynamic": float(np.mean([m["dynamic_wrrf_ndcg"] for m in fold_metrics])),
    }


def _run_within_dataset_for_features(
    rows_by_dataset,
    dataset_cache_map,
    selected_features,
    cfg,
    datasets,
    device,
    ndcg_k,
    rrf_k,
):
    """Run within-dataset repeated split evaluation using selected features."""
    routing_cfg = cfg.get("supervised_routing", {})
    base_seed = int(routing_cfg.get("seed", 42))

    within_cfg = cfg.get("within_dataset_evaluation", {})
    train_fraction = float(within_cfg.get("train_fraction", 0.8))
    n_repeats = int(within_cfg.get("n_repeats", 5))
    shuffle = bool(within_cfg.get("shuffle", True))

    if n_repeats <= 0:
        raise ValueError("within_dataset_evaluation.n_repeats must be > 0.")

    dataset_summaries = []

    for dataset_name in datasets:
        ds_rows = list(rows_by_dataset[dataset_name])
        if len(ds_rows) < 2:
            raise ValueError(
                f"Dataset {dataset_name} has only {len(ds_rows)} query rows; at least 2 required."
            )

        ds_offset = dataset_seed_offset(dataset_name)
        repeat_metrics = []

        for repeat_idx in range(n_repeats):
            repeat_seed = base_seed + repeat_idx + ds_offset
            train_rows, test_rows = split_rows_train_test(
                ds_rows,
                train_fraction=train_fraction,
                repeat_seed=repeat_seed,
                shuffle=shuffle,
            )

            X_train_raw, y_train, _, _ = rows_to_matrix_subset(train_rows, selected_features)
            X_test_raw, _, test_qids, _ = rows_to_matrix_subset(test_rows, selected_features)

            train_mean, train_std = compute_zscore_stats(X_train_raw)
            X_train = apply_zscore(X_train_raw, train_mean, train_std)

            X_test = apply_zscore(X_test_raw, train_mean, train_std)

            model = train_logistic_regression_pytorch(X_train, y_train, cfg, device)
            alphas = predict_alpha(model, X_test, device)
            alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}

            ds_cache = dataset_cache_map[dataset_name]
            metrics = evaluate_benchmark_methods_for_qids(
                bm25_results=ds_cache["bm25_results"],
                dense_results=ds_cache["dense_results"],
                qrels=ds_cache["qrels"],
                ndcg_k=ndcg_k,
                rrf_k=rrf_k,
                query_ids=test_qids,
                alpha_map=alpha_map,
            )
            repeat_metrics.append(metrics)

        dataset_summaries.append(
            {
                "dense": float(np.mean([m["dense_only_ndcg"] for m in repeat_metrics])),
                "sparse": float(np.mean([m["sparse_only_ndcg"] for m in repeat_metrics])),
                "static": float(np.mean([m["static_rrf_ndcg"] for m in repeat_metrics])),
                "dynamic": float(np.mean([m["dynamic_wrrf_ndcg"] for m in repeat_metrics])),
            }
        )

    return {
        "dense": float(np.mean([m["dense"] for m in dataset_summaries])),
        "sparse": float(np.mean([m["sparse"] for m in dataset_summaries])),
        "static": float(np.mean([m["static"] for m in dataset_summaries])),
        "dynamic": float(np.mean([m["dynamic"] for m in dataset_summaries])),
    }


def run_ablation_experiment(
    rows_by_dataset,
    dataset_cache_map,
    selected_features,
    eval_mode,
    cfg,
    datasets,
    device,
    ndcg_k,
    rrf_k,
):
    """Run one ablation experiment and return aggregate benchmark metrics."""
    filtered_by_dataset = {
        ds: filter_rows_by_features(rows_by_dataset[ds], selected_features)
        for ds in datasets
    }

    if eval_mode == "within_dataset":
        return _run_within_dataset_for_features(
            rows_by_dataset=filtered_by_dataset,
            dataset_cache_map=dataset_cache_map,
            selected_features=selected_features,
            cfg=cfg,
            datasets=datasets,
            device=device,
            ndcg_k=ndcg_k,
            rrf_k=rrf_k,
        )
    if eval_mode == "loodo":
        return _run_loodo_for_features(
            rows_by_dataset=filtered_by_dataset,
            dataset_cache_map=dataset_cache_map,
            selected_features=selected_features,
            cfg=cfg,
            datasets=datasets,
            device=device,
            ndcg_k=ndcg_k,
            rrf_k=rrf_k,
        )

    raise ValueError(f"Unsupported eval_mode={eval_mode!r}. Use 'within_dataset' or 'loodo'.")


def build_experiments():
    """Return ordered ablation experiment definitions."""
    experiments = [("all_features", "ALL", ALL_FEATURES)]

    for group_name, group_features in FEATURE_GROUPS.items():
        selected = [f for f in ALL_FEATURES if f not in group_features]
        experiments.append(("leave_one_out", group_name, selected))

    for group_name, group_features in FEATURE_GROUPS.items():
        experiments.append(("single_group", group_name, list(group_features)))

    return experiments


def save_ablation_results(results, output_csv):
    """Save ablation study metrics to CSV."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "experiment",
                "feature_set",
                "evaluation_mode",
                "dense_ndcg",
                "sparse_ndcg",
                "static_rrf_ndcg",
                "dynamic_wrrf_ndcg",
            ]
        )
        for row in results:
            writer.writerow(
                [
                    row["experiment"],
                    row["feature_set"],
                    row["evaluation_mode"],
                    f"{row['dense']:.6f}",
                    f"{row['sparse']:.6f}",
                    f"{row['static']:.6f}",
                    f"{row['dynamic']:.6f}",
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="Run structured feature ablation for supervised routing."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    _validate_feature_setup()

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    routing_cfg = cfg.get("supervised_routing", {})
    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model_name = cfg["embeddings"]["model_name"]
    short_model = model_short_name(model_name)
    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))
    rrf_k = int(cfg["benchmark"].get("rrf", {}).get("k", 60))
    eval_mode = str(cfg.get("benchmark", {}).get("evaluation_mode", "loodo")).strip().lower()
    if eval_mode not in {"within_dataset", "loodo"}:
        raise ValueError(
            f"Unsupported benchmark.evaluation_mode={eval_mode!r}. "
            "Use 'within_dataset' or 'loodo'."
        )

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print(f"Evaluation mode: {eval_mode}")

    print("\n[1/3] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        print(f"\n{'=' * 68}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 68}")
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/3] Building or loading per-query feature/label cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(
        dataset_cache_map,
        cfg,
        short_model,
    )

    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/3] Running ablation experiments ...")
    experiments = build_experiments()
    results = []

    for experiment, group_name, selected_features in tqdm(
        experiments,
        desc="Ablation experiments",
        dynamic_ncols=True,
    ):
        ordered_features = ordered_feature_subset(selected_features)
        print(
            f"\n[ABLATION] Mode={eval_mode} | Experiment={experiment} | Group={group_name}"
        )
        print(f"Using features (ordered): {ordered_features}")
        metrics = run_ablation_experiment(
            rows_by_dataset=rows_by_dataset,
            dataset_cache_map=dataset_cache_map,
            selected_features=selected_features,
            eval_mode=eval_mode,
            cfg=cfg,
            datasets=datasets,
            device=device,
            ndcg_k=ndcg_k,
            rrf_k=rrf_k,
        )
        print(f"Dynamic NDCG@{ndcg_k}: {metrics['dynamic']:.4f}")

        results.append(
            {
                "experiment": experiment,
                "feature_set": group_name,
                "evaluation_mode": eval_mode,
                **metrics,
            }
        )

    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_dir = os.path.join(results_root, short_model, "ablation")
    ensure_dir(out_dir)
    out_csv = os.path.join(out_dir, "ablation_results.csv")
    save_ablation_results(results, out_csv)

    print("\n" + "=" * 72)
    print("Ablation study completed.")
    print(f"Results CSV: {out_csv}")
    print("=" * 72)


if __name__ == "__main__":
    main()
