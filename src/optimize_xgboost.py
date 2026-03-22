"""optimize_xgboost.py

Search XGBoost router hyperparameters against the downstream IR objective:
maximize dynamic_wrrf_ndcg.

Supported optimization modes:
  - within_dataset: optimize separately for each dataset using repeated splits.
  - loodo: optimize separately for each held-out dataset/fold.

The script reuses cached retrieval artifacts, cached query-level features,
shared model training/prediction helpers, and the same benchmark evaluation
logic used in retrieve_and_evaluate.py.
"""

import argparse
import csv
import itertools
import json
import os
import sys
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np
import torch
import yaml
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
    predict_router_alpha,
    set_global_seed,
    split_rows_train_test,
    train_router_model,
)
from src.utils import ensure_dir, get_config_path, load_config, model_short_name


PARAM_ORDER = [
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "reg_lambda",
    "reg_alpha",
]


def rows_to_matrix(rows, feature_names):
    """Convert cached feature rows to X, y, qids using explicit feature order."""
    qids = [r["query_id"] for r in rows]
    X = np.asarray(
        [[float(r["features"][name]) for name in feature_names] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(r["soft_label"]) for r in rows], dtype=np.float32)
    return X, y, qids


def parse_mode(cli_mode, cfg):
    """Resolve optimization mode from CLI override or config."""
    if cli_mode:
        mode = cli_mode.strip().lower()
    else:
        mode = str(cfg.get("xgboost_optimization", {}).get("mode", "within_dataset")).strip().lower()

    if mode not in {"within_dataset", "loodo"}:
        raise ValueError(
            f"Unsupported optimization mode={mode!r}. Use 'within_dataset' or 'loodo'."
        )
    return mode


def get_optimization_cfg(cfg):
    """Return xgboost_optimization config with defaults and validation."""
    opt_cfg = cfg.get("xgboost_optimization", {}) or {}
    search_type = str(opt_cfg.get("search_type", "grid")).strip().lower()
    if search_type != "grid":
        raise ValueError(
            f"Unsupported xgboost_optimization.search_type={search_type!r}. "
            "Only 'grid' is currently supported."
        )

    max_configs = opt_cfg.get("max_configs", None)
    if max_configs is not None:
        max_configs = int(max_configs)
        if max_configs <= 0:
            raise ValueError("xgboost_optimization.max_configs must be > 0 when set.")

    grid = opt_cfg.get("param_grid", {}) or {}
    missing = [k for k in PARAM_ORDER if k not in grid]
    if missing:
        raise ValueError(
            "xgboost_optimization.param_grid is missing required keys: "
            + ", ".join(missing)
        )

    normalized_grid = {}
    for key in PARAM_ORDER:
        values = list(grid.get(key, []))
        if not values:
            raise ValueError(f"xgboost_optimization.param_grid.{key} must be non-empty.")
        normalized_grid[key] = values

    return {
        "search_type": search_type,
        "max_configs": max_configs,
        "param_grid": normalized_grid,
    }


def generate_grid_candidates(param_grid, max_configs):
    """Generate deterministic candidate dictionaries from param_grid."""
    value_lists = [param_grid[k] for k in PARAM_ORDER]
    combos = list(itertools.product(*value_lists))

    candidates = []
    for idx, combo in enumerate(combos, start=1):
        params = {k: combo[pos] for pos, k in enumerate(PARAM_ORDER)}
        candidates.append({"config_index": idx, "params": params})

    if max_configs is not None:
        candidates = candidates[:max_configs]

    return candidates, len(combos)


def build_candidate_cfg(base_cfg, params):
    """Create one evaluation config forcing xgboost model_type and candidate params."""
    cfg = deepcopy(base_cfg)
    cfg.setdefault("supervised_routing", {})
    cfg["supervised_routing"]["model_type"] = "xgboost"

    merged_xgb = dict(cfg.get("xgboost", {}) or {})
    merged_xgb.update(params)
    cfg["xgboost"] = merged_xgb
    return cfg


def evaluate_within_dataset_candidate(
    dataset_name,
    ds_rows,
    ds_cache,
    cfg,
    device,
    ndcg_k,
    rrf_k,
    feature_names,
):
    """Evaluate one candidate in within-dataset mode for a single dataset."""
    routing_cfg = cfg.get("supervised_routing", {})
    base_seed = int(routing_cfg.get("seed", 42))

    within_cfg = cfg.get("within_dataset_evaluation", {})
    train_fraction = float(within_cfg.get("train_fraction", 0.8))
    n_repeats = int(within_cfg.get("n_repeats", 5))
    shuffle = bool(within_cfg.get("shuffle", True))

    if n_repeats <= 0:
        raise ValueError("within_dataset_evaluation.n_repeats must be > 0.")

    ds_seed_offset = dataset_seed_offset(dataset_name)
    repeat_scores = []

    for repeat_idx in range(n_repeats):
        repeat_seed = base_seed + repeat_idx + ds_seed_offset
        train_rows, test_rows = split_rows_train_test(
            ds_rows,
            train_fraction=train_fraction,
            repeat_seed=repeat_seed,
            shuffle=shuffle,
        )

        X_train_raw, y_train, _ = rows_to_matrix(train_rows, feature_names)
        X_test_raw, _, test_qids = rows_to_matrix(test_rows, feature_names)

        train_mean, train_std = compute_zscore_stats(X_train_raw)
        X_train = apply_zscore(X_train_raw, train_mean, train_std)
        X_test = apply_zscore(X_test_raw, train_mean, train_std)

        model_bundle = train_router_model(X_train, y_train, cfg, device)
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
        repeat_scores.append(float(metrics["dynamic_wrrf_ndcg"]))

    return float(np.mean(repeat_scores))


def evaluate_loodo_candidate(
    heldout,
    rows_by_dataset,
    dataset_cache_map,
    cfg,
    device,
    ndcg_k,
    rrf_k,
    feature_names,
):
    """Evaluate one candidate in LOODO mode for one held-out dataset."""
    train_rows = [row for ds, rows in rows_by_dataset.items() if ds != heldout for row in rows]
    test_rows = list(rows_by_dataset[heldout])

    if not train_rows or not test_rows:
        raise ValueError(f"Fold {heldout}: train/test rows are empty.")

    X_train_raw, y_train, _ = rows_to_matrix(train_rows, feature_names)
    X_test_raw, _, test_qids = rows_to_matrix(test_rows, feature_names)

    train_mean, train_std = compute_zscore_stats(X_train_raw)
    X_train = apply_zscore(X_train_raw, train_mean, train_std)
    X_test = apply_zscore(X_test_raw, train_mean, train_std)

    model_bundle = train_router_model(X_train, y_train, cfg, device)
    alphas = predict_router_alpha(model_bundle, X_test, cfg, device)
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
    return float(metrics["dynamic_wrrf_ndcg"])


def write_trial_results_csv(rows, output_csv):
    """Write all evaluated configurations and scores to CSV."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "optimization_mode",
                "target_dataset",
                "config_index",
                "n_estimators",
                "max_depth",
                "learning_rate",
                "subsample",
                "colsample_bytree",
                "min_child_weight",
                "reg_lambda",
                "reg_alpha",
                "dynamic_wrrf_ndcg",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row["optimization_mode"],
                    row["target_dataset"],
                    row["config_index"],
                    row["n_estimators"],
                    row["max_depth"],
                    row["learning_rate"],
                    row["subsample"],
                    row["colsample_bytree"],
                    row["min_child_weight"],
                    row["reg_lambda"],
                    row["reg_alpha"],
                    f"{row['dynamic_wrrf_ndcg']:.6f}",
                ]
            )


def write_summary_csv(best_by_dataset, output_csv):
    """Write compact best-per-dataset summary."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["target_dataset", "best_dynamic_wrrf_ndcg", "best_config_index"])
        for dataset in sorted(best_by_dataset.keys()):
            item = best_by_dataset[dataset]
            writer.writerow(
                [
                    dataset,
                    f"{item['best_dynamic_wrrf_ndcg']:.6f}",
                    item["best_config_index"],
                ]
            )


def build_best_payload(mode, best_by_dataset):
    """Build JSON/YAML-friendly best-params payload with config paste structure."""
    branch = {dataset: dict(item["params"]) for dataset, item in best_by_dataset.items()}

    return {
        "optimization_mode": mode,
        "xgboost_per_dataset": {
            "within_dataset": branch if mode == "within_dataset" else {},
            "loodo": branch if mode == "loodo" else {},
        },
        "best_results": {
            dataset: {
                "best_dynamic_wrrf_ndcg": float(item["best_dynamic_wrrf_ndcg"]),
                "best_config_index": int(item["best_config_index"]),
            }
            for dataset, item in best_by_dataset.items()
        },
    }


def write_best_outputs(mode, best_by_dataset, json_path, yaml_path):
    """Write best-per-dataset outputs in JSON and YAML."""
    payload = build_best_payload(mode, best_by_dataset)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def write_search_metadata(output_path, mode, total_grid_configs, evaluated_configs, cfg, datasets):
    """Write metadata for reproducibility and traceability."""
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "optimization_mode": mode,
        "total_configs": int(total_grid_configs),
        "evaluated_configs": int(evaluated_configs),
        "datasets": list(datasets),
        "seed": int(cfg.get("supervised_routing", {}).get("seed", 42)),
        "model_name": cfg["embeddings"]["model_name"],
        "search_type": str(cfg.get("xgboost_optimization", {}).get("search_type", "grid")),
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def optimize_xgboost(cfg, mode):
    """Run XGBoost optimization for selected mode and return result rows."""
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    if mode == "loodo" and len(datasets) < 2:
        raise ValueError("LOODO optimization requires at least two configured datasets.")

    routing_cfg = cfg.get("supervised_routing", {})
    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    ndcg_k = int(cfg.get("benchmark", {}).get("ndcg_k", 10))
    rrf_k = int(cfg.get("benchmark", {}).get("rrf", {}).get("k", 60))

    opt_cfg = get_optimization_cfg(cfg)
    candidates, total_grid_configs = generate_grid_candidates(
        opt_cfg["param_grid"],
        opt_cfg["max_configs"],
    )

    short_model = model_short_name(cfg["embeddings"]["model_name"])

    print("=" * 72)
    print("XGBoost router hyperparameter optimization")
    print(f"Mode                 : {mode}")
    print(f"Device               : {device}")
    print(f"Model                : {cfg['embeddings']['model_name']}")
    print(f"Datasets ({len(datasets)})       : {', '.join(datasets)}")
    print(f"Search type          : {opt_cfg['search_type']}")
    print(f"Total grid configs   : {total_grid_configs}")
    print(f"Configs to evaluate  : {len(candidates)}")
    print("=" * 72)

    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        print(f"\n{'=' * 68}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 68}")
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Building or loading per-query feature/label cache ...")
    all_rows, _, _ = build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model)

    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    feature_names = list(FEATURE_NAMES)

    print("\n[3/4] Running XGBoost search ...")
    trial_rows = []
    best_by_dataset = {}

    target_iter = datasets
    for target_dataset in target_iter:
        print(f"\n{'-' * 72}")
        print(f"Target dataset: {target_dataset}")
        print(f"{'-' * 72}")

        best_score = -1.0
        best_params = None
        best_idx = None

        for rank_idx, candidate in enumerate(
            tqdm(candidates, desc=f"Configs for {target_dataset}", dynamic_ncols=True),
            start=1,
        ):
            config_index = int(candidate["config_index"])
            params = dict(candidate["params"])
            eval_cfg = build_candidate_cfg(cfg, params)

            if mode == "within_dataset":
                score = evaluate_within_dataset_candidate(
                    dataset_name=target_dataset,
                    ds_rows=rows_by_dataset[target_dataset],
                    ds_cache=dataset_cache_map[target_dataset],
                    cfg=eval_cfg,
                    device=device,
                    ndcg_k=ndcg_k,
                    rrf_k=rrf_k,
                    feature_names=feature_names,
                )
            else:
                score = evaluate_loodo_candidate(
                    heldout=target_dataset,
                    rows_by_dataset=rows_by_dataset,
                    dataset_cache_map=dataset_cache_map,
                    cfg=eval_cfg,
                    device=device,
                    ndcg_k=ndcg_k,
                    rrf_k=rrf_k,
                    feature_names=feature_names,
                )

            row = {
                "optimization_mode": mode,
                "target_dataset": target_dataset,
                "config_index": config_index,
                "dynamic_wrrf_ndcg": float(score),
                **params,
            }
            trial_rows.append(row)

            print(
                f"  [{rank_idx}/{len(candidates)}] "
                f"config_index={config_index} "
                f"params={params} "
                f"score={score:.6f} "
                f"best_so_far={max(best_score, score):.6f}"
            )

            if score > best_score:
                best_score = float(score)
                best_params = params
                best_idx = config_index

        if best_params is None:
            raise RuntimeError(f"No candidate evaluated for dataset {target_dataset}.")

        best_by_dataset[target_dataset] = {
            "best_dynamic_wrrf_ndcg": float(best_score),
            "best_config_index": int(best_idx),
            "params": best_params,
        }

        print(
            "Best for "
            f"{target_dataset}: score={best_score:.6f}, "
            f"config_index={best_idx}, params={best_params}"
        )

    print("\n[4/4] Writing optimization outputs ...")
    results_root = get_config_path(cfg, "results_folder", "data/results")
    out_dir = os.path.join(results_root, short_model, "xgboost_optimization", mode)
    ensure_dir(out_dir)

    trial_csv = os.path.join(out_dir, "trial_results.csv")
    summary_csv = os.path.join(out_dir, "summary.csv")
    best_json = os.path.join(out_dir, "best_params_per_dataset.json")
    best_yaml = os.path.join(out_dir, "best_params_per_dataset.yaml")
    metadata_json = os.path.join(out_dir, "search_metadata.json")

    write_trial_results_csv(trial_rows, trial_csv)
    write_summary_csv(best_by_dataset, summary_csv)
    write_best_outputs(mode, best_by_dataset, best_json, best_yaml)
    write_search_metadata(
        metadata_json,
        mode,
        total_grid_configs,
        len(candidates),
        cfg,
        datasets,
    )

    print("\n" + "=" * 72)
    print("XGBoost optimization completed.")
    print(f"Trial results CSV        : {trial_csv}")
    print(f"Summary CSV              : {summary_csv}")
    print(f"Best params JSON         : {best_json}")
    print(f"Best params YAML         : {best_yaml}")
    print(f"Search metadata JSON     : {metadata_json}")
    print("=" * 72)


def main():
    parser = argparse.ArgumentParser(
        description="Optimize XGBoost router hyperparameters using dynamic_wrrf_ndcg objective."
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
        help="Override xgboost_optimization.mode from config.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    mode = parse_mode(args.mode, cfg)
    optimize_xgboost(cfg, mode)


if __name__ == "__main__":
    main()
