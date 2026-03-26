"""optimize_bm25.py

Grid-search globally optimal BM25 parameters over all configured datasets.

For each (k1, b, use_stemming) combination:
  1) Build/reuse BM25 sparse artifacts with cache signatures.
  2) Run/reuse BM25 retrieval results per dataset.
  3) Evaluate NDCG@k per dataset.
  4) Compute macro-average NDCG@k across datasets.

Outputs:
  - data/results/bm25_optimization.csv
  - data/results/bm25_optimization_macro.csv
"""

import argparse
import csv
import itertools
import json
import os
import random
import sys

import numpy as np
from tqdm import tqdm

# Ensure project root is importable regardless of launch location.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.preprocess import (
    build_bm25_and_word_freq_index,
    build_doc_freq_index,
    preprocess_corpus,
    preprocess_queries,
)
from src.retrieve_and_evaluate import calculate_dataset_ndcg_at_k, run_bm25_retrieval
from src.utils import (
    download_beir_dataset,
    ensure_dir,
    file_exists,
    get_config_path,
    load_beir_dataset,
    load_config,
    load_pickle,
    load_qrels,
    load_queries,
    model_short_name,
    save_pickle,
    write_corpus_jsonl,
    write_qrels_tsv,
    write_queries_jsonl,
)


def set_seed(seed):
    """Set deterministic seeds for repeatable optimization runs."""
    random.seed(seed)
    np.random.seed(seed)


def ensure_base_dataset_exports(dataset_name, cfg):
    """Ensure corpus/queries/qrels exports exist for a dataset and return ds_dir."""
    datasets_folder = get_config_path(cfg, "datasets_folder", "data/datasets")
    processed_folder = get_config_path(cfg, "processed_folder", "data/processed_data")

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    ds_dir = os.path.join(processed_folder, short_model, dataset_name)
    ensure_dir(ds_dir)

    corpus_jsonl = os.path.join(ds_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")

    if file_exists(corpus_jsonl) and file_exists(queries_jsonl) and file_exists(qrels_tsv):
        return ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv

    dataset_path = download_beir_dataset(dataset_name, datasets_folder)
    if dataset_path is None:
        raise RuntimeError(f"Dataset download/verification failed: {dataset_name}")

    corpus, queries, qrels, split = load_beir_dataset(dataset_path)
    if corpus is None:
        raise RuntimeError(f"Failed to load BEIR dataset: {dataset_name}")

    print(
        f"  Exporting raw artifacts for {dataset_name} "
        f"(split={split}, corpus={len(corpus):,}, queries={len(queries):,})"
    )
    write_corpus_jsonl(corpus, corpus_jsonl)
    write_queries_jsonl(queries, queries_jsonl)
    write_qrels_tsv(qrels, qrels_tsv)
    return ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv


def ensure_sparse_artifacts(dataset_name, cfg, k1, b, use_stemming):
    """Ensure tokenized data, frequency indexes, and BM25 index exist for one config."""
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    ds_dir, corpus_jsonl, queries_jsonl, qrels_tsv = ensure_base_dataset_exports(dataset_name, cfg)

    top_k = int(cfg.get("benchmark", {}).get("top_k", 100))
    sparse_paths = u.bm25_artifact_paths(ds_dir, k1, b, use_stemming, top_k=top_k)

    if not file_exists(sparse_paths["tokenized_corpus_jsonl"]):
        preprocess_corpus(
            corpus_jsonl=corpus_jsonl,
            output_jsonl=sparse_paths["tokenized_corpus_jsonl"],
            stemmer_lang=stemmer_lang,
            use_stemming=use_stemming,
        )

    preprocess_queries(
        queries_jsonl=queries_jsonl,
        tokenized_queries_jsonl=sparse_paths["tokenized_queries_jsonl"],
        query_tokens_pkl=sparse_paths["query_tokens_pkl"],
        stemmer_lang=stemmer_lang,
        use_stemming=use_stemming,
    )

    needs_freq = not file_exists(sparse_paths["word_freq_pkl"]) or not file_exists(
        sparse_paths["doc_freq_pkl"]
    )
    needs_bm25 = not file_exists(sparse_paths["bm25_pkl"]) or not file_exists(
        sparse_paths["bm25_docids_pkl"]
    )

    bm25 = None
    bm25_doc_ids = None
    if needs_freq or needs_bm25:
        bm25, bm25_doc_ids, global_counts, total_corpus_tokens = build_bm25_and_word_freq_index(
            sparse_paths["tokenized_corpus_jsonl"],
            k1=k1,
            b=b,
        )

    if needs_freq:
        doc_freq, total_docs = build_doc_freq_index(sparse_paths["tokenized_corpus_jsonl"])
        save_pickle((global_counts, total_corpus_tokens), sparse_paths["word_freq_pkl"])
        save_pickle((doc_freq, total_docs), sparse_paths["doc_freq_pkl"])

    if needs_bm25:
        save_pickle(bm25, sparse_paths["bm25_pkl"])
        save_pickle(bm25_doc_ids, sparse_paths["bm25_docids_pkl"])

    return {
        "paths": sparse_paths,
        "queries_jsonl": queries_jsonl,
        "qrels_tsv": qrels_tsv,
        "stemmer_lang": stemmer_lang,
    }


def run_or_load_bm25_results(dataset_name, cfg, k1, b, use_stemming):
    """Load cached BM25 results or run retrieval for one dataset/config."""
    benchmark_cfg = cfg.get("benchmark", {})
    top_k = int(benchmark_cfg.get("top_k", 100))

    ds_inputs = ensure_sparse_artifacts(dataset_name, cfg, k1, b, use_stemming)
    sparse_paths = ds_inputs["paths"]

    if file_exists(sparse_paths["bm25_results_pkl"]):
        bm25_results = load_pickle(sparse_paths["bm25_results_pkl"])
    else:
        queries = load_queries(ds_inputs["queries_jsonl"])
        bm25 = load_pickle(sparse_paths["bm25_pkl"])
        bm25_doc_ids = load_pickle(sparse_paths["bm25_docids_pkl"])
        bm25_results = run_bm25_retrieval(
            bm25=bm25,
            doc_ids=bm25_doc_ids,
            queries=queries,
            stemmer_lang=ds_inputs["stemmer_lang"],
            top_k=top_k,
            use_stemming=use_stemming,
        )
        save_pickle(bm25_results, sparse_paths["bm25_results_pkl"])

    qrels = load_qrels(ds_inputs["qrels_tsv"])
    return bm25_results, qrels, sparse_paths["bm25_signature"]


def optimize_bm25(cfg):
    """Run global BM25 grid-search over configured datasets and return results."""
    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured. Add entries under 'datasets' in config.yaml.")

    benchmark_cfg = cfg.get("benchmark", {})
    ndcg_k = int(benchmark_cfg.get("ndcg_k", 10))

    grid_cfg = cfg.get("bm25_optimization", {}) or {}
    k1_values = [float(v) for v in grid_cfg.get("k1_values", [0.8, 1.2, 1.6, 2.0])]
    b_values = [float(v) for v in grid_cfg.get("b_values", [0.0, 0.25, 0.5, 0.75, 1.0])]
    stem_values = [bool(v) for v in grid_cfg.get("use_stemming_values", [True, False])]

    if not k1_values or not b_values or not stem_values:
        raise ValueError("bm25_optimization grid values must be non-empty.")

    all_combos = list(itertools.product(k1_values, b_values, stem_values))
    per_dataset_rows = []
    macro_rows = []

    combo_progress = tqdm(all_combos, desc="BM25 grid search", dynamic_ncols=True)
    for k1, b, use_stemming in combo_progress:
        combo_progress.set_postfix(
            k1=f"{k1:.2f}",
            b=f"{b:.2f}",
            stemming=int(use_stemming),
        )

        ds_scores = []
        ds_progress = tqdm(
            datasets,
            desc=f"  datasets(k1={k1:.2f}, b={b:.2f}, stem={int(use_stemming)})",
            dynamic_ncols=True,
            leave=False,
        )
        for dataset_name in ds_progress:
            bm25_results, qrels, signature = run_or_load_bm25_results(
                dataset_name=dataset_name,
                cfg=cfg,
                k1=k1,
                b=b,
                use_stemming=use_stemming,
            )
            bm25_score_map = {
                qid: {doc_id: score for doc_id, score in pairs}
                for qid, pairs in bm25_results.items()
            }
            ndcg = calculate_dataset_ndcg_at_k(bm25_score_map, qrels, ndcg_k)
            ds_scores.append(ndcg)

            per_dataset_rows.append(
                {
                    "k1": k1,
                    "b": b,
                    "use_stemming": use_stemming,
                    "dataset": dataset_name,
                    f"ndcg@{ndcg_k}": ndcg,
                    "bm25_signature": signature,
                }
            )

        macro_ndcg = float(np.mean(ds_scores)) if ds_scores else 0.0
        macro_rows.append(
            {
                "k1": k1,
                "b": b,
                "use_stemming": use_stemming,
                f"macro_ndcg@{ndcg_k}": macro_ndcg,
            }
        )

    macro_rows.sort(key=lambda r: r[f"macro_ndcg@{ndcg_k}"], reverse=True)
    best = macro_rows[0]
    return per_dataset_rows, macro_rows, best


def write_results(cfg, per_dataset_rows, macro_rows, best):
    """Write optimization outputs to CSV files in results folder."""
    results_root = get_config_path(cfg, "results_folder", "data/results")
    ensure_dir(results_root)

    ndcg_k = int(cfg.get("benchmark", {}).get("ndcg_k", 10))
    per_dataset_csv = os.path.join(results_root, "bm25_optimization.csv")
    macro_csv = os.path.join(results_root, "bm25_optimization_macro.csv")
    best_json = os.path.join(results_root, "bm25_optimization_best.json")

    with open(per_dataset_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "b", "use_stemming", "dataset", f"ndcg@{ndcg_k}"])
        for row in per_dataset_rows:
            writer.writerow(
                [
                    f"{row['k1']:.4f}",
                    f"{row['b']:.4f}",
                    int(row["use_stemming"]),
                    row["dataset"],
                    f"{row[f'ndcg@{ndcg_k}']:.6f}",
                ]
            )

    with open(macro_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["k1", "b", "use_stemming", f"macro_ndcg@{ndcg_k}"])
        for row in macro_rows:
            writer.writerow(
                [
                    f"{row['k1']:.4f}",
                    f"{row['b']:.4f}",
                    int(row["use_stemming"]),
                    f"{row[f'macro_ndcg@{ndcg_k}']:.6f}",
                ]
            )

    with open(best_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_bm25": {
                    "k1": float(best["k1"]),
                    "b": float(best["b"]),
                    "use_stemming": bool(best["use_stemming"]),
                },
                f"best_macro_ndcg@{ndcg_k}": float(best[f"macro_ndcg@{ndcg_k}"]),
            },
            f,
            indent=2,
        )

    return per_dataset_csv, macro_csv, best_json


def main():
    parser = argparse.ArgumentParser(
        description="Find globally optimal BM25 parameters across all configured datasets."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    seed = int(cfg.get("supervised_routing", {}).get("seed", 42))
    set_seed(seed)

    print("=" * 72)
    print("BM25 global optimization")
    print(f"Datasets: {', '.join(cfg.get('datasets', []))}")
    print(f"Model cache namespace: {cfg['embeddings']['model_name']}")
    print("=" * 72)

    per_dataset_rows, macro_rows, best = optimize_bm25(cfg)
    per_dataset_csv, macro_csv, best_json = write_results(cfg, per_dataset_rows, macro_rows, best)

    ndcg_k = int(cfg.get("benchmark", {}).get("ndcg_k", 10))
    print("\nBest BM25 configuration:")
    print(
        f"k1={best['k1']:.4f}, b={best['b']:.4f}, "
        f"use_stemming={bool(best['use_stemming'])}"
    )
    print(f"macro_ndcg@{ndcg_k}={best[f'macro_ndcg@{ndcg_k}']:.6f}")

    print("\nSaved optimization outputs:")
    print(f"- {per_dataset_csv}")
    print(f"- {macro_csv}")
    print(f"- {best_json}")


if __name__ == "__main__":
    main()
