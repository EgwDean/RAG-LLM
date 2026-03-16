"""retrieval.py -- Dataset-wide retrieval and evaluation for cached artifacts.

Usage:
    python src/retrieval.py
    python src/retrieval.py --config config.yaml

This script assumes preprocessing artifacts already exist under:
    data/processed_data/<model>/<dataset>/

It evaluates each configured dataset with:
  1) BM25 only
  2) Dense only
  3) Static RRF
  4) Dynamic wRRF (JSD, KLD, CE) via a lightweight grid search:
       max_df in [0.5, 0.8]
       sigmoid slope k in [1.0, 2.0, 3.0]

Model-level outputs are written under:
    data/results/<model>/
without creating per-dataset result folders.
"""

import argparse
import csv
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import util as st_util
from tqdm import tqdm

# Ensure project root is on sys.path and cwd is project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    load_config,
    load_pickle,
    save_pickle,
    file_exists,
    load_queries,
    load_qrels,
    model_short_name,
    stem_and_tokenize,
    get_config_path,
    ensure_dir,
)


def sigmoid(x, slope, center=0.0):
    """Sigmoid routing with configurable center bias."""
    return 1.0 / (1.0 + math.exp(-slope * (x - center)))


def _missing_paths(paths):
    """Return the subset of paths that do not exist."""
    return [p for p in paths if not file_exists(p)]


def get_sorted_docs(scores_dict, top_k):
    """Sort score dict by value descending and return top-k pairs."""
    return sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]


def calculate_ndcg_at_k(fused_results, qrels, top_k):
    """Compute average NDCG@k across all queries in qrels."""
    ndcg_scores = []
    for qid, rels in qrels.items():
        if qid not in fused_results:
            ndcg_scores.append(0.0)
            continue

        ranked = get_sorted_docs(fused_results[qid], top_k)
        dcg = 0.0
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            rel = rels.get(doc_id, 0)
            dcg += rel / math.log2(rank + 1)

        ideal_rels = sorted(rels.values(), reverse=True)[:top_k]
        idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcg_scores)) if ndcg_scores else 0.0


def apply_rrf_fusion(bm25_results, dense_results, k=60):
    """Reciprocal Rank Fusion baseline."""
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())
    for qid in all_qids:
        doc_scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        for rank, (doc_id, _) in enumerate(dense_results.get(qid, []), start=1):
            doc_scores[doc_id] = doc_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        fused[qid] = doc_scores
    return fused


def run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k):
    """Run BM25 retrieval and return {query_id: [(doc_id, score), ...]}."""
    stemmer = SnowballStemmer(stemmer_lang)
    results = {}
    for qid, qtext in tqdm(queries.items(), desc="  BM25 retrieval", dynamic_ncols=True):
        tokens = stem_and_tokenize(qtext, stemmer)
        scores = bm25.get_scores(tokens)
        k = min(top_k, len(scores))
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        results[qid] = [(doc_ids[i], float(scores[i])) for i in top_idx]
    return results


def run_dense_retrieval(
    query_vectors,
    query_ids,
    corpus_vectors,
    corpus_ids,
    top_k,
    corpus_chunk_size,
    device,
    query_chunk_size=100,
):
    """Run chunked cosine-similarity dense retrieval."""
    results = {}
    n_queries = len(query_ids)
    for q_start in tqdm(
        range(0, n_queries, query_chunk_size),
        desc="  Dense retrieval",
        dynamic_ncols=True,
    ):
        q_end = min(q_start + query_chunk_size, n_queries)
        q_batch = query_vectors[q_start:q_end].to(device)
        hits = st_util.semantic_search(
            q_batch,
            corpus_vectors,
            top_k=top_k,
            corpus_chunk_size=corpus_chunk_size,
        )
        for idx, hit_list in enumerate(hits):
            qid = query_ids[q_start + idx]
            results[qid] = [
                (corpus_ids[h["corpus_id"]], float(h["score"]))
                for h in hit_list
            ]
    return results


def ensure_english_stopwords():
    """Load NLTK English stopwords, downloading once if missing."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def ensure_doc_freq_index(tokenized_corpus_jsonl, doc_freq_pkl):
    """Load or build document-frequency index: token -> document count."""
    if file_exists(doc_freq_pkl):
        return load_pickle(doc_freq_pkl)

    doc_freq = {}
    total_docs = 0
    with open(tokenized_corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            total_docs += 1
            for token in set(d.get("tokens", [])):
                doc_freq[token] = doc_freq.get(token, 0) + 1

    save_pickle((doc_freq, total_docs), doc_freq_pkl)
    return doc_freq, total_docs


def build_dynamic_wrrf(bm25_results, dense_results, alpha_map, rrf_k=60):
    """Fuse retrievers with dynamic weighted Reciprocal Rank Fusion.

    For each query:
      score = alpha * (1 / (rrf_k + bm_rank))
            + (1 - alpha) * (1 / (rrf_k + dense_rank))

    Documents not present in one retriever receive default rank 1000 for that
    retriever.
    """
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())
    for qid in all_qids:
        alpha = alpha_map.get(qid, 0.5)

        bm_pairs = sorted(bm25_results.get(qid, []), key=lambda x: x[1], reverse=True)
        de_pairs = sorted(dense_results.get(qid, []), key=lambda x: x[1], reverse=True)

        bm_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(bm_pairs, start=1)}
        de_ranks = {doc_id: rank for rank, (doc_id, _) in enumerate(de_pairs, start=1)}

        docs = set(bm_ranks.keys()) | set(de_ranks.keys())
        q_scores = {}
        for doc_id in docs:
            bm_rank = bm_ranks.get(doc_id, 1000)
            de_rank = de_ranks.get(doc_id, 1000)
            q_scores[doc_id] = (
                alpha * (1.0 / (rrf_k + bm_rank))
                + (1.0 - alpha) * (1.0 / (rrf_k + de_rank))
            )
        fused[qid] = q_scores
    return fused


def clean_query_tokens(
    query_text,
    stemmer,
    stopword_stems,
    doc_freq,
    total_docs,
    max_df,
):
    """Apply stopword + max_df filtering to query tokens."""
    tokens = stem_and_tokenize(query_text, stemmer)
    filtered = []
    for token in tokens:
        if token in stopword_stems:
            continue
        if total_docs > 0 and (doc_freq.get(token, 0) / total_docs) > max_df:
            continue
        filtered.append(token)
    return filtered


def compute_query_metrics(clean_tokens, global_counts, total_corpus_tokens):
    """Compute KLD, JSD, CE_avg for a cleaned query.

    Laplace smoothing is hardcoded to 1.0 as requested.
    """
    if not clean_tokens:
        return None

    laplace_alpha = 1.0
    token_freq = {}
    for token in clean_tokens:
        token_freq[token] = token_freq.get(token, 0) + 1
    unique_tokens = list(token_freq.keys())

    n_query = len(clean_tokens)
    p_q = np.array([token_freq[t] / n_query for t in unique_tokens], dtype=np.float64)

    vocab_size = len(global_counts)
    smoothed_total = total_corpus_tokens + laplace_alpha * vocab_size
    p_c = np.array(
        [(global_counts.get(t, 0) + laplace_alpha) / smoothed_total for t in unique_tokens],
        dtype=np.float64,
    )

    # KLD and JSD from probability distributions.
    kld = float(np.sum(p_q * np.log2(p_q / p_c)))
    m = 0.5 * (p_q + p_c)
    kl_q_m = np.sum(p_q * np.log2(p_q / m))
    kl_c_m = np.sum(p_c * np.log2(p_c / m))
    
    kl_c_m += (1.0 - np.sum(p_c))
    jsd = float(0.5 * kl_q_m + 0.5 * kl_c_m)

    # CE length-normalized by cleaned query length.
    total_ce = 0.0
    for token in clean_tokens:
        token_pc = (global_counts.get(token, 0) + laplace_alpha) / smoothed_total
        total_ce += -math.log2(token_pc)
    ce_avg = float(total_ce / n_query)

    return {"kld": kld, "jsd": jsd, "ce": ce_avg}


def build_alpha_map(metric_values, slope, center):
    """Convert raw metric values to alpha via z-score then centered sigmoid."""
    valid = [v for v in metric_values.values() if v is not None]
    if not valid:
        return {qid: 0.0 for qid in metric_values}

    mean_val = float(np.mean(valid))
    std_val = float(np.std(valid))
    if std_val <= 0:
        std_val = 1.0

    alpha_map = {}
    for qid, value in metric_values.items():
        if value is None:
            alpha_map[qid] = 0.0
        else:
            z = (value - mean_val) / std_val
            alpha_map[qid] = sigmoid(z, slope, center=center)
    return alpha_map


def run_dynamic_grid_search(
    queries,
    qrels,
    stemmer,
    stopword_stems,
    doc_freq,
    total_docs,
    global_counts,
    total_corpus_tokens,
    bm25_results,
    dense_results,
    rrf_k,
    ndcg_k,
    max_df_values,
    k_values,
    center_values,
):
    """Run one unified grid search for JSD, KLD, and CE.

    For each max_df value, query cleaning and metric computation are done
    exactly once and then reused for all metric/k combinations.
    """
    metric_cache = {}
    metric_keys = ("jsd", "kld", "ce")

    for max_df in max_df_values:
        metric_by_key = {k: {} for k in metric_keys}
        for qid, qtext in queries.items():
            cleaned = clean_query_tokens(
                qtext,
                stemmer,
                stopword_stems,
                doc_freq,
                total_docs,
                max_df,
            )
            metrics = compute_query_metrics(cleaned, global_counts, total_corpus_tokens)
            if metrics is None:
                for key in metric_keys:
                    metric_by_key[key][qid] = None
            else:
                for key in metric_keys:
                    metric_by_key[key][qid] = metrics[key]

        metric_cache[max_df] = metric_by_key

    best_by_metric = {
        "jsd": {
            "metric": "JSD",
            "best_max_df": None,
            "best_k": None,
            "best_center": None,
            "best_ndcg": -1.0,
        },
        "kld": {
            "metric": "KLD",
            "best_max_df": None,
            "best_k": None,
            "best_center": None,
            "best_ndcg": -1.0,
        },
        "ce": {
            "metric": "CE",
            "best_max_df": None,
            "best_k": None,
            "best_center": None,
            "best_ndcg": -1.0,
        },
    }

    for max_df in max_df_values:
        metric_by_key = metric_cache[max_df]
        for slope in k_values:
            for center in center_values:
                for metric_key in metric_keys:
                    alpha_map = build_alpha_map(metric_by_key[metric_key], slope, center)
                    fused = build_dynamic_wrrf(
                        bm25_results,
                        dense_results,
                        alpha_map,
                        rrf_k=rrf_k,
                    )
                    ndcg = calculate_ndcg_at_k(fused, qrels, ndcg_k)
                    if ndcg > best_by_metric[metric_key]["best_ndcg"]:
                        best_by_metric[metric_key]["best_ndcg"] = ndcg
                        best_by_metric[metric_key]["best_max_df"] = max_df
                        best_by_metric[metric_key]["best_k"] = slope
                        best_by_metric[metric_key]["best_center"] = center

    return best_by_metric


def save_summary_csv(summary_rows, output_path, ndcg_k):
    """Write per-dataset method scores in wide format."""
    method_order = [
        "BM25 Only",
        "Dense Only",
        "RRF",
        "Dynamic JSD",
        "Dynamic KLD",
        "Dynamic CE",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset"] + [f"{m} (NDCG@{ndcg_k})" for m in method_order])
        for row in summary_rows:
            writer.writerow([row["dataset"]] + [f"{row[m]:.4f}" for m in method_order])


def save_params_csv(param_rows, output_path):
    """Write best dynamic parameters per dataset and metric."""
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Dataset", "Metric", "Best max_df", "Best k", "Best center", "Best NDCG@10"])
        for row in param_rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["metric"],
                    row["best_max_df"],
                    row["best_k"],
                    row["best_center"],
                    f"{row['best_ndcg']:.4f}",
                ]
            )


def save_summary_chart(summary_rows, output_path):
    """Render grouped bars: methods on x-axis, one group per dataset."""
    datasets = [row["dataset"] for row in summary_rows]
    methods = ["BM25 Only", "Dense Only", "RRF", "Dynamic JSD", "Dynamic KLD", "Dynamic CE"]
    values = {m: [row[m] for row in summary_rows] for m in methods}

    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.13

    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, method in enumerate(methods):
        offset = (idx - (len(methods) - 1) / 2.0) * width
        ax.bar(x + offset, values[method], width=width, label=method)

    ax.set_title("Retrieval Summary Across Datasets")
    ax.set_ylabel("NDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.legend(ncols=3, fontsize=9)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def evaluate_dataset(dataset_name, cfg, device):
    """Evaluate one dataset and return summary + best dynamic params."""
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    top_k = cfg["benchmark"]["top_k"]
    ndcg_k = cfg["benchmark"]["ndcg_k"]
    rrf_k = cfg["benchmark"]["rrf"]["k"]
    dense_cfg = cfg["dense_search"]
    dynamic_cfg = cfg["benchmark"]["dynamic_wrrf"]

    max_df_values = dynamic_cfg["max_df_values"]
    k_values = dynamic_cfg["k_values"]
    center_values = dynamic_cfg["center_values"]

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir = os.path.join(processed_root, short_model, dataset_name)

    queries_jsonl = os.path.join(ds_dir, "queries.jsonl")
    qrels_tsv = os.path.join(ds_dir, "qrels.tsv")
    tokenized_corpus_jsonl = os.path.join(ds_dir, "tokenized_corpus.jsonl")
    word_freq_pkl = os.path.join(ds_dir, "word_freq_index.pkl")
    doc_freq_pkl = os.path.join(ds_dir, "doc_freq_index.pkl")

    bm25_pkl = os.path.join(ds_dir, "bm25_index.pkl")
    bm25_docids_pkl = os.path.join(ds_dir, "bm25_doc_ids.pkl")
    bm25_results_pkl = os.path.join(ds_dir, "bm25_results.pkl")

    corpus_emb_pt = os.path.join(ds_dir, "corpus_embeddings.pt")
    corpus_ids_pkl = os.path.join(ds_dir, "corpus_ids.pkl")
    query_vectors_pt = os.path.join(ds_dir, "query_vectors.pt")
    query_ids_pkl = os.path.join(ds_dir, "query_ids.pkl")
    dense_results_pkl = os.path.join(ds_dir, "dense_results.pkl")

    missing = _missing_paths([
        queries_jsonl,
        qrels_tsv,
        tokenized_corpus_jsonl,
        word_freq_pkl,
        bm25_pkl,
        bm25_docids_pkl,
        corpus_emb_pt,
        corpus_ids_pkl,
        query_vectors_pt,
        query_ids_pkl,
    ])
    if missing:
        raise FileNotFoundError(
            "Missing required processed artifacts for "
            f"{dataset_name}:\n  - " + "\n  - ".join(missing)
        )

    print(f"\n{'=' * 70}")
    print(f"Dataset: {dataset_name}")
    print(f"Processed dir: {ds_dir}")
    print(f"{'=' * 70}")

    queries = load_queries(queries_jsonl)
    qrels = load_qrels(qrels_tsv)
    global_counts, total_corpus_tokens = load_pickle(word_freq_pkl)
    doc_freq, total_docs = ensure_doc_freq_index(tokenized_corpus_jsonl, doc_freq_pkl)

    # BM25 retrieval (cached)
    if file_exists(bm25_results_pkl):
        print("[1/5] Loading cached BM25 results ...")
        bm25_results = load_pickle(bm25_results_pkl)
    else:
        print("[1/5] Running BM25 retrieval ...")
        bm25 = load_pickle(bm25_pkl)
        bm25_doc_ids = load_pickle(bm25_docids_pkl)
        bm25_results = run_bm25_retrieval(
            bm25,
            bm25_doc_ids,
            queries,
            stemmer_lang,
            top_k,
        )
        save_pickle(bm25_results, bm25_results_pkl)

    # Dense retrieval (cached)
    if file_exists(dense_results_pkl):
        print("[2/5] Loading cached Dense results ...")
        dense_results = load_pickle(dense_results_pkl)
    else:
        print("[2/5] Running Dense retrieval ...")
        corpus_embeddings = torch.load(corpus_emb_pt, weights_only=True)
        corpus_ids = load_pickle(corpus_ids_pkl)
        query_vectors = torch.load(query_vectors_pt, weights_only=True)
        query_ids = load_pickle(query_ids_pkl)
        dense_results = run_dense_retrieval(
            query_vectors,
            query_ids,
            corpus_embeddings,
            corpus_ids,
            top_k,
            dense_cfg["corpus_chunk_size"],
            device=device,
            query_chunk_size=dense_cfg.get("query_chunk_size", 100),
        )
        save_pickle(dense_results, dense_results_pkl)

    print("[3/5] Evaluating baselines ...")
    bm25_only = {qid: {d: s for d, s in pairs} for qid, pairs in bm25_results.items()}
    dense_only = {qid: {d: s for d, s in pairs} for qid, pairs in dense_results.items()}
    rrf_scores = apply_rrf_fusion(bm25_results, dense_results, k=rrf_k)

    bm25_ndcg = calculate_ndcg_at_k(bm25_only, qrels, ndcg_k)
    dense_ndcg = calculate_ndcg_at_k(dense_only, qrels, ndcg_k)
    rrf_ndcg = calculate_ndcg_at_k(rrf_scores, qrels, ndcg_k)

    print("[4/5] Preparing dynamic routing inputs ...")

    print("[5/5] Grid search for dynamic methods ...")
    stemmer = SnowballStemmer(stemmer_lang)
    stopword_stems = {stemmer.stem(w) for w in ensure_english_stopwords()}

    best_dynamic = run_dynamic_grid_search(
        queries,
        qrels,
        stemmer,
        stopword_stems,
        doc_freq,
        total_docs,
        global_counts,
        total_corpus_tokens,
        bm25_results,
        dense_results,
        rrf_k,
        ndcg_k,
        max_df_values,
        k_values,
        center_values,
    )
    best_jsd = best_dynamic["jsd"]
    best_kld = best_dynamic["kld"]
    best_ce = best_dynamic["ce"]

    print(f"  BM25 Only    : {bm25_ndcg:.4f}")
    print(f"  Dense Only   : {dense_ndcg:.4f}")
    print(f"  RRF          : {rrf_ndcg:.4f}")
    print(
        f"  Dynamic JSD  : {best_jsd['best_ndcg']:.4f}  "
        f"(max_df={best_jsd['best_max_df']}, k={best_jsd['best_k']}, center={best_jsd['best_center']})"
    )
    print(
        f"  Dynamic KLD  : {best_kld['best_ndcg']:.4f}  "
        f"(max_df={best_kld['best_max_df']}, k={best_kld['best_k']}, center={best_kld['best_center']})"
    )
    print(
        f"  Dynamic CE   : {best_ce['best_ndcg']:.4f}  "
        f"(max_df={best_ce['best_max_df']}, k={best_ce['best_k']}, center={best_ce['best_center']})"
    )

    summary_row = {
        "dataset": dataset_name,
        "BM25 Only": bm25_ndcg,
        "Dense Only": dense_ndcg,
        "RRF": rrf_ndcg,
        "Dynamic JSD": best_jsd["best_ndcg"],
        "Dynamic KLD": best_kld["best_ndcg"],
        "Dynamic CE": best_ce["best_ndcg"],
    }

    param_rows = []
    for best in [best_jsd, best_kld, best_ce]:
        param_rows.append(
            {
                "dataset": dataset_name,
                "metric": best["metric"],
                "best_max_df": best["best_max_df"],
                "best_k": best["best_k"],
                "best_center": best["best_center"],
                "best_ndcg": best["best_ndcg"],
            }
        )

    return summary_row, param_rows


def main():
    parser = argparse.ArgumentParser(
        description="Run retrieval + evaluation for all configured datasets."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        print("No datasets configured. Nothing to evaluate.")
        sys.exit(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = cfg["embeddings"]["model_name"]
    short_model = model_short_name(model_name)
    results_root = get_config_path(cfg, "results_folder", "data/results")
    model_results_dir = os.path.join(results_root, short_model)
    ensure_dir(model_results_dir)

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")

    started = time.time()
    summary_rows = []
    all_param_rows = []
    for dataset_name in datasets:
        row, param_rows = evaluate_dataset(dataset_name, cfg, device)
        summary_rows.append(row)
        all_param_rows.extend(param_rows)

    ndcg_k = cfg["benchmark"]["ndcg_k"]
    summary_csv = os.path.join(model_results_dir, "summary_ndcg.csv")
    params_csv = os.path.join(model_results_dir, "best_dynamic_params.csv")
    summary_png = os.path.join(model_results_dir, "summary_ndcg.png")
    timing_csv = os.path.join(model_results_dir, "retrieval_timing.csv")

    save_summary_csv(summary_rows, summary_csv, ndcg_k)
    save_params_csv(all_param_rows, params_csv)
    save_summary_chart(summary_rows, summary_png)

    elapsed = time.time() - started
    with open(timing_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "Time (s)"])
        writer.writerow(["Retrieval + evaluation all datasets", f"{elapsed:.2f}"])

    print("\n" + "=" * 70)
    print(f"Summary CSV   : {summary_csv}")
    print(f"Params CSV    : {params_csv}")
    print(f"Summary chart : {summary_png}")
    print(f"Timing CSV    : {timing_csv}")
    print("=" * 70)


if __name__ == "__main__":
    main()
