"""retrieve_and_evaluate.py

Run supervised query-level routing in a leave-one-dataset-out (LOODO) setting.

This script reuses preprocessing artifacts and retrieval caches to benchmark:
  1) Dense only
  2) Sparse only (BM25)
  3) Static RRF
  4) Dynamic weighted RRF with per-query alpha predicted by logistic regression

Important methodological points:
  - No oracle search on the held-out dataset.
  - Soft labels are query-level targets in [0, 1], not hard classes.
  - Logistic regression is implemented in PyTorch with BCEWithLogitsLoss.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sentence_transformers import util as st_util
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


# Ensure project root is importable and set as working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import src.utils as u
from src.utils import (
    ensure_dir,
    file_exists,
    get_config_path,
    load_config,
    load_pickle,
    load_qrels,
    load_queries,
    model_short_name,
    save_pickle,
    stem_and_tokenize,
)


FEATURE_NAMES = [
    "cross_entropy",
    "agreement",
    "query_length",
    "dense_confidence",
    "sparse_confidence",
    "top_dense_score",
    "top_sparse_score",
    "average_idf",
    "max_idf",
    "stopword_ratio",
]


def set_global_seed(seed):
    """Set random seeds for reproducible model training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_english_stopwords():
    """Load NLTK English stopwords, downloading once if needed."""
    try:
        return set(stopwords.words("english"))
    except LookupError:
        import nltk
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))


def _missing_paths(paths):
    """Return all paths that do not exist."""
    return [p for p in paths if not file_exists(p)]


def query_ndcg_at_k(ranked_pairs, rels, k):
    """Compute query-level NDCG@k from a ranked list of (doc_id, score)."""
    if not rels:
        return 0.0

    dcg = 0.0
    for rank, (doc_id, _) in enumerate(ranked_pairs[:k], start=1):
        rel = rels.get(doc_id, 0)
        gain = (2.0 ** rel) - 1.0
        dcg += gain / math.log2(rank + 1)

    ideal_rels = sorted(rels.values(), reverse=True)[:k]
    idcg = sum((((2.0 ** r) - 1.0) / math.log2(i + 2)) for i, r in enumerate(ideal_rels))
    if idcg <= 0:
        return 0.0
    return float(dcg / idcg)


def calculate_dataset_ndcg_at_k(score_map_by_qid, qrels, k):
    """Compute average NDCG@k for a dataset from qid -> {doc_id: score}."""
    ndcgs = []
    for qid, rels in qrels.items():
        q_scores = score_map_by_qid.get(qid, {})
        ranked = sorted(q_scores.items(), key=lambda x: x[1], reverse=True)
        ndcgs.append(query_ndcg_at_k(ranked, rels, k))
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def bm25_and_dense_to_score_maps(bm25_results, dense_results):
    """Convert ranked retrieval lists to score dictionaries per query."""
    sparse_scores = {qid: {doc_id: score for doc_id, score in pairs} for qid, pairs in bm25_results.items()}
    dense_scores = {qid: {doc_id: score for doc_id, score in pairs} for qid, pairs in dense_results.items()}
    return sparse_scores, dense_scores


def apply_static_rrf(bm25_results, dense_results, rrf_k):
    """Fuse sparse and dense ranked lists with static Reciprocal Rank Fusion."""
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())

    for qid in all_qids:
        q_scores = {}
        for rank, (doc_id, _) in enumerate(bm25_results.get(qid, []), start=1):
            q_scores[doc_id] = q_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, (doc_id, _) in enumerate(dense_results.get(qid, []), start=1):
            q_scores[doc_id] = q_scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
        fused[qid] = q_scores

    return fused


def apply_dynamic_wrrf(bm25_results, dense_results, alpha_map, rrf_k):
    """Fuse sparse and dense rankings with per-query weighted RRF.

    score = alpha * 1 / (rrf_k + rank_sparse)
          + (1 - alpha) * 1 / (rrf_k + rank_dense)
    """
    fused = {}
    all_qids = set(bm25_results.keys()) | set(dense_results.keys())

    for qid in all_qids:
        alpha = float(alpha_map.get(qid, 0.5))

        bm_pairs = bm25_results.get(qid, [])
        de_pairs = dense_results.get(qid, [])
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


def run_bm25_retrieval(bm25, doc_ids, queries, stemmer_lang, top_k, use_stemming):
    """Run BM25 retrieval and return {qid: [(doc_id, score), ...]} with top_k docs."""
    stemmer = SnowballStemmer(stemmer_lang) if use_stemming else None
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
    query_chunk_size,
):
    """Run chunked dense retrieval and return {qid: [(doc_id, score), ...]}."""
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
            results[qid] = [(corpus_ids[h["corpus_id"]], float(h["score"])) for h in hit_list]

    return results


def prepare_dataset_inputs(dataset_name, cfg):
    """Resolve and validate required processed artifact paths for one dataset."""
    short_model = model_short_name(cfg["embeddings"]["model_name"])
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    ds_dir = os.path.join(processed_root, short_model, dataset_name)
    bm25_params = u.get_bm25_params(cfg)
    bm25_paths = u.bm25_artifact_paths(
        ds_dir,
        bm25_params["k1"],
        bm25_params["b"],
        bm25_params["use_stemming"],
    )

    paths = {
        "dataset_dir": ds_dir,
        "queries_jsonl": os.path.join(ds_dir, "queries.jsonl"),
        "qrels_tsv": os.path.join(ds_dir, "qrels.tsv"),
        "word_freq_pkl": bm25_paths["word_freq_pkl"],
        "doc_freq_pkl": bm25_paths["doc_freq_pkl"],
        "query_tokens_pkl": bm25_paths["query_tokens_pkl"],
        "bm25_pkl": bm25_paths["bm25_pkl"],
        "bm25_docids_pkl": bm25_paths["bm25_docids_pkl"],
        "corpus_emb_pt": os.path.join(ds_dir, "corpus_embeddings.pt"),
        "corpus_ids_pkl": os.path.join(ds_dir, "corpus_ids.pkl"),
        "query_vectors_pt": os.path.join(ds_dir, "query_vectors.pt"),
        "query_ids_pkl": os.path.join(ds_dir, "query_ids.pkl"),
        "bm25_results_pkl": bm25_paths["bm25_results_pkl"],
        "dense_results_pkl": os.path.join(ds_dir, "dense_results.pkl"),
        "bm25_signature": bm25_paths["bm25_signature"],
    }

    missing = _missing_paths([
        paths["queries_jsonl"],
        paths["qrels_tsv"],
        paths["word_freq_pkl"],
        paths["doc_freq_pkl"],
        paths["query_tokens_pkl"],
        paths["bm25_pkl"],
        paths["bm25_docids_pkl"],
        paths["corpus_emb_pt"],
        paths["corpus_ids_pkl"],
        paths["query_vectors_pt"],
        paths["query_ids_pkl"],
    ])
    if missing:
        raise FileNotFoundError(
            "Missing preprocessing artifacts for "
            f"{dataset_name}:\n  - " + "\n  - ".join(missing)
        )

    return paths


def ensure_retrieval_results_cached(dataset_name, cfg, device):
    """Load or compute cached sparse and dense retrieval results for one dataset."""
    paths = prepare_dataset_inputs(dataset_name, cfg)

    top_k = cfg["benchmark"]["top_k"]
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    dense_cfg = cfg.get("dense_search", {})
    bm25_params = u.get_bm25_params(cfg)

    print(
        "  BM25 config: "
        f"k1={bm25_params['k1']}, b={bm25_params['b']}, "
        f"use_stemming={bm25_params['use_stemming']}"
    )
    print(f"  BM25 signature: {paths['bm25_signature']}")

    queries = load_queries(paths["queries_jsonl"])

    if file_exists(paths["bm25_results_pkl"]):
        print("  Loading cached BM25 retrieval results ...")
        bm25_results = load_pickle(paths["bm25_results_pkl"])
    else:
        print("  Running BM25 retrieval and caching results ...")
        bm25 = load_pickle(paths["bm25_pkl"])
        bm25_doc_ids = load_pickle(paths["bm25_docids_pkl"])
        bm25_results = run_bm25_retrieval(
            bm25,
            bm25_doc_ids,
            queries,
            stemmer_lang,
            top_k,
            use_stemming=bm25_params["use_stemming"],
        )
        save_pickle(bm25_results, paths["bm25_results_pkl"])

    if file_exists(paths["dense_results_pkl"]):
        print("  Loading cached dense retrieval results ...")
        dense_results = load_pickle(paths["dense_results_pkl"])
    else:
        print("  Running dense retrieval and caching results ...")
        corpus_embeddings = torch.load(paths["corpus_emb_pt"], weights_only=True)
        corpus_ids = load_pickle(paths["corpus_ids_pkl"])
        query_vectors = torch.load(paths["query_vectors_pt"], weights_only=True)
        query_ids = load_pickle(paths["query_ids_pkl"])
        dense_results = run_dense_retrieval(
            query_vectors=query_vectors,
            query_ids=query_ids,
            corpus_vectors=corpus_embeddings,
            corpus_ids=corpus_ids,
            top_k=top_k,
            corpus_chunk_size=dense_cfg.get("corpus_chunk_size", 50000),
            device=device,
            query_chunk_size=dense_cfg.get("query_chunk_size", 100),
        )
        save_pickle(dense_results, paths["dense_results_pkl"])

    qrels = load_qrels(paths["qrels_tsv"])
    query_tokens = load_pickle(paths["query_tokens_pkl"])
    word_freq, total_corpus_tokens = load_pickle(paths["word_freq_pkl"])
    doc_freq, total_docs = load_pickle(paths["doc_freq_pkl"])

    return {
        "dataset": dataset_name,
        "paths": paths,
        "qrels": qrels,
        "query_tokens": query_tokens,
        "word_freq": word_freq,
        "total_corpus_tokens": total_corpus_tokens,
        "doc_freq": doc_freq,
        "total_docs": total_docs,
        "bm25_results": bm25_results,
        "dense_results": dense_results,
    }


def compute_feature_row_for_query(
    dataset_name,
    qid,
    query_tokens,
    qrels_for_query,
    bm25_for_query,
    dense_for_query,
    word_freq,
    total_corpus_tokens,
    doc_freq,
    total_docs,
    stopword_stems,
    overlap_k,
    epsilon,
    ce_smoothing_alpha,
    ndcg_k,
):
    """Compute one query row with features and soft label."""
    total_tokens = len(query_tokens)
    if total_tokens == 0:
        stopword_ratio = 0.0
        cleaned_tokens = []
    else:
        n_stop = sum(1 for t in query_tokens if t in stopword_stems)
        stopword_ratio = float(n_stop / total_tokens)
        cleaned_tokens = [t for t in query_tokens if t not in stopword_stems]

    # Feature: query length from original tokenized query (no stopword removal).
    query_length = float(total_tokens)

    # Feature: top-k overlap between sparse and dense result sets.
    top_sparse = [doc_id for doc_id, _ in bm25_for_query[:overlap_k]]
    top_dense = [doc_id for doc_id, _ in dense_for_query[:overlap_k]]
    overlap = len(set(top_sparse) & set(top_dense))
    agreement = float(overlap / max(1, overlap_k))

    # Features: confidence margins from the top two scores.
    if len(dense_for_query) >= 2:
        dense_confidence = float(dense_for_query[0][1] - dense_for_query[1][1])
    elif len(dense_for_query) == 1:
        dense_confidence = float(dense_for_query[0][1])
    else:
        dense_confidence = 0.0

    top_dense_score = float(dense_for_query[0][1]) if dense_for_query else 0.0

    if len(bm25_for_query) >= 2:
        sparse_confidence = float(bm25_for_query[0][1] - bm25_for_query[1][1])
    elif len(bm25_for_query) == 1:
        sparse_confidence = float(bm25_for_query[0][1])
    else:
        sparse_confidence = 0.0

    top_sparse_score = float(bm25_for_query[0][1]) if bm25_for_query else 0.0

    # Features requiring stopword-filtered tokens.
    vocab_size = max(1, len(word_freq))
    corpus_mass = total_corpus_tokens + ce_smoothing_alpha * vocab_size

    if not cleaned_tokens:
        cross_entropy = 0.0
        average_idf = 0.0
        max_idf = 0.0
    else:
        ce_sum = 0.0
        idf_values = []
        for token in cleaned_tokens:
            prob = (word_freq.get(token, 0) + ce_smoothing_alpha) / corpus_mass
            ce_sum += -math.log2(max(prob, epsilon))

            # Smooth IDF to avoid singularities and keep finite values.
            idf = math.log((total_docs + 1.0) / (doc_freq.get(token, 0) + 1.0)) + 1.0
            idf_values.append(idf)

        cross_entropy = float(ce_sum / len(cleaned_tokens))
        average_idf = float(sum(idf_values) / len(idf_values))
        max_idf = float(max(idf_values))

    # Soft label from sparse-only and dense-only query-level NDCG@k.
    sparse_q_ndcg = query_ndcg_at_k(bm25_for_query, qrels_for_query, ndcg_k)
    dense_q_ndcg = query_ndcg_at_k(dense_for_query, qrels_for_query, ndcg_k)
    label = 0.5 * (((sparse_q_ndcg - dense_q_ndcg) / (sparse_q_ndcg + dense_q_ndcg + epsilon)) + 1.0)
    label = float(np.clip(label, 0.0, 1.0))

    return {
        "dataset": dataset_name,
        "query_id": qid,
        "features": {
            "cross_entropy": cross_entropy,
            "agreement": agreement,
            "query_length": query_length,
            "dense_confidence": dense_confidence,
            "sparse_confidence": sparse_confidence,
            "top_dense_score": top_dense_score,
            "top_sparse_score": top_sparse_score,
            "average_idf": average_idf,
            "max_idf": max_idf,
            "stopword_ratio": stopword_ratio,
        },
        "soft_label": label,
        "sparse_q_ndcg": sparse_q_ndcg,
        "dense_q_ndcg": dense_q_ndcg,
    }


def build_or_load_query_feature_cache(dataset_cache_map, cfg, short_model):
    """Build and cache per-query features + soft labels across all datasets."""
    processed_root = get_config_path(cfg, "processed_folder", "data/processed_data")
    cache_dir = os.path.join(processed_root, short_model, "routing_cache")
    ensure_dir(cache_dir)

    routing_cfg = cfg.get("supervised_routing", {})
    overlap_k = int(routing_cfg.get("overlap_k", 100))
    epsilon = float(routing_cfg.get("epsilon", 1.0e-8))
    ce_smoothing_alpha = float(routing_cfg.get("ce_smoothing_alpha", 1.0))
    feature_workers = int(routing_cfg.get("feature_workers", max(1, (os.cpu_count() or 4) // 2)))
    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))

    datasets = sorted(dataset_cache_map.keys())
    signature_payload = {
        "datasets": datasets,
        "feature_names": FEATURE_NAMES,
        "overlap_k": overlap_k,
        "epsilon": epsilon,
        "ce_smoothing_alpha": ce_smoothing_alpha,
        "ndcg_k": ndcg_k,
        "top_k": int(cfg["benchmark"].get("top_k", 100)),
        "model": cfg["embeddings"]["model_name"],
        "bm25": u.get_bm25_params(cfg),
    }
    signature = hashlib.md5(json.dumps(signature_payload, sort_keys=True).encode("utf-8")).hexdigest()

    feature_cache_pkl = os.path.join(cache_dir, "query_feature_label_cache.pkl")
    feature_cache_csv = os.path.join(cache_dir, "query_feature_label_cache.csv")

    if file_exists(feature_cache_pkl):
        cached = load_pickle(feature_cache_pkl)
        if isinstance(cached, dict) and cached.get("signature") == signature:
            print("[Feature Cache] Reusing cached per-query feature/label table.")
            return cached["rows"], feature_cache_pkl, feature_cache_csv

    print("[Feature Cache] Building per-query feature/label table from cached retrieval outputs ...")

    english_stopwords = ensure_english_stopwords()
    stemmer_lang = cfg["preprocessing"]["stemmer_language"]
    stemmer = SnowballStemmer(stemmer_lang)
    stopword_stems = {stemmer.stem(w) for w in english_stopwords}

    all_rows = []
    for dataset_name in datasets:
        ds = dataset_cache_map[dataset_name]
        qids = sorted(ds["query_tokens"].keys())

        print(f"  Dataset {dataset_name}: computing query-level features for {len(qids):,} queries ...")

        def _worker(qid):
            return compute_feature_row_for_query(
                dataset_name=dataset_name,
                qid=qid,
                query_tokens=ds["query_tokens"].get(qid, []),
                qrels_for_query=ds["qrels"].get(qid, {}),
                bm25_for_query=ds["bm25_results"].get(qid, []),
                dense_for_query=ds["dense_results"].get(qid, []),
                word_freq=ds["word_freq"],
                total_corpus_tokens=ds["total_corpus_tokens"],
                doc_freq=ds["doc_freq"],
                total_docs=ds["total_docs"],
                stopword_stems=stopword_stems,
                overlap_k=overlap_k,
                epsilon=epsilon,
                ce_smoothing_alpha=ce_smoothing_alpha,
                ndcg_k=ndcg_k,
            )

        with ThreadPoolExecutor(max_workers=max(1, feature_workers)) as ex:
            for row in tqdm(
                ex.map(_worker, qids),
                total=len(qids),
                desc=f"  Features {dataset_name}",
                dynamic_ncols=True,
            ):
                all_rows.append(row)

    save_pickle(
        {
            "signature": signature,
            "signature_payload": signature_payload,
            "rows": all_rows,
        },
        feature_cache_pkl,
    )

    with open(feature_cache_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "query_id",
            *FEATURE_NAMES,
            "soft_label",
            "sparse_q_ndcg",
            "dense_q_ndcg",
        ])
        for row in all_rows:
            writer.writerow([
                row["dataset"],
                row["query_id"],
                *(row["features"][name] for name in FEATURE_NAMES),
                row["soft_label"],
                row["sparse_q_ndcg"],
                row["dense_q_ndcg"],
            ])

    print(f"[Feature Cache] Wrote {len(all_rows):,} rows to cache.")
    return all_rows, feature_cache_pkl, feature_cache_csv


def rows_to_matrix(rows):
    """Convert feature rows to X matrix, y vector, and qid list."""
    qids = [r["query_id"] for r in rows]
    X = np.asarray(
        [[float(r["features"][name]) for name in FEATURE_NAMES] for r in rows],
        dtype=np.float32,
    )
    y = np.asarray([float(r["soft_label"]) for r in rows], dtype=np.float32)
    return X, y, qids


def compute_zscore_stats(X):
    """Compute z-score stats with zero-variance safeguard."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std <= 1.0e-12] = 1.0
    return mean, std


def apply_zscore(X, mean, std):
    """Apply z-score normalization to a feature matrix."""
    return (X - mean) / std


class LogisticRegressor(nn.Module):
    """Simple logistic regression: one linear layer producing a scalar logit."""

    def __init__(self, in_features, fit_intercept=True):
        super().__init__()
        self.linear = nn.Linear(in_features, 1, bias=fit_intercept)

    def forward(self, x):
        return self.linear(x).squeeze(-1)


def train_logistic_regression_pytorch(X_train, y_train, cfg, device):
    """Train logistic regression with BCEWithLogitsLoss on soft labels.

    LBFGS is the default because this is a convex one-layer logistic model,
    which makes LBFGS a close match to classical logistic regression training.
    Adam is retained as an explicit fallback option for experimentation.
    """
    routing_cfg = cfg.get("supervised_routing", {})

    fit_intercept = bool(routing_cfg.get("fit_intercept", True))
    optimizer_name = str(routing_cfg.get("optimizer", "lbfgs")).lower()
    learning_rate = float(routing_cfg.get("learning_rate", 0.05))
    epochs = int(routing_cfg.get("epochs", 300))
    batch_size = int(routing_cfg.get("batch_size", 256))
    patience = int(routing_cfg.get("early_stopping_patience", 30))
    min_delta = float(routing_cfg.get("early_stopping_min_delta", 1.0e-6))
    regularization = str(routing_cfg.get("regularization", "l2")).lower()
    C = float(routing_cfg.get("C", 1.0))

    if regularization != "l2":
        print(f"  [WARN] regularization={regularization} is not supported. Falling back to l2.")
    l2_lambda = 1.0 / max(C, 1.0e-12)

    model = LogisticRegressor(in_features=X_train.shape[1], fit_intercept=fit_intercept).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    x_tensor = torch.from_numpy(X_train).to(device)
    y_tensor = torch.from_numpy(y_train).to(device)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    if optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS(
            model.parameters(),
            lr=learning_rate,
            max_iter=1,
            history_size=20,
            line_search_fn="strong_wolfe",
        )

        pbar = tqdm(range(1, epochs + 1), desc="  Training logistic model (LBFGS)", dynamic_ncols=True)
        for epoch in pbar:
            model.train()

            def closure():
                optimizer.zero_grad(set_to_none=True)
                logits = model(x_tensor)
                data_loss = loss_fn(logits, y_tensor)
                # Apply L2 regularization to weights (not bias) for classical behavior.
                l2_penalty = (model.linear.weight ** 2).sum()
                total_loss = data_loss + (l2_lambda * l2_penalty)
                total_loss.backward()
                return total_loss

            optimizer.step(closure)

            model.eval()
            with torch.no_grad():
                logits = model(x_tensor)
                data_loss = loss_fn(logits, y_tensor)
                l2_penalty = (model.linear.weight ** 2).sum()
                epoch_loss = float((data_loss + (l2_lambda * l2_penalty)).item())

            pbar.set_postfix(loss=f"{epoch_loss:.6f}")

            if epoch_loss + min_delta < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best training loss={best_loss:.6f}).")
                break
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=l2_lambda)
        train_loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=min(batch_size, len(x_tensor)),
            shuffle=True,
        )

        pbar = tqdm(range(1, epochs + 1), desc="  Training logistic model (Adam)", dynamic_ncols=True)
        for epoch in pbar:
            model.train()
            epoch_loss = 0.0

            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item()) * len(xb)

            epoch_loss /= len(x_tensor)
            pbar.set_postfix(loss=f"{epoch_loss:.6f}")

            if epoch_loss + min_delta < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (best training loss={best_loss:.6f}).")
                break
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}. Use 'lbfgs' or 'adam'.")

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def predict_alpha(model, X, device):
    """Predict per-query alpha in [0, 1] using sigmoid(logit)."""
    with torch.no_grad():
        x_tensor = torch.from_numpy(X).to(device)
        logits = model(x_tensor)
        probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy().astype(np.float32)


def extract_model_coefficients(model, feature_names):
    """Return intercept and feature coefficients from a trained logistic model."""
    weights = model.linear.weight.detach().cpu().numpy().reshape(-1)
    if model.linear.bias is None:
        intercept = 0.0
    else:
        intercept = float(model.linear.bias.detach().cpu().item())

    coef_by_feature = {name: float(weights[idx]) for idx, name in enumerate(feature_names)}
    return intercept, coef_by_feature


def save_per_dataset_results(rows, output_csv):
    """Save per-dataset benchmark results as CSV."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset",
            "dense_only_ndcg",
            "sparse_only_ndcg",
            "static_rrf_ndcg",
            "dynamic_wrrf_ndcg",
        ])
        for row in rows:
            writer.writerow([
                row["dataset"],
                f"{row['dense_only_ndcg']:.6f}",
                f"{row['sparse_only_ndcg']:.6f}",
                f"{row['static_rrf_ndcg']:.6f}",
                f"{row['dynamic_wrrf_ndcg']:.6f}",
            ])


def save_macro_summary(rows, output_csv):
    """Save macro-average NDCG summary across datasets."""
    methods = [
        ("Dense only", "dense_only_ndcg"),
        ("Sparse only", "sparse_only_ndcg"),
        ("Static RRF", "static_rrf_ndcg"),
        ("Dynamic wRRF", "dynamic_wrrf_ndcg"),
    ]

    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "macro_avg_ndcg"])
        for label, key in methods:
            writer.writerow([label, f"{np.mean([r[key] for r in rows]):.6f}"])


def save_fold_coefficients_csv(rows, output_csv):
    """Save one readable coefficient table across all LOODO folds."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["heldout_dataset", "term", "coefficient"])
        for row in rows:
            writer.writerow([row["heldout_dataset"], row["term"], f"{row['coefficient']:.12f}"])


def save_fold_normalization_stats_csv(rows, output_csv):
    """Save train/test z-score statistics for each fold and feature."""
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["heldout_dataset", "split", "feature", "mean", "std"])
        for row in rows:
            writer.writerow(
                [
                    row["heldout_dataset"],
                    row["split"],
                    row["feature"],
                    f"{row['mean']:.12f}",
                    f"{row['std']:.12f}",
                ]
            )


def save_plots(rows, alpha_rows, output_dir):
    """Save publication-style plots for per-dataset and macro comparisons."""
    ensure_dir(output_dir)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
        }
    )

    datasets = [r["dataset"] for r in rows]
    dense_vals = [r["dense_only_ndcg"] for r in rows]
    sparse_vals = [r["sparse_only_ndcg"] for r in rows]
    rrf_vals = [r["static_rrf_ndcg"] for r in rows]
    dyn_vals = [r["dynamic_wrrf_ndcg"] for r in rows]

    x = np.arange(len(datasets), dtype=np.float64)
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 1.5 * width, dense_vals, width=width, label="Dense only", color="#355C7D")
    ax.bar(x - 0.5 * width, sparse_vals, width=width, label="Sparse only", color="#6C5B7B")
    ax.bar(x + 0.5 * width, rrf_vals, width=width, label="Static RRF", color="#F67280")
    ax.bar(x + 1.5 * width, dyn_vals, width=width, label="Dynamic wRRF", color="#C06C84")

    ax.set_title("LOODO Benchmark per Dataset")
    ax.set_ylabel("NDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "per_dataset_comparison.png"), dpi=220)
    plt.close(fig)

    macro_values = [
        np.mean(dense_vals),
        np.mean(sparse_vals),
        np.mean(rrf_vals),
        np.mean(dyn_vals),
    ]
    macro_labels = ["Dense only", "Sparse only", "Static RRF", "Dynamic wRRF"]
    macro_colors = ["#355C7D", "#6C5B7B", "#F67280", "#C06C84"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(macro_labels, macro_values, color=macro_colors)
    ax.set_title("Macro-Average LOODO Performance")
    ax.set_ylabel("NDCG@10")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25)
    for bar, val in zip(bars, macro_values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.4f}", ha="center", va="bottom")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "macro_average_comparison.png"), dpi=220)
    plt.close(fig)

    if alpha_rows:
        by_dataset = {}
        for row in alpha_rows:
            by_dataset.setdefault(row["dataset"], []).append(row["alpha"])

        labels = sorted(by_dataset.keys())
        values = [by_dataset[k] for k in labels]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.boxplot(values, labels=labels, patch_artist=True)
        ax.set_title("Predicted Alpha Distribution per Held-Out Dataset")
        ax.set_ylabel("Predicted alpha (0=dense, 1=sparse)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, "alpha_distribution.png"), dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Run LOODO supervised query-level routing benchmark."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if len(datasets) < 2:
        raise ValueError("LOODO requires at least two configured datasets.")

    routing_cfg = cfg.get("supervised_routing", {})
    seed = int(routing_cfg.get("seed", 42))
    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True))
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    model_name = cfg["embeddings"]["model_name"]
    short_model = model_short_name(model_name)
    ndcg_k = int(cfg["benchmark"].get("ndcg_k", 10))
    rrf_k = int(cfg["benchmark"].get("rrf", {}).get("k", 60))

    results_root = get_config_path(cfg, "results_folder", "data/results")
    benchmark_dir = os.path.join(results_root, short_model, "supervised_routing")
    plots_dir = os.path.join(benchmark_dir, "plots")
    model_cache_dir = os.path.join(benchmark_dir, "fold_models")
    ensure_dir(benchmark_dir)
    ensure_dir(plots_dir)
    ensure_dir(model_cache_dir)

    print(f"Device : {device}")
    print(f"Model  : {model_name}")
    print(f"Datasets ({len(datasets)}): {', '.join(datasets)}")
    print("\n[1/4] Loading cached retrieval artifacts per dataset ...")

    dataset_cache_map = {}
    for dataset_name in datasets:
        print(f"\n{'=' * 68}")
        print(f"Dataset: {dataset_name}")
        print(f"{'=' * 68}")
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/4] Building or loading per-query feature/label cache ...")
    all_rows, feature_cache_pkl, feature_cache_csv = build_or_load_query_feature_cache(
        dataset_cache_map,
        cfg,
        short_model,
    )

    rows_by_dataset = {ds: [] for ds in datasets}
    for row in all_rows:
        rows_by_dataset[row["dataset"]].append(row)

    # Keep query-order deterministic across runs.
    for ds in datasets:
        rows_by_dataset[ds].sort(key=lambda r: r["query_id"])

    print("\n[3/4] Running LOODO folds with cached model reuse ...")
    fold_results = []
    alpha_rows = []
    coefficient_rows = []
    fold_norm_rows = []

    for heldout in datasets:
        train_datasets = [d for d in datasets if d != heldout]
        train_rows = [row for d in train_datasets for row in rows_by_dataset[d]]
        test_rows = list(rows_by_dataset[heldout])

        if not train_rows or not test_rows:
            raise ValueError(f"Fold {heldout}: train/test rows are empty.")

        print(f"\n{'-' * 68}")
        print(f"Held-out dataset: {heldout}")
        print(f"Train datasets   : {', '.join(train_datasets)}")
        print(f"Train queries    : {len(train_rows):,} | Test queries: {len(test_rows):,}")
        print(f"{'-' * 68}")

        X_train_raw, y_train, _ = rows_to_matrix(train_rows)
        X_test_raw, _, test_qids = rows_to_matrix(test_rows)

        # Train split normalized by training-fold statistics.
        train_mean, train_std = compute_zscore_stats(X_train_raw)
        X_train = apply_zscore(X_train_raw, train_mean, train_std)

        # Held-out dataset normalized independently using unlabeled test features.
        test_mean, test_std = compute_zscore_stats(X_test_raw)
        X_test = apply_zscore(X_test_raw, test_mean, test_std)

        for feat_idx, feat_name in enumerate(FEATURE_NAMES):
            fold_norm_rows.append(
                {
                    "heldout_dataset": heldout,
                    "split": "train",
                    "feature": feat_name,
                    "mean": float(train_mean[feat_idx]),
                    "std": float(train_std[feat_idx]),
                }
            )
            fold_norm_rows.append(
                {
                    "heldout_dataset": heldout,
                    "split": "test",
                    "feature": feat_name,
                    "mean": float(test_mean[feat_idx]),
                    "std": float(test_std[feat_idx]),
                }
            )

        fold_signature_payload = {
            "heldout": heldout,
            "train_datasets": train_datasets,
            "feature_names": FEATURE_NAMES,
            "bm25": u.get_bm25_params(cfg),
            "training_cfg": {
                "regularization": routing_cfg.get("regularization", "l2"),
                "C": routing_cfg.get("C", 1.0),
                "fit_intercept": routing_cfg.get("fit_intercept", True),
                "optimizer": routing_cfg.get("optimizer", "lbfgs"),
                "learning_rate": routing_cfg.get("learning_rate", 0.05),
                "epochs": routing_cfg.get("epochs", 300),
                "batch_size": routing_cfg.get("batch_size", 256),
                "early_stopping_patience": routing_cfg.get("early_stopping_patience", 30),
                "early_stopping_min_delta": routing_cfg.get("early_stopping_min_delta", 1.0e-6),
                "seed": routing_cfg.get("seed", 42),
            },
            "model": model_name,
        }
        fold_signature = hashlib.md5(
            json.dumps(fold_signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        fold_model_path = os.path.join(model_cache_dir, f"fold_{heldout}.pt")

        model = LogisticRegressor(
            in_features=len(FEATURE_NAMES),
            fit_intercept=bool(routing_cfg.get("fit_intercept", True)),
        ).to(device)

        if file_exists(fold_model_path):
            payload = torch.load(fold_model_path, map_location=device, weights_only=False)
            if payload.get("signature") == fold_signature:
                print("  Reusing cached fold model.")
                model.load_state_dict(payload["state_dict"])
            else:
                print("  Fold cache exists but signature changed; retraining.")
                model = train_logistic_regression_pytorch(X_train, y_train, cfg, device)
                torch.save(
                    {
                        "signature": fold_signature,
                        "signature_payload": fold_signature_payload,
                        "state_dict": model.state_dict(),
                    },
                    fold_model_path,
                )
        else:
            print("  Training fold model from scratch ...")
            model = train_logistic_regression_pytorch(X_train, y_train, cfg, device)
            torch.save(
                {
                    "signature": fold_signature,
                    "signature_payload": fold_signature_payload,
                    "state_dict": model.state_dict(),
                },
                fold_model_path,
            )

        intercept, coef_by_feature = extract_model_coefficients(model, FEATURE_NAMES)
        coefficient_rows.append(
            {
                "heldout_dataset": heldout,
                "term": "intercept",
                "coefficient": intercept,
            }
        )
        for feat_name in FEATURE_NAMES:
            coefficient_rows.append(
                {
                    "heldout_dataset": heldout,
                    "term": feat_name,
                    "coefficient": coef_by_feature[feat_name],
                }
            )

        alphas = predict_alpha(model, X_test, device)
        alpha_map = {qid: float(alpha) for qid, alpha in zip(test_qids, alphas)}
        alpha_rows.extend(
            {"dataset": heldout, "query_id": qid, "alpha": float(alpha)}
            for qid, alpha in zip(test_qids, alphas)
        )

        ds_cache = dataset_cache_map[heldout]
        bm25_results = ds_cache["bm25_results"]
        dense_results = ds_cache["dense_results"]
        qrels = ds_cache["qrels"]

        sparse_scores, dense_scores = bm25_and_dense_to_score_maps(bm25_results, dense_results)
        static_rrf_scores = apply_static_rrf(bm25_results, dense_results, rrf_k=rrf_k)
        dynamic_scores = apply_dynamic_wrrf(bm25_results, dense_results, alpha_map, rrf_k=rrf_k)

        dense_only_ndcg = calculate_dataset_ndcg_at_k(dense_scores, qrels, ndcg_k)
        sparse_only_ndcg = calculate_dataset_ndcg_at_k(sparse_scores, qrels, ndcg_k)
        static_rrf_ndcg = calculate_dataset_ndcg_at_k(static_rrf_scores, qrels, ndcg_k)
        dynamic_wrrf_ndcg = calculate_dataset_ndcg_at_k(dynamic_scores, qrels, ndcg_k)

        print(f"  Dense only   NDCG@{ndcg_k}: {dense_only_ndcg:.4f}")
        print(f"  Sparse only  NDCG@{ndcg_k}: {sparse_only_ndcg:.4f}")
        print(f"  Static RRF   NDCG@{ndcg_k}: {static_rrf_ndcg:.4f}")
        print(f"  Dynamic wRRF NDCG@{ndcg_k}: {dynamic_wrrf_ndcg:.4f}")

        fold_results.append(
            {
                "dataset": heldout,
                "dense_only_ndcg": dense_only_ndcg,
                "sparse_only_ndcg": sparse_only_ndcg,
                "static_rrf_ndcg": static_rrf_ndcg,
                "dynamic_wrrf_ndcg": dynamic_wrrf_ndcg,
                "fold_model_path": fold_model_path,
            }
        )

    print("\n[4/4] Writing benchmark outputs and plots ...")

    per_dataset_csv = os.path.join(benchmark_dir, "per_dataset_results.csv")
    macro_csv = os.path.join(benchmark_dir, "loodo_macro_summary.csv")
    alpha_csv = os.path.join(benchmark_dir, "predicted_alphas.csv")
    fold_coefficients_csv = os.path.join(benchmark_dir, "fold_logistic_coefficients.csv")
    fold_norm_csv = os.path.join(benchmark_dir, "fold_normalization_stats.csv")

    save_per_dataset_results(fold_results, per_dataset_csv)
    save_macro_summary(fold_results, macro_csv)
    save_fold_coefficients_csv(coefficient_rows, fold_coefficients_csv)
    save_fold_normalization_stats_csv(fold_norm_rows, fold_norm_csv)

    with open(alpha_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "query_id", "alpha"])
        for row in alpha_rows:
            writer.writerow([row["dataset"], row["query_id"], f"{row['alpha']:.6f}"])

    save_plots(fold_results, alpha_rows, plots_dir)

    print("\n" + "=" * 72)
    print("LOODO supervised routing benchmark completed.")
    print(f"Per-query feature cache (pkl): {feature_cache_pkl}")
    print(f"Per-query feature cache (csv): {feature_cache_csv}")
    print(f"Fold models directory        : {model_cache_dir}")
    print(f"Per-dataset results CSV      : {per_dataset_csv}")
    print(f"Macro summary CSV            : {macro_csv}")
    print(f"Predicted alphas CSV         : {alpha_csv}")
    print(f"Fold coefficients CSV        : {fold_coefficients_csv}")
    print(f"Fold normalization stats CSV : {fold_norm_csv}")
    print(f"Plots directory              : {plots_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
