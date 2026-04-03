"""probe_mlp_on_query_embeddings.py

Quick probe: can a simple MLP predict query-level alpha directly from query
embeddings and beat static RRF?

Method:
1) Load cached sparse+dense retrieval outputs per dataset.
2) Build soft labels with the same logic used in the current router:
     alpha_label = 0.5 * (((sparse_ndcg - dense_ndcg) / (sparse_ndcg + dense_ndcg + eps)) + 1)
3) Train a small MLP on query embeddings -> alpha_label.
4) Evaluate static RRF vs dynamic wRRF (with predicted alpha) on held-out
   per-dataset query splits.

This is intentionally lightweight for fast feasibility checks before investing
in larger embedding-router training.
"""

import argparse
import csv
import os
import random
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
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
from src.retrieve_and_evaluate import (
    dataset_seed_offset,
    ensure_retrieval_results_cached,
    evaluate_benchmark_methods_for_qids,
    load_config,
    load_pickle,
    query_ndcg_at_k,
    set_global_seed,
)
from src.utils import ensure_dir, get_config_path, model_short_name


@dataclass
class DatasetSplit:
    dataset: str
    train_qids: list
    test_qids: list
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor


class BasicMLP(nn.Module):
    """Small MLP regressor returning alpha logits."""

    def __init__(self, in_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_test_qid_split(qids, train_fraction, seed, shuffle=True):
    """Deterministic per-dataset split for quick probing."""
    if len(qids) < 2:
        raise ValueError("Need at least 2 queries to create train/test split.")

    if not (0.0 < train_fraction < 1.0):
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}.")

    order = list(qids)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)

    n_train = int(len(order) * train_fraction)
    n_train = max(1, min(len(order) - 1, n_train))
    return order[:n_train], order[n_train:]


def build_alpha_labels(ds_cache, qids, ndcg_k, epsilon):
    """Build soft labels using the exact current supervised-routing formula."""
    labels = {}
    for qid in qids:
        qrels = ds_cache["qrels"].get(qid, {})
        sparse = ds_cache["bm25_results"].get(qid, [])
        dense = ds_cache["dense_results"].get(qid, [])

        sparse_ndcg = query_ndcg_at_k(sparse, qrels, ndcg_k)
        dense_ndcg = query_ndcg_at_k(dense, qrels, ndcg_k)

        alpha = 0.5 * (((sparse_ndcg - dense_ndcg) / (sparse_ndcg + dense_ndcg + epsilon)) + 1.0)
        labels[qid] = float(np.clip(alpha, 0.0, 1.0))
    return labels


def make_dataset_split(dataset_name, ds_cache, train_fraction, seed, ndcg_k, epsilon, normalize_inputs=True):
    """Build train/test tensors for one dataset."""
    query_vectors = torch.load(ds_cache["paths"]["query_vectors_pt"], weights_only=True).float()
    query_ids = load_pickle(ds_cache["paths"]["query_ids_pkl"])

    if len(query_ids) != query_vectors.shape[0]:
        raise ValueError(
            f"Dataset {dataset_name}: mismatch query_ids ({len(query_ids)}) vs "
            f"query_vectors ({query_vectors.shape[0]})."
        )

    available_qids = [qid for qid in query_ids if qid in ds_cache["qrels"]]
    if len(available_qids) < 2:
        raise ValueError(f"Dataset {dataset_name}: not enough labeled queries for split.")

    label_map = build_alpha_labels(ds_cache, available_qids, ndcg_k=ndcg_k, epsilon=epsilon)
    id_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}

    split_seed = seed + dataset_seed_offset(dataset_name)
    train_qids, test_qids = train_test_qid_split(
        available_qids,
        train_fraction=train_fraction,
        seed=split_seed,
        shuffle=True,
    )

    train_idx = [id_to_idx[qid] for qid in train_qids]
    test_idx = [id_to_idx[qid] for qid in test_qids]

    X_train = query_vectors[train_idx]
    X_test = query_vectors[test_idx]

    if normalize_inputs:
        X_train = F.normalize(X_train, p=2, dim=1)
        X_test = F.normalize(X_test, p=2, dim=1)

    y_train = torch.tensor([label_map[qid] for qid in train_qids], dtype=torch.float32)
    y_test = torch.tensor([label_map[qid] for qid in test_qids], dtype=torch.float32)

    return DatasetSplit(
        dataset=dataset_name,
        train_qids=train_qids,
        test_qids=test_qids,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
    )


def fit_mlp(split, device, epochs, batch_size, learning_rate, weight_decay, hidden_dim, dropout):
    """Train one dataset-specific MLP and return predicted test alphas."""
    model = BasicMLP(
        in_dim=split.X_train.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = DataLoader(
        TensorDataset(split.X_train, split.y_train),
        batch_size=min(batch_size, len(split.X_train)),
        shuffle=True,
    )

    model.train()
    pbar = tqdm(range(1, epochs + 1), desc=f"  Train MLP [{split.dataset}]", dynamic_ncols=True)
    for _ in pbar:
        total_loss = 0.0
        n_items = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * len(xb)
            n_items += len(xb)

        mean_loss = total_loss / max(1, n_items)
        pbar.set_postfix(loss=f"{mean_loss:.5f}")

    model.eval()
    with torch.no_grad():
        logits = model(split.X_test.to(device))
        pred = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    return pred


def write_outputs(output_dir, per_dataset_rows, ndcg_k):
    """Persist per-dataset and macro summary CSVs."""
    ensure_dir(output_dir)
    per_dataset_csv = os.path.join(output_dir, "per_dataset_results.csv")
    macro_csv = os.path.join(output_dir, "macro_summary.csv")

    with open(per_dataset_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "train_queries",
                "test_queries",
                f"static_rrf_ndcg@{ndcg_k}",
                f"dynamic_wrrf_mlp_ndcg@{ndcg_k}",
                "delta_dynamic_minus_static",
                "mean_pred_alpha",
            ]
        )
        for row in per_dataset_rows:
            writer.writerow(
                [
                    row["dataset"],
                    row["train_queries"],
                    row["test_queries"],
                    f"{row['static_rrf_ndcg']:.6f}",
                    f"{row['dynamic_wrrf_ndcg']:.6f}",
                    f"{row['delta']:.6f}",
                    f"{row['mean_pred_alpha']:.6f}",
                ]
            )

    static_vals = [r["static_rrf_ndcg"] for r in per_dataset_rows]
    dynamic_vals = [r["dynamic_wrrf_ndcg"] for r in per_dataset_rows]
    macro_static = float(np.mean(static_vals)) if static_vals else 0.0
    macro_dynamic = float(np.mean(dynamic_vals)) if dynamic_vals else 0.0

    with open(macro_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow([f"macro_static_rrf_ndcg@{ndcg_k}", f"{macro_static:.6f}"])
        writer.writerow([f"macro_dynamic_wrrf_mlp_ndcg@{ndcg_k}", f"{macro_dynamic:.6f}"])
        writer.writerow(["macro_delta_dynamic_minus_static", f"{(macro_dynamic - macro_static):.6f}"])

    return per_dataset_csv, macro_csv, macro_static, macro_dynamic


def main():
    parser = argparse.ArgumentParser(
        description="Quick probe: MLP on query embeddings for alpha prediction and wRRF comparison."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config file.")
    parser.add_argument("--train-fraction", type=float, default=0.8, help="Per-dataset train fraction.")
    parser.add_argument("--epochs", type=int, default=25, help="MLP training epochs.")
    parser.add_argument("--batch-size", type=int, default=256, help="MLP training batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="MLP learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="MLP AdamW weight decay.")
    parser.add_argument("--hidden-dim", type=int, default=256, help="MLP first hidden size.")
    parser.add_argument("--dropout", type=float, default=0.1, help="MLP dropout.")
    parser.add_argument(
        "--no-l2-normalize",
        action="store_true",
        help="Disable L2 normalization on input query embeddings.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training even if CUDA is available.",
    )
    args = parser.parse_args()

    u.CONFIG_PATH = args.config
    cfg = load_config()

    datasets = cfg.get("datasets", [])
    if not datasets:
        raise ValueError("No datasets configured.")

    routing_cfg = cfg.get("supervised_routing", {})
    seed = int(routing_cfg.get("seed", 42))
    epsilon = float(routing_cfg.get("epsilon", 1.0e-8))
    ndcg_k = int(cfg.get("benchmark", {}).get("ndcg_k", 10))
    rrf_k = int(cfg.get("benchmark", {}).get("rrf", {}).get("k", 60))

    set_global_seed(seed)

    use_cuda = bool(routing_cfg.get("use_cuda_if_available", True)) and (not args.cpu)
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    short_model = model_short_name(cfg["embeddings"]["model_name"])
    results_root = get_config_path(cfg, "results_folder", "data/results")
    output_dir = os.path.join(results_root, short_model, "embedding_mlp_probe")
    ensure_dir(output_dir)

    print("=" * 72)
    print("Embedding-MLP alpha probe")
    print(f"Device              : {device}")
    print(f"Embedding model     : {cfg['embeddings']['model_name']}")
    print(f"Datasets ({len(datasets)})      : {', '.join(datasets)}")
    print(f"Train fraction      : {args.train_fraction}")
    print(f"Epochs              : {args.epochs}")
    print(f"Label logic         : same soft-label formula as current router")
    print("=" * 72)

    print("\n[1/3] Loading cached retrieval artifacts per dataset ...")
    dataset_cache_map = {}
    for dataset_name in datasets:
        print(f"  - {dataset_name}")
        dataset_cache_map[dataset_name] = ensure_retrieval_results_cached(dataset_name, cfg, device)

    print("\n[2/3] Training one embedding-MLP per dataset and evaluating ...")
    per_dataset_rows = []
    for dataset_name in datasets:
        ds_cache = dataset_cache_map[dataset_name]
        split = make_dataset_split(
            dataset_name=dataset_name,
            ds_cache=ds_cache,
            train_fraction=args.train_fraction,
            seed=seed,
            ndcg_k=ndcg_k,
            epsilon=epsilon,
            normalize_inputs=not args.no_l2_normalize,
        )

        pred_alpha = fit_mlp(
            split=split,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        )
        alpha_map = {
            qid: float(alpha)
            for qid, alpha in zip(split.test_qids, pred_alpha)
        }

        metrics = evaluate_benchmark_methods_for_qids(
            bm25_results=ds_cache["bm25_results"],
            dense_results=ds_cache["dense_results"],
            qrels=ds_cache["qrels"],
            ndcg_k=ndcg_k,
            rrf_k=rrf_k,
            query_ids=split.test_qids,
            alpha_map=alpha_map,
        )

        static_ndcg = float(metrics["static_rrf_ndcg"])
        dynamic_ndcg = float(metrics["dynamic_wrrf_ndcg"])

        row = {
            "dataset": dataset_name,
            "train_queries": len(split.train_qids),
            "test_queries": len(split.test_qids),
            "static_rrf_ndcg": static_ndcg,
            "dynamic_wrrf_ndcg": dynamic_ndcg,
            "delta": dynamic_ndcg - static_ndcg,
            "mean_pred_alpha": float(np.mean(pred_alpha)) if len(pred_alpha) else 0.0,
        }
        per_dataset_rows.append(row)

        print(
            f"  {dataset_name:<18} | static={static_ndcg:.4f} | "
            f"dynamic={dynamic_ndcg:.4f} | delta={row['delta']:+.4f}"
        )

    print("\n[3/3] Writing outputs ...")
    per_dataset_csv, macro_csv, macro_static, macro_dynamic = write_outputs(
        output_dir=output_dir,
        per_dataset_rows=per_dataset_rows,
        ndcg_k=ndcg_k,
    )

    print("\nCompleted embedding-MLP probe.")
    print(f"Per-dataset results : {per_dataset_csv}")
    print(f"Macro summary       : {macro_csv}")
    print(
        f"Macro static={macro_static:.6f} | "
        f"macro dynamic={macro_dynamic:.6f} | "
        f"delta={macro_dynamic - macro_static:+.6f}"
    )


if __name__ == "__main__":
    main()
