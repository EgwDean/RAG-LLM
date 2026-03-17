"""Plot learned dynamic routing sigmoid curves per dataset.

Reads:
    data/results/<model_name>/best_dynamic_params.csv

Writes:
    data/results/<model_name>/dataset_sigmoid_curves.png
"""

import argparse
import csv
import math
import os
from collections import defaultdict

import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "Dataset",
    "Metric",
    "Best k",
    "Best center",
}

METRIC_ORDER = ("JSD", "KLD", "CE", "RANDOM")
METRIC_COLORS = {
    "JSD": "#1f77b4",  # blue
    "KLD": "#ff7f0e",  # orange
    "CE": "#2ca02c",   # green
    "RANDOM": "#d62728",  # red
}

METRIC_STYLES = {
    "JSD": {"linestyle": "-", "linewidth": 3.0, "alpha": 0.7},
    "KLD": {"linestyle": "--", "linewidth": 2.5, "alpha": 0.8},
    "CE": {"linestyle": ":", "linewidth": 2.5, "alpha": 0.9},
    "RANDOM": {"linestyle": "-.", "linewidth": 2.0, "alpha": 1.0},
}


def centered_sigmoid(z_values, slope, center):
    """Centered sigmoid used by the retrieval pipeline.

    alpha = 1 / (1 + exp(-k * (z - center)))
    """
    return 1.0 / (1.0 + np.exp(-slope * (z_values - center)))


def load_config(config_path):
    """Load YAML config used by the pipeline."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def model_short_name(full_name):
    """Convert model path to the short folder name used in results paths."""
    return (full_name or "").split("/")[-1]


def read_best_params(csv_path):
    """Read best parameter rows grouped by dataset and metric."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    by_dataset = defaultdict(dict)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - header
        if missing:
            raise ValueError(
                "Missing required columns in CSV: " + ", ".join(sorted(missing))
            )

        for row in reader:
            dataset = (row.get("Dataset") or "").strip()
            metric = (row.get("Metric") or "").strip().upper()
            if not dataset or metric not in METRIC_ORDER:
                continue

            k = float(row["Best k"])
            center = float(row["Best center"])
            by_dataset[dataset][metric] = {"k": k, "center": center}

    if not by_dataset:
        raise ValueError("No valid dataset/metric rows found in input CSV.")

    return by_dataset


def build_figure(by_dataset, output_path):
    """Create a dynamic subplot grid and save to output_path."""
    datasets = sorted(by_dataset.keys())
    n_datasets = len(datasets)

    n_cols = 3
    n_rows = math.ceil(n_datasets / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.6 * n_rows))
    axes = np.array(axes).reshape(-1)

    z_values = np.linspace(-3.0, 3.0, 200)

    for idx, dataset in enumerate(datasets):
        ax = axes[idx]
        metric_params = by_dataset[dataset]

        for metric in METRIC_ORDER:
            if metric not in metric_params:
                continue
            k = metric_params[metric]["k"]
            center = metric_params[metric]["center"]
            alpha_values = centered_sigmoid(z_values, k, center)
            style = METRIC_STYLES[metric]
            ax.plot(
                z_values,
                alpha_values,
                color=METRIC_COLORS[metric],
                linestyle=style["linestyle"],
                linewidth=style["linewidth"],
                alpha=style["alpha"],
                label=f"{metric} (k={k:g}, c={center:g})",
            )

        # Static reference guides
        ax.axhline(0.5, color="#777777", linestyle="--", linewidth=1.0, alpha=0.5)
        ax.axvline(0.0, color="#777777", linestyle="--", linewidth=1.0, alpha=0.5)

        ax.set_title(dataset)
        ax.set_xlim(-3.0, 3.0)
        ax.set_ylim(0.0, 1.0)
        ax.set_xlabel("Query Entropy Z-Score")
        ax.set_ylabel("alpha Weight (0=Dense, 1=BM25)")
        ax.grid(alpha=0.2)

    # Hide unused subplots if grid has extra cells
    for idx in range(n_datasets, len(axes)):
        axes[idx].axis("off")

    # Single master legend for the entire figure
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color=METRIC_COLORS[m],
            linestyle=METRIC_STYLES[m]["linestyle"],
            linewidth=METRIC_STYLES[m]["linewidth"],
            alpha=METRIC_STYLES[m]["alpha"],
            label=m,
        )
        for m in METRIC_ORDER
    ]
    fig.legend(legend_handles, [m for m in METRIC_ORDER], loc="upper center", ncol=3, frameon=False)

    fig.suptitle("Learned Dynamic Routing Curves Per Dataset", y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.965])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot per-dataset learned sigmoid routing curves from best_dynamic_params.csv"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file (default: config.yaml)",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model folder override under data/results. If omitted, uses config embeddings.model_name.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg_model_name = cfg.get("embeddings", {}).get("model_name")
    if not cfg_model_name and args.model_name is None:
        raise ValueError(
            "Missing embeddings.model_name in config and no --model-name override was provided."
        )

    selected_model = args.model_name or model_short_name(cfg_model_name)
    results_root = cfg.get("paths", {}).get("results_folder", "data/results")

    base_dir = os.path.join(results_root, selected_model)
    csv_path = os.path.join(base_dir, "best_dynamic_params.csv")
    output_path = os.path.join(base_dir, "dataset_sigmoid_curves.png")

    by_dataset = read_best_params(csv_path)
    build_figure(by_dataset, output_path)

    print(f"Loaded datasets: {len(by_dataset)}")
    print(f"Selected model : {selected_model}")
    print(f"Input CSV      : {csv_path}")
    print(f"Output PNG     : {output_path}")


if __name__ == "__main__":
    main()
