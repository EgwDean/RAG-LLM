"""Analyze dynamic wRRF parameter winners and derive zero-shot consensus.

Reads:
    data/results/<model_name>/best_dynamic_params.csv

Writes:
    data/results/<model_name>/parameter_distributions.png

Prints:
    YAML-formatted zero-shot parameter summary for JSD/KLD/CE.
    Dataset acronym mapping.
"""

import argparse
import csv
import os
import statistics
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED_COLUMNS = {
    "Dataset",
    "Metric",
    "Best max_df",
    "Best k",
    "Best center",
    "Best rrf_k",
    "Best NDCG@10",
}

METRICS = ("JSD", "KLD", "CE")
PARAMS = (
    ("Best max_df", "max_df"),
    ("Best rrf_k", "rrf_k"),
    ("Best k", "k"),
    ("Best center", "center"),
)


def to_float(value, col_name):
    """Convert a CSV field to float with context-rich errors."""
    try:
        return float(value)
    except Exception as exc:
        raise ValueError(f"Could not parse '{value}' as float in column '{col_name}'.") from exc


def to_int(value, col_name):
    """Convert a CSV field to int with context-rich errors."""
    try:
        return int(float(value))
    except Exception as exc:
        raise ValueError(f"Could not parse '{value}' as int in column '{col_name}'.") from exc


def get_acronym(dataset_name):
    """Generate a short 2-3 letter acronym for chart labeling."""
    parts = dataset_name.replace("-", " ").replace("_", " ").split()
    if len(parts) > 1:
        return "".join(p[0].upper() for p in parts)[:3]
    return dataset_name[:3].upper()


def load_rows(csv_path):
    """Load and validate rows from best_dynamic_params.csv."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - header
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in CSV: {missing_str}")

        for raw in reader:
            metric = (raw["Metric"] or "").strip().upper()
            if metric not in METRICS:
                continue

            rows.append(
                {
                    "Dataset": (raw["Dataset"] or "").strip(),
                    "Metric": metric,
                    "Best max_df": to_float(raw["Best max_df"], "Best max_df"),
                    "Best rrf_k": to_int(raw["Best rrf_k"], "Best rrf_k"),
                    "Best k": to_float(raw["Best k"], "Best k"),
                    "Best center": to_float(raw["Best center"], "Best center"),
                    "Best NDCG@10": to_float(raw["Best NDCG@10"], "Best NDCG@10"),
                }
            )

    if not rows:
        raise ValueError("No valid rows found for metrics JSD/KLD/CE.")

    return rows


def mode_with_tiebreak(values, prefer):
    """Return mode with explicit tie-break behavior."""
    counts = Counter(values)
    top = max(counts.values())
    tied = [v for v, c in counts.items() if c == top]
    if prefer == "higher":
        return max(tied)
    if prefer == "lower":
        return min(tied)
    raise ValueError(f"Unknown tie-break preference: {prefer}")


def compute_consensus(rows):
    """Compute per-metric zero-shot consensus parameters."""
    by_metric = defaultdict(list)
    for row in rows:
        by_metric[row["Metric"]].append(row)

    consensus = {}
    for metric in METRICS:
        metric_rows = by_metric.get(metric, [])
        if not metric_rows:
            continue

        max_df_vals = [r["Best max_df"] for r in metric_rows]
        rrf_k_vals = [r["Best rrf_k"] for r in metric_rows]
        k_vals = [r["Best k"] for r in metric_rows]
        center_vals = [r["Best center"] for r in metric_rows]

        consensus[metric] = {
            "max_df": mode_with_tiebreak(max_df_vals, prefer="higher"),
            "rrf_k": mode_with_tiebreak(rrf_k_vals, prefer="lower"),
            "k": float(statistics.median(k_vals)),
            "center": float(statistics.median(center_vals)),
        }

    return consensus


def save_distribution_plot(rows, output_path):
    """Create a 2x2 chart with dataset acronyms labeled above the bars."""
    colors = {"JSD": "#1f77b4", "KLD": "#ff7f0e", "CE": "#2ca02c"}

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()

    # Track maximum height across all plots to scale the Y-axis consistently
    max_global_freq = 0

    for ax, (col_name, label) in zip(axes, PARAMS):
        values_by_metric = {
            metric: [r[col_name] for r in rows if r["Metric"] == metric]
            for metric in METRICS
        }

        unique_vals = sorted(set(v for vals in values_by_metric.values() for v in vals))
        x_positions = list(range(len(unique_vals)))
        width = 0.25

        local_max_freq = 0

        for idx, metric in enumerate(METRICS):
            heights = []
            ds_labels = []
            
            for v in unique_vals:
                # Find which datasets picked this value for this metric
                matching_datasets = [
                    get_acronym(r["Dataset"]) for r in rows 
                    if r["Metric"] == metric and r[col_name] == v
                ]
                freq = len(matching_datasets)
                heights.append(freq)
                ds_labels.append(", ".join(matching_datasets))
                
                if freq > local_max_freq:
                    local_max_freq = freq

            offset = (idx - 1) * width
            bars = ax.bar(
                [x + offset for x in x_positions],
                heights,
                width=width,
                label=metric,
                color=colors[metric],
            )

            # Add dataset acronym text above the bars
            for bar, label_text in zip(bars, ds_labels):
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.1,
                        label_text,
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        rotation=90,
                        color="#333333"
                    )

        if local_max_freq > max_global_freq:
            max_global_freq = local_max_freq

        ax.set_title(f"Optimal {label} Parameter")
        ax.set_xticks(x_positions)

        if label == "rrf_k":
            tick_labels = [str(int(v)) for v in unique_vals]
        else:
            tick_labels = [f"{v:g}" for v in unique_vals]

        ax.set_xticklabels(tick_labels)
        ax.set_ylabel("Number of Datasets")
        ax.grid(axis="y", alpha=0.2)

    # Scale all Y-axes uniformly, giving plenty of headroom for the rotated text
    for ax in axes:
        ax.set_ylim(0, max_global_freq + 2.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=11)
    fig.suptitle("Dynamic wRRF Parameter Distributions by Dataset", y=0.98, fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def print_yaml_summary(model_name, consensus, rows):
    """Print zero-shot consensus and dataset acronym mapping."""
    unique_datasets = sorted(set(r["Dataset"] for r in rows))
    
    print("\n--- Dataset Acronym Mapping ---")
    for ds in unique_datasets:
        print(f" {get_acronym(ds):>4} : {ds}")
        
    print("\n--- Final Zero-Shot Configuration ---")
    print("zero_shot_dynamic_wrrf:")
    for metric in METRICS:
        if metric not in consensus:
            continue
        params = consensus[metric]
        print(f"  {metric}:")
        print(f"    max_df: {params['max_df']:g}")
        print(f"    rrf_k: {params['rrf_k']}")
        print(f"    k: {params['k']:g}")
        print(f"    center: {params['center']:g}")
    print(f"  model_name: {model_name}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze best dynamic params and produce zero-shot consensus + plots."
    )
    parser.add_argument(
        "--model-name",
        default="bge-m3",
        help="Model folder name under data/results (default: bge-m3)",
    )
    args = parser.parse_args()

    base_dir = os.path.join("data", "results", args.model_name)
    csv_path = os.path.join(base_dir, "best_dynamic_params.csv")
    plot_path = os.path.join(base_dir, "parameter_distributions.png")

    rows = load_rows(csv_path)
    consensus = compute_consensus(rows)
    save_distribution_plot(rows, plot_path)

    print(f"Loaded rows: {len(rows)}")
    print(f"Saved plot : {plot_path}")
    print_yaml_summary(args.model_name, consensus, rows)


if __name__ == "__main__":
    main()