# RAG-LLM Hybrid Retrieval Benchmark

This repository benchmarks sparse, dense, and hybrid retrieval on BEIR datasets with a two-phase workflow:

1. Preprocess and cache all heavy artifacts once.
2. Run retrieval and evaluation repeatedly on the cached data.

The current hybrid logic uses supervised query-level routing in a leave-one-dataset-out (LOODO) setup.

## Methods evaluated

For each configured dataset, the evaluation script reports:

1. BM25 Only
2. Dense Only
3. RRF (static)
4. Dynamic wRRF (alpha predicted by a PyTorch logistic regression model)

All methods are scored with NDCG@k (default k=10).

## Repository layout

```text
config.yaml
requirements.txt
src/
  pipeline.py      # legacy reference pipeline (kept for comparison)
  utils.py
  download.py      # dataset downloader
  preprocess.py    # preprocessing/cache builder
  retrieve_and_evaluate.py  # LOODO supervised benchmark
  correct.py       # migrate old cached files from results -> processed_data
  fix.py           # one-time stale-cache cleanup for new retrieval logic
data/
  datasets/        # raw BEIR datasets
  processed_data/  # reusable cache artifacts per model/dataset
  results/         # model-level summaries and charts
```

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Configuration

Edit config.yaml to choose:

1. Active datasets (all uncommented entries are processed/evaluated).
2. Embedding model and batch size.
3. Retrieval/evaluation settings.
4. Supervised routing feature/training settings.

Important sections:

```yaml
datasets:
  - scifact
  - nfcorpus

paths:
  datasets_folder: "data/datasets"
  results_folder: "data/results"
  processed_folder: "data/processed_data"

embeddings:
  model_name: "BAAI/bge-m3"
  batch_size: 64

benchmark:
  top_k: 100
  ndcg_k: 10
  rrf:
    k: 60

supervised_routing:
  overlap_k: 100
  epsilon: 1.0e-9
  C: 1.0
  optimizer: "adam"
  learning_rate: 0.05
```

## Run order

1. Download datasets

```bash
python src/download.py
```

2. Build caches (corpus/query files, BM25 index, frequency index, embeddings)

```bash
python src/preprocess.py
```

3. Optional migration for previously generated caches

```bash
python src/correct.py
```

4. Optional one-time stale cleanup before new retrieval logic

```bash
python src/fix.py
```

5. Run LOODO retrieval + evaluation

```bash
python src/retrieve_and_evaluate.py
```

## Supervised Routing Details

The evaluation stage applies the following:

1. Build per-query feature/label rows from cached sparse+dense retrieval.
2. Train one logistic regression model per LOODO fold.
3. Predict query-level alpha values on held-out datasets.
4. Fuse sparse+dense rankings with dynamic wRRF and compare against baselines.

## Outputs

The supervised benchmark writes files under data/results/<model>/supervised_routing/:

1. per_dataset_results.csv
2. loodo_macro_summary.csv
3. predicted_alphas.csv
4. fold_models/*.pt
5. plots/*.png

Processed artifacts are cached under data/processed_data/<model>/<dataset>/ and reused across runs.

## Notes

1. CPU and CUDA are both supported.
2. Preprocessing uses multiprocessing for corpus tokenization.
3. Dense retrieval uses chunking controls from config for memory safety.
