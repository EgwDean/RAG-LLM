import os
import json
import csv
import yaml
import gc
from beir import util
from beir.datasets.data_loader import GenericDataLoader

CONFIG_PATH = "config.yaml"
DATA_FOLDER = os.path.join("data", "datasets")
OUTPUT_FOLDER = os.path.join("data", "beir_merged")

def load_config():
    """Load configuration from config.yaml."""
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Configuration file not found at: {os.path.abspath(CONFIG_PATH)}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path):
    """Create directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def initialize_output_files(corpus_path, queries_path, qrels_path):
    """Initialize output files and write qrels header."""
    print("Initializing output files...")
    
    open(corpus_path, 'w', encoding='utf-8').close()
    open(queries_path, 'w', encoding='utf-8').close()
    
    ensure_dir(os.path.dirname(qrels_path))
    with open(qrels_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["query-id", "corpus-id", "score"])

def append_corpus_to_jsonl(corpus_dict, filepath, dataset_prefix):
    """Append corpus documents to JSONL file with dataset prefix."""
    if not corpus_dict: return
    
    print(f"  Writing {len(corpus_dict)} documents...")
    with open(filepath, 'a', encoding='utf-8') as f:
        for doc_id, doc in corpus_dict.items():
            new_id = f"{dataset_prefix}_{doc_id}"
            entry = {
                "_id": new_id,
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {})
            }
            json.dump(entry, f)
            f.write('\n')

def append_queries_to_jsonl(queries_dict, filepath, dataset_prefix):
    """Append queries to JSONL file with dataset prefix."""
    if not queries_dict: return
    
    print(f"  Writing {len(queries_dict)} queries...")
    with open(filepath, 'a', encoding='utf-8') as f:
        for q_id, q_text in queries_dict.items():
            new_id = f"{dataset_prefix}_{q_id}"
            entry = {
                "_id": new_id,
                "text": q_text,
                "metadata": {}
            }
            json.dump(entry, f)
            f.write('\n')

def append_qrels_to_tsv(qrels_dict, filepath, dataset_prefix):
    """Append relevance judgments to TSV file with dataset prefix."""
    if not qrels_dict: return

    print(f"  Writing qrels...")
    with open(filepath, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        for q_id, doc_map in qrels_dict.items():
            new_q_id = f"{dataset_prefix}_{q_id}"
            for doc_id, score in doc_map.items():
                new_doc_id = f"{dataset_prefix}_{doc_id}"
                writer.writerow([new_q_id, new_doc_id, int(score)])

def main():
    # 1. Config & Setup
    config = load_config()
    datasets_to_merge = config.get("datasets", [])
    
    if not datasets_to_merge:
        print("[ERROR] No datasets found in config.yaml")
    config = load_config()
    datasets_to_merge = config.get("datasets", [])
    
    if not datasets_to_merge:
        print("Error: No datasets found in config.yaml")
        return

    print(f"Datasets to merge: {datasets_to_merge}")
    ensure_dir(DATA_FOLDER)
    ensure_dir(OUTPUT_FOLDER)
    
    out_corpus = os.path.join(OUTPUT_FOLDER, "corpus.jsonl")
    out_queries = os.path.join(OUTPUT_FOLDER, "queries.jsonl")
    out_qrels = os.path.join(OUTPUT_FOLDER, "qrels", "test.tsv")
    
    initialize_output_files(out_corpus, out_queries, out_qrels)
    
    print("\nStarting merge process...")
    
    for dataset_name in datasets_to_merge:
        print(f"\nProcessing: {dataset_name}")
        dataset_path = os.path.join(DATA_FOLDER, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"  Downloading {dataset_name}...")
            url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
            try:
                util.download_and_unzip(url, DATA_FOLDER)
            except Exception as e:
                print(f"  Error downloading {dataset_name}: {e}")
                continue

        print(f"  Loading data...")
        try:
            loader = GenericDataLoader(dataset_path)
            if 'test' in loader.qrels_file: 
                 split = 'test'
            elif 'train' in loader.qrels_file:
                 split = 'train'
            else:
                 split = 'dev'
                 
            corpus, queries, qrels = loader.load(split=split)
            print(f"  Loaded split: {split}")
            
        except Exception as e:
            print(f"  Warning: Failed to load {dataset_name}, skipping. ({e})")
            continue

        append_corpus_to_jsonl(corpus, out_corpus, dataset_name)
        append_queries_to_jsonl(queries, out_queries, dataset_name)
        append_qrels_to_tsv(qrels, out_qrels, dataset_name)

        # Free memory after processing each dataset
        del corpus
        del queries
        del qrels
        gc.collect() 

    print(f"\nMerge complete. Output saved to: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()