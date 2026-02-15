import os
import json
import pickle
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from utils import load_config, ensure_dir

def main():
    config = load_config()
    paths = config['paths']
    
    print("Building BM25 index...")
    
    tokenized_file = paths['tokenized_corpus']
    output_file = paths['bm25_index']
    ensure_dir(os.path.dirname(output_file))
    
    tokenized_corpus = []
    doc_ids = []
    
    print("Loading preprocessed tokens...")
    with open(tokenized_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading documents"):
            entry = json.loads(line)
            tokenized_corpus.append(entry['tokens'])
            doc_ids.append(entry['_id'])

    print("Computing BM25 scores...")
    bm25 = BM25Okapi(tokenized_corpus)

    with open(output_file, "wb") as f:
        pickle.dump({"model": bm25, "doc_ids": doc_ids}, f)
    
    print(f"BM25 index saved to {output_file}")

if __name__ == "__main__":
    main()