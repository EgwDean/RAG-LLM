import os
import json
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from utils import load_config

def main():
    config = load_config()
    paths = config['paths']
    
    print("Preprocessing corpus...")
    
    stemmer = SnowballStemmer(config['preprocessing']['stemmer_language'])
    corpus_path = paths['corpus']
    output_path = paths['tokenized_corpus']
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        with open(corpus_path, 'r', encoding='utf-8') as f_in:
            total_lines = sum(1 for _ in open(corpus_path, 'r', encoding='utf-8'))
            f_in.seek(0)
            
            for line in tqdm(f_in, total=total_lines, desc="Stemming documents"):
                doc = json.loads(line)
                
                raw_text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
                tokens = [stemmer.stem(t) for t in raw_text.lower().split()]
                
                output_entry = {
                    "_id": doc["_id"],
                    "tokens": tokens
                }
                json.dump(output_entry, f_out)
                f_out.write('\n')

    print(f"Tokenized corpus saved to {output_path}")

if __name__ == "__main__":
    main()