import os
import json
import pickle
from collections import Counter
from tqdm import tqdm
from utils import load_config, ensure_dir

def main():
    config = load_config()
    paths = config['paths']
    
    print("Building frequency index...")
    
    tokenized_file = paths['tokenized_corpus']
    output_file = paths['freq_index']
    ensure_dir(os.path.dirname(output_file))
    
    global_counter = Counter()
    total_tokens = 0
    
    with open(tokenized_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Counting tokens"):
            entry = json.loads(line)
            tokens = entry['tokens']
            
            global_counter.update(tokens)
            total_tokens += len(tokens)

    with open(output_file, "wb") as f:
        pickle.dump({"counts": dict(global_counter), "total_tokens": total_tokens}, f)
    
    print(f"Frequency index saved to {output_file}")

if __name__ == "__main__":
    main()