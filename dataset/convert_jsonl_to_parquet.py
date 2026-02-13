
import pandas as pd
import json
import os

def convert_jsonl_to_parquet(jsonl_path, parquet_path):
    print(f"Reading {jsonl_path}...")
    data = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    # Create data item with required columns
                    # RLHF expects: prompt, optional response (reference), optional ground_truth
                    # SFT expects: prompt, response
                    
                    # Typhoon data has 'prompt' and 'response'.
                    # We copy 'response' to 'ground_truth' for the Qwen Judge.
                    
                    new_item = {
                        'prompt': item['prompt'],
                        'response': item['response'],       # For SFT and Reference
                        'ground_truth': item['response']    # For Qwen Judge
                    }
                    data.append(new_item)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid line: {e}")

    df = pd.DataFrame(data)
    print(f"Captured {len(df)} records.")
    
    # Save to Parquet
    print(f"Saving to {parquet_path}...")
    df.to_parquet(parquet_path, index=False)
    print("Done!")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    jsonl_file = os.path.join(base_dir, "typhoon_data.jsonl")
    parquet_file = os.path.join(base_dir, "typhoon_data.parquet")
    
    if os.path.exists(jsonl_file):
        convert_jsonl_to_parquet(jsonl_file, parquet_file)
    else:
        print(f"Error: {jsonl_file} not found.")
