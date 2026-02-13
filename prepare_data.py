
import os
import json
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --- Configuration ---
SEED = 42

def process_local_dataset(file_path):
    print(f"Processing local dataset: {file_path}...")
    data = []
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: File {file_path} not found.")
        return pd.DataFrame()
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # Ensure requirement fields
                if 'prompt' in item and 'response' in item:
                    # Construct reward_model structure for RL
                    # We can try to extract <answer>...</answer> for ground truth
                    ground_truth = ""
                    if "<answer>" in item['response']:
                        try:
                            ground_truth = item['response'].split("<answer>")[1].split("</answer>")[0].strip()
                        except:
                            pass
                    
                    data.append({
                        "prompt": item['prompt'],
                        "reward_model": {"ground_truth": ground_truth, "target": ground_truth},
                        "id": item.get('id', f"typhoon_{len(data)}"),
                        "source": "typhoon_custom"
                    })
                    
            except json.JSONDecodeError:
                continue
                
    return pd.DataFrame(data)

def save_parquet(df, filename):
    table = pa.Table.from_pandas(df)
    os.makedirs("data", exist_ok=True)
    pq.write_table(table, f"data/{filename}")
    print(f"Saved {len(df)} rows to data/{filename}")

if __name__ == "__main__":
    print("--- 1. Loading Typhoon Dataset (Comprehensive) ---")
    
    # 1. Load ONLY Typhoon Data
    typhoon_df = process_local_dataset("dataset/typhoon_data.jsonl")
    
    if len(typhoon_df) == 0:
        print("❌ Error: No data found in dataset/typhoon_data.jsonl")
        print("Please generate data using 'python dataset/generate_typhoon.py' first.")
        exit(1)

    print(f"Total Typhoon Data: {len(typhoon_df)} samples")
    
    # Shuffle
    typhoon_df = typhoon_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # 2. Split for RL (Train/Valid)
    # We use this data for RL (PPO) training
    valid_size = int(len(typhoon_df) * 0.05)
    if valid_size < 1: valid_size = 1 # Ensure at least 1 validation sample
    
    train_df = typhoon_df[:-valid_size]
    valid_df = typhoon_df[-valid_size:]
    
    save_parquet(train_df, "train.parquet")
    save_parquet(valid_df, "valid.parquet")
    save_parquet(valid_df.head(500), "valid_500.parquet") # Capped validation set

    # 3. Step for SFT (Extend Phase)
    # We use the SAME dataset for SFT to teach the policy the format
    print("\n--- 2. Preparing SFT Data ---")
    # Using the same split or full dataset? Usually SFT is done on the whole set or a train split.
    # Let's use the Training split to avoid leakage if we strictly separate, 
    # but often for these tasks we just use the whole corpus if validation is separate.
    # We'll use the 'train_df' for SFT to maintain rigorous train/val separation.
    save_parquet(train_df, "sft_train.parquet")

    print("\n✅ Dataset preparation complete! (Typhoon Only)")
    print(f"1. data/train.parquet      ({len(train_df)} samples) -> For RL Training")
    print(f"2. data/valid.parquet      ({len(valid_df)} samples) -> For RL Validation")
    print(f"3. data/sft_train.parquet  ({len(train_df)} samples) -> For Supervised Fine-Tuning")
