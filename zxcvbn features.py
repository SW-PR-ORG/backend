import pandas as pd
import numpy as np
from zxcvbn import zxcvbn
import os

# -----------------------------
# 1️⃣ Function to get zxcvbn features
# -----------------------------
def get_zxcvbn_features(password):
    """
    Returns a 3-tuple:
    (score, guesses, crack_time_seconds)
    """
    try:
        res = zxcvbn(str(password))
        return (
            res.get('score', np.nan),
            res.get('guesses', np.nan),
            res.get('crack_times_seconds', {}).get('offline_slow_hashing_1e4_per_second', np.nan)
        )
    except Exception:
        return (np.nan, np.nan, np.nan)

# -----------------------------
# 2️⃣ File paths
# -----------------------------
input_csv = "E:/New SW PR/backend/Ignored datasets/rockyou_dataset_cleaned.csv"
output_csv = "Ignored dataset/sample_rockyou_zxcvbn_features.csv"

# Ensure output folder exists
os.makedirs(os.path.dirname(output_csv), exist_ok=True)

# -----------------------------
# 3️⃣ Chunked processing
# -----------------------------
chunk_size = 100_000  # adjust based on memory
dfs = []
total_rows = 0

for i, chunk in enumerate(pd.read_csv(input_csv, 
                                      chunksize=chunk_size, 
                                      encoding="utf-8", 
                                      keep_default_na=False)):
    print(f"Processing chunk {i+1}...")
    
    # Compute zxcvbn features
    features = [get_zxcvbn_features(p) for p in chunk['password'].astype(str)]
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features, columns=[
        'zxcvbn_score', 'zxcvbn_guesses', 'zxcvbn_crack_time_seconds'
    ])
    
    # Compute log10 of guesses safely
    features_df['zxcvbn_guesses'] = pd.to_numeric(features_df['zxcvbn_guesses'], errors='coerce')
    features_df['zxcvbn_log10_guesses'] = np.log10(features_df['zxcvbn_guesses'].clip(lower=1))
    
    # Combine with original chunk
    combined = pd.concat([chunk.reset_index(drop=True), features_df], axis=1)
    
    dfs.append(combined)
    total_rows += len(chunk)
    print(f"Chunk {i+1} done, total rows processed: {total_rows}")

# -----------------------------
# 4️⃣ Concatenate all chunks and save
# -----------------------------
df_final = pd.concat(dfs, axis=0)
df_final.to_csv(output_csv, index=False)

print(f"All done! Zxcvbn features saved to: {output_csv}")
