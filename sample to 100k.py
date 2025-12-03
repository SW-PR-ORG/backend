import csv
import random
import pandas as pd

INPUT_CSV = "rockyou_dataset.csv"         # your full dataset
OUTPUT_CSV = "rockyou_100k.csv"   # sampled dataset
SAMPLE_SIZE = 100_000

def reservoir_sample(input_path, output_path, sample_size):
    # open input CSV
    with open(input_path, newline='', encoding='utf‑8', errors='ignore') as f_in:
        reader = csv.reader(f_in)
        header = next(reader)  # assume first line is header
        
        # initialize reservoir with first sample_size rows
        reservoir = []
        for i, row in enumerate(reader, start=1):
            if i <= sample_size:
                reservoir.append(row)
            else:
                # random replace logic
                j = random.randrange(i + 1)
                if j < sample_size:
                    reservoir[j] = row
        
    # write sample to new csv
    with open(output_path, 'w', newline='', encoding='utf‑8') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerows(reservoir)
    print(f"Saved {len(reservoir)} sampled rows to {output_path}")

if __name__ == "__main__":
    reservoir_sample(INPUT_CSV, OUTPUT_CSV, SAMPLE_SIZE)
    
    # (Optionally) read with pandas to inspect
    df_sample = pd.read_csv(OUTPUT_CSV)
    print(df_sample.shape)
    print(df_sample.head())
