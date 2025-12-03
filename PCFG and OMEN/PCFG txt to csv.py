import pandas as pd

# Input: the PCFG/OMEN output txt file
input_file = "pcfg_scored.txt"
output_file = "pcfg_omen_results.csv"

# Read the lines
data = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):  # skip empty lines or headers
            continue
        parts = line.split("\t")
        if len(parts) == 4:
            password, typ, pcfg_prob, omen_level = parts
            data.append({
                "password": password,
                "type": typ,
                "PCFG_probability": float(pcfg_prob),
                "OMEN_level": int(omen_level)
            })

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} entries to {output_file}")
print(df.head())
