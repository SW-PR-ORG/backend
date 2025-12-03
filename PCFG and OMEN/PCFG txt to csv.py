import pandas as pd

input_file = "pcfg_scored.txt"
output_file = "pcfg_omen_results.csv"

data = []
with open(input_file, "r", encoding="utf-16") as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"): 
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

df = pd.DataFrame(data)

df.to_csv(output_file, index=False)

print(f"Saved {len(df)} entries to {output_file}")
print(df.head())
