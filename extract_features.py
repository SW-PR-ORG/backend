import math
import re
import numpy as np
import pandas as pd
from collections import Counter
from itertools import tee
from zxcvbn import zxcvbn
import joblib
import subprocess
import os

# -----------------------------
# Load resources ONCE
# -----------------------------
def load_word_set(filename):
    # Get path 
    base_dir = os.path.dirname(os.path.abspath(__file__))  # features folder
    dicts_dir = os.path.join(base_dir, "dict")
    path = os.path.join(dicts_dir, filename)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return set(w.strip().lower() for w in f if w.strip())

# Load dictionaries
WORDS_SET = load_word_set("words_alpha.txt")
NAMES_SET = load_word_set("names.txt")

# Get path 
base_dir = os.path.dirname(os.path.abspath(__file__))  # features folder
pca_dir = os.path.join(base_dir, "PCA_data_cd")

# Load scaler and PCA artifacts
scaler = joblib.load(os.path.join(pca_dir, "entropy_scaler.joblib"))
pca = joblib.load(os.path.join(pca_dir, "entropy_pca.joblib"))
pca_min = joblib.load(os.path.join(pca_dir, "pca_min.joblib"))
pca_max = joblib.load(os.path.join(pca_dir, "pca_max.joblib"))

# -----------------------------
# Entropy functions
# -----------------------------
def shannon_entropy(pwd):
    if not pwd:
        return 0
    freq = {c: pwd.count(c)/len(pwd) for c in set(pwd)}
    return -sum(p * math.log2(p) for p in freq.values())

def bigram_entropy(pwd):
    pwd = str(pwd)
    if len(pwd) < 2:
        return 0
    bigrams = list(zip(pwd, pwd[1:]))
    freq = {bg: bigrams.count(bg)/len(bigrams) for bg in set(bigrams)}
    return -sum(p * math.log2(p) for p in freq.values())

sequences = ["abcdefghijklmnopqrstuvwxyz", "0123456789", "qwertyuiop", "asdfghjkl", "zxcvbnm"]
def pattern_entropy(pwd):
    pwd = pwd.lower()
    penalty = 0
    for seq in sequences:
        for i in range(len(seq)-2):
            if seq[i:i+3] in pwd:
                penalty += 1
    return max(0, shannon_entropy(pwd) - penalty*0.2)

keyboard_sequences = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
def keyboard_entropy(pwd):
    pwd = pwd.lower()
    penalty = 0
    for seq in keyboard_sequences:
        for i in range(len(seq)-2):
            if seq[i:i+3] in pwd:
                penalty += 1
    return max(0, shannon_entropy(pwd) - penalty*0.2)

# -----------------------------
# Dictionary / names
# -----------------------------
def dictionary_features(pwd):
    pwd_lower = pwd.lower()
    found_words = [w for w in WORDS_SET if w in pwd_lower and len(w) >= 3]
    contains_dict = int(len(found_words) > 0)
    longest_word = max((len(w) for w in found_words), default=0)
    coverage = sum(len(w) for w in found_words) / len(pwd) if pwd else 0
    return contains_dict, longest_word, coverage

def name_features(pwd):
    pwd_lower = pwd.lower()
    found_names = [n for n in NAMES_SET if n in pwd_lower and len(n) >= 3]
    contains_common_name = int(any(len(n) >= 4 for n in found_names))
    return contains_common_name

# -----------------------------
# OMEN / PCFG
# -----------------------------
import subprocess
import os
import tempfile
import math

def omen_log10_score(password):
    """
    Run OMEN PCFG scorer and return log10(OMEN level).
    Returns 0.0 if scoring fails.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        pcfg_dir = os.path.join(base_dir, "pcfg_cracker")

        # Create temporary password file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(password + "\n")
            temp_file = f.name

        try:
            cmd = [
                "python",
                "password_scorer.py",
                "-r",
                "full_rockyou_trained_cleaned_pcfg",
                "-i",
                temp_file,
            ]

            result = subprocess.run(
                cmd,
                cwd=pcfg_dir,
                capture_output=True,
                text=True,
                timeout=10,
            )
            print("=== RAW OMEN OUTPUT ===")
            print(result.stdout)
            print("=======================")

            # Parse output: password  TYPE  probability  OMEN_LEVEL
            if result.stdout:
                for line in result.stdout.strip().split("\n"):
                    if password in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            try:
                                omen_level = float(parts[3])
                                return math.log10(omen_level + 1)
                            except ValueError:
                                continue

            return 0.0

        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)

    except Exception:
        return 0.0


# -----------------------------
# Feature extraction
# -----------------------------
def extract_password_features(password: str) -> pd.DataFrame:
    pwd = password or ""
    length = len(pwd)

    # Basic counts
    num_upper = sum(c.isupper() for c in pwd)
    num_lower = sum(c.islower() for c in pwd)
    num_digits = sum(c.isdigit() for c in pwd)
    num_special_char = sum(not c.isalnum() for c in pwd)

    # Positional
    first_is_upper = int(pwd[:1].isupper())
    first_is_digit = int(pwd[:1].isdigit())
    first_is_special = int(bool(pwd[:1]) and not pwd[:1].isalnum())
    last_is_upper = int(pwd[-1:].isupper())
    last_is_digit = int(pwd[-1:].isdigit())
    last_is_special = int(bool(pwd[-1:]) and not pwd[-1:].isalnum())

    # Entropy
    sh_ent = shannon_entropy(pwd)
    length_adj_entropy = sh_ent * length
    bi_ent = bigram_entropy(pwd)
    pat_ent = pattern_entropy(pwd)
    key_ent = keyboard_entropy(pwd)

    # PCA
    ent_vec = np.array([[sh_ent, length_adj_entropy, bi_ent, pat_ent, key_ent]])
    ent_scaled = scaler.transform(ent_vec)
    pca_val = pca.transform(ent_scaled)[0, 0]
    combined_entropy_pca_norm = (pca_val - pca_min) / (pca_max - pca_min)

    # Spread
    digit_idx = [i for i,c in enumerate(pwd) if c.isdigit()]
    letter_idx = [i for i,c in enumerate(pwd) if c.isalpha()]
    special_idx = [i for i,c in enumerate(pwd) if not c.isalnum()]
    spread = lambda idx: (max(idx)-min(idx))/length if len(idx)>1 else 0
    digit_spread = spread(digit_idx)
    letter_spread = spread(letter_idx)
    special_spread = spread(special_idx)

    # Consecutive runs
    num_consecutive_digit_runs = len(re.findall(r'\d+', pwd))
    num_consecutive_letter_runs = len(re.findall(r'[A-Za-z]+', pwd))
    num_consecutive_upper_runs = len(re.findall(r'[A-Z]+', pwd))
    num_consecutive_special_runs = len(re.findall(r'[^A-Za-z0-9]+', pwd))

    # Transition / alternating
    letter_to_digit = len(re.findall(r'[A-Za-z]\d', pwd))
    digit_to_letter = len(re.findall(r'\d[A-Za-z]', pwd))
    alternating_pattern_score = letter_to_digit + digit_to_letter
    transitions = sum(pwd[i]!=pwd[i-1] for i in range(1,length))
    transitions_to_length_ratio = transitions/length if length else 0

    # Streaks / mixing
    longest_same_char_streak = max((len(m.group(0)) for m in re.finditer(r'(.)\1*', pwd)), default=0)
    digit_letter_mixing_score = (num_digits*num_lower)/length if length else 0

    # Dictionary / names
    contains_dictionary_word, longest_dictionary_word_length, dictionary_coverage_ratio = dictionary_features(pwd)
    contains_common_name = name_features(pwd)
    contains_year = int(bool(re.search(r'(19|20)\d{2}', pwd)))

    # zxcvbn + omen
    zxcvbn_log10_guesses = math.log10(zxcvbn(pwd)["guesses"]) if pwd else 0
    omen_log10 = omen_log10_score(pwd)

    # Final feature order
    return pd.DataFrame([[ 
        length, num_upper, num_lower, num_digits, num_special_char,
        first_is_upper, first_is_digit, first_is_special,
        last_is_upper, last_is_digit, last_is_special,
        length_adj_entropy, bi_ent, combined_entropy_pca_norm,
        digit_spread, letter_spread, special_spread,
        num_consecutive_digit_runs, num_consecutive_letter_runs,
        num_consecutive_upper_runs, num_consecutive_special_runs,
        letter_to_digit, digit_to_letter, alternating_pattern_score,
        transitions_to_length_ratio, longest_same_char_streak,
        digit_letter_mixing_score,
        contains_dictionary_word, longest_dictionary_word_length,
        dictionary_coverage_ratio,
        contains_common_name, contains_year,
        zxcvbn_log10_guesses, omen_log10
    ]], columns=[
        "length","num_upper","num_lower","num_digits","num_special_char",
        "first_is_upper","first_is_digit","first_is_special",
        "last_is_upper","last_is_digit","last_is_special",
        "length_adjusted_entropy","bigram_entropy","combined_entropy_pca_norm",
        "digit_spread","letter_spread","special_spread",
        "num_consecutive_digit_runs","num_consecutive_letter_runs",
        "num_consecutive_upper_runs","num_consecutive_special_runs",
        "letter_to_digit","digit_to_letter","alternating_pattern_score",
        "transitions_to_length_ratio","longest_same_char_streak",
        "digit_letter_mixing_score",
        "contains_dictionary_word","longest_dictionary_word_length",
        "dictionary_coverage_ratio",
        "contains_common_name","contains_year",
        "zxcvbn_log10_guesses","omen_log10"
    ])
