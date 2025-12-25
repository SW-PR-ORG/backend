# shap_utils.py
import numpy as np

# Your exact feature order (CRITICAL)
FEATURE_NAMES = [
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
]

def shap_to_dict(shap_values):
    """
    Convert SHAP values to {feature: contribution}
    """
    values = shap_values.values[0]
    return {
        FEATURE_NAMES[i]: float(values[i])
        for i in range(len(FEATURE_NAMES))
    }

def top_contributors(shap_dict, top_k=5):
    """
    Return strongest positive and negative contributors
    """
    sorted_items = sorted(
        shap_dict.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    positives = [(f, v) for f, v in sorted_items if v > 0][:top_k]
    negatives = [(f, v) for f, v in sorted_items if v < 0][:top_k]

    return {
        "positive": positives,
        "negative": negatives
    }

def human_explanation(top_features):
    """
    Convert top SHAP features into human-friendly explanation
    """
    reasons = []

    for feature, value in top_features["negative"]:
        if feature in ["contains_dictionary_word", "dictionary_coverage_ratio"]:
            reasons.append("contains dictionary words")
        elif feature in ["contains_common_name"]:
            reasons.append("contains common names")
        elif feature in ["length", "length_adjusted_entropy"]:
            reasons.append("low password length or entropy")
        elif feature in ["num_digits", "num_special_char"]:
            reasons.append("insufficient character variety")
        elif feature in ["zxcvbn_log10_guesses", "omen_log10"]:
            reasons.append("easy to guess using cracking models")
        elif feature == "longest_same_char_streak":
            reasons.append("repeated characters")

    return list(set(reasons))  # remove duplicates
