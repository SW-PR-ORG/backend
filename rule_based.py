import re

def rule_password_scorer(password: str) -> float:
    """Simple rule-based password scoring (max 10)."""
    score = 0.0
    if len(password) >= 8:
        score += 2.5
    if re.search(r"[A-Z]", password):
        score += 2.5
    if re.search(r"\d", password):
        score += 2.5
    if re.search(r"[^\w]", password):
        score += 2.5
    return score