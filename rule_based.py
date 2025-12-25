import re

def rule_password_scorer(password: str, return_feedback=False):
    """Simple rule-based password scoring (max 10)."""
    score = 0.0
    feedback = []
    
    if len(password) >= 8:
        score += 2.5
    else:
        feedback.append("You need at least 8 characters")
    
    if re.search(r"[A-Z]", password):
        score += 2.5
    else:
        feedback.append("You need at least 1 uppercase letter")
    
    if re.search(r"\d", password):
        score += 2.5
    else:
        feedback.append("You need at least 1 digit")
    
    if re.search(r"[^\w]", password):
        score += 2.5
    else:
        feedback.append("You need at least 1 special character")
    
    if return_feedback:
        return score, feedback
    return score