from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import shap
import os
import sys

from features.extract_features import extract_password_features
from rule_based.rule_based import rule_password_scorer
from is_leaked.is_leaked import is_leaked
from shap_utils import shap_to_dict, top_contributors, human_explanation

app = FastAPI()

# -------------------------
# Load model (Booster)
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model", "final_model.json")

model = xgb.Booster()
model.load_model(model_path)

# Create SHAP explainer (DO NOT pickle-load)
explainer = shap.TreeExplainer(model)



# -------------------------
# Request schema
# -------------------------
class PasswordRequest(BaseModel):
    password: str

# -------------------------
# Rule-based endpoint
# -------------------------
@app.post("/rule-score")
def rule_score(req: PasswordRequest):
    score, feedback = rule_password_scorer(req.password, return_feedback=True)
    
    return {
        "rule_score": score,
        "feedback": feedback,
        "is_leaked": is_leaked(req.password)
    }

# -------------------------
# ML + SHAP endpoint
# -------------------------
@app.post("/check-password")
def check_password(req: PasswordRequest):
    pwd = req.password

    if is_leaked(pwd):
        return {"error": "Password exists in rockyou 2009 dataset. It will almost be cracked immediately"}

    # Extract features
    features_df = extract_password_features(pwd)

    # Convert to DMatrix (REQUIRED)
    dmat = xgb.DMatrix(features_df)

    # Predict
    score = float(model.predict(dmat)[0])

    # SHAP (CORRECT)
    shap_vals = explainer(dmat)

    shap_dict = shap_to_dict(shap_vals)
    top_feats = top_contributors(shap_dict)
    explanation = human_explanation(top_feats)

    return {
        "ml_score": score,
        "top_factors": top_feats,
        "explanation": explanation
    }
