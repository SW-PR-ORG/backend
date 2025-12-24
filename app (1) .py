from fastapi import FastAPI
from pydantic import BaseModel
import xgboost as xgb
import pickle

from features.extract_features import extract_password_features
from rule_based.ruleBased import rule_password_scorer
from leak_check import is_leaked
from shap_utils import shap_to_dict, top_contributors, human_explanation

app = FastAPI()

# Load model
model = xgb.XGBRegressor()
model.load_model("model/xgboost_model.json")

# Load SHAP explainer
with open("model/explainer.pkl", "rb") as f:
    explainer = pickle.load(f)

class PasswordRequest(BaseModel):
    password: str

# -------------------------
# Rule-based endpoint
# -------------------------
@app.post("/rule-score")
def rule_score(req: PasswordRequest):
    return {
        "rule_score": rule_password_scorer(req.password),
        "is_leaked": is_leaked(req.password)
    }
