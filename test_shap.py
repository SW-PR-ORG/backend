import os
import sys
import xgboost as xgb
import shap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
sys.path.append(FEATURES_DIR)

from extract_features import extract_password_features
from shap_utils import shap_to_dict, top_contributors, human_explanation


def test_shap_explanation_pipeline():
    model_path = os.path.join(BASE_DIR, "model", "final_model.json")

    model = xgb.Booster()
    model.load_model(model_path)

    pwd = "Lo56@dsa"
    X = extract_password_features(pwd)
    dmatrix = xgb.DMatrix(X)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(dmatrix)

    shap_dict = shap_to_dict(shap_values)
    top_feats = top_contributors(shap_dict)
    explanation = human_explanation(top_feats)

    # Assertions
    assert isinstance(shap_dict, dict)
    assert "positive" in top_feats
    assert "negative" in top_feats
    assert isinstance(explanation, list)
