# test_shap_simple.py
import os
import sys
import pandas as pd
import xgboost as xgb
import shap
import joblib
import numpy as np

from shap_utils import shap_to_dict, top_contributors, human_explanation

# Add features folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
sys.path.append(FEATURES_DIR)

from extract_features import extract_password_features

# Load model
model_path = os.path.join(BASE_DIR, "model", "final_model.json")
model = xgb.Booster()
model.load_model(model_path)
print("Model loaded")

# Extract features
pwd = "Lo56@dsa"
X_test = extract_password_features(pwd)
print(f"Features shape: {X_test.shape}")

# Convert to DMatrix
dmatrix = xgb.DMatrix(X_test)

# Make prediction
prediction = model.predict(dmatrix)
print(f"Prediction: {prediction}")

# Create TreeExplainer
explainer = shap.TreeExplainer(model)

# Calculate SHAP values
shap_values = explainer(dmatrix)

# Convert SHAP values to dict
shap_dict = shap_to_dict(shap_values)

# Get top contributors
top_feats = top_contributors(shap_dict)

# Human explanation
explanation = human_explanation(top_feats)

print("\nTop positive contributors:")
for f, v in top_feats["positive"]:
    print(f"{f:30}: {v:.6f}")

print("\nTop negative contributors:")
for f, v in top_feats["negative"]:
    print(f"{f:30}: {v:.6f}")

print("\nHuman explanation:")
for reason in explanation:
    print(f"{reason}")

# Force plot
# shap.force_plot(
#     explainer.expected_value,
#     shap_values.values[0],
#     X_test.iloc[0],
#     feature_names=X_test.columns.tolist(),
#     matplotlib=True
# )