import sys
import os
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "features"))

from extract_features import extract_password_features

# Load model
model = xgb.Booster()
model.load_model(os.path.join(BASE_DIR, "model", "final_model.json"))

# Test passwords
pwd = "P@ssw0rd123"
pwd = "ilovefang1"

df = extract_password_features(pwd)

# Create DMatrix and predict
dmat = xgb.DMatrix(df)
score = model.predict(dmat)

print("Model score:", score)
