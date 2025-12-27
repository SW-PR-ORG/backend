import os
import sys
import xgboost as xgb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
sys.path.append(FEATURES_DIR)

from extract_features import extract_password_features


def test_model_prediction_runs():
    model_path = os.path.join(BASE_DIR, "model", "final_model.json")

    assert os.path.exists(model_path), "Model file missing"

    model = xgb.Booster()
    model.load_model(model_path)

    pwd = "Lo56@dsa"
    X_test = extract_password_features(pwd)

    dmatrix = xgb.DMatrix(X_test)
    prediction = model.predict(dmatrix)

    # Assertions
    assert prediction is not None
    assert len(prediction) == 1
    assert 0.0 <= prediction[0] <= 1.0    # assuming probability
