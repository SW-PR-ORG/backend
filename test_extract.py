import sys
import os

# Add the features folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
sys.path.append(FEATURES_DIR)

from extract_features import extract_password_features


def test_extract_password_features_basic():
    pwd = "P@ssw0rd123"

    df = extract_password_features(pwd)

    # Assertions (this is what makes it a REAL test)
    assert df is not None
    assert df.shape[0] == 1              # one password
    assert df.shape[1] > 0               # has features
    assert "length" in df.columns        # example feature
