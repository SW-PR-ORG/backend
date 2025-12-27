import sys
import os

# Add the features folder to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_DIR = os.path.join(BASE_DIR, "features")
sys.path.append(FEATURES_DIR)

# Import the function
from extract_features import extract_password_features

# Test password
pwd = "P@ssw0rd123"

# Extract features
df = extract_password_features(pwd)

# Show results
print(df)
print("\nColumns order:")
print(list(df.columns))
