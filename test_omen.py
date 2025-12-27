import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "features"))

from extract_features import omen_log10_score

tests = [
    "password",
    "P@ssw0rd123",
    "A9$kLm!2Qx",
    "123456",
    "qwerty",
]

for p in tests:
    print(p, "=>", omen_log10_score(p))
