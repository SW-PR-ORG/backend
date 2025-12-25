import os
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PCFG_BASE_DIR = os.path.join(BASE_DIR, "pcfg_cracker")
RULES_DIR = os.path.join(PCFG_BASE_DIR, "Rules")

REPO_URL = "https://github.com/SW-PR-ORG/Rules"

def clone_rules_if_missing():
    # Rules already present → skip
    if os.path.exists(RULES_DIR) and os.listdir(RULES_DIR):
        print("PCFG Rules already exist — skipping clone")
        return

    print("Cloning PCFG Rules repository...")
    os.makedirs(PCFG_BASE_DIR, exist_ok=True)

    try:
        subprocess.run(
            ["git", "clone", REPO_URL, RULES_DIR],
            check=True
        )
        print("PCFG Rules cloned successfully.")
    except subprocess.CalledProcessError as e:
        print("Failed to clone PCFG Rules:", e)
        raise
