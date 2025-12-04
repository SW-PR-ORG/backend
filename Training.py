# fixed_full_xgboost_pipeline.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ---------------- SETTINGS ----------------
CSV_PATH = r"C:\Users\omara\Desktop\rockyou_100k_PCFG_OMEN_target.csv"  # <- adjust if necessary
RANDOM_SEED = 42
TEST_SIZE = 0.2
CAP_LOG10 = 16.0   # cap for extremely large log10 values (tunable)
MIN_LOG10 = 2.0    # used later for mapping to 1-10
MODEL_OUTPUT_DIR = "model_output"
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

# Quick mode: set to True for a fast run (no RandomizedSearchCV)
QUICK_TRAIN = False

# ---------- HELPERS ----------
def sanitize_numeric_cols(df, cols, cap_log10=CAP_LOG10):
    """Convert to numeric safely for the estimator columns, cap ±inf, fill NaN with 0."""
    for c in cols:
        if c not in df.columns:
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce")
        posinf_mask = np.isposinf(df[c].values)
        neginf_mask = np.isneginf(df[c].values)
        if posinf_mask.any():
            df.loc[posinf_mask, c] = cap_log10
        if neginf_mask.any():
            df.loc[neginf_mask, c] = 0.0
    df[cols] = df[cols].fillna(0.0)
    return df

def log10_to_score_continuous(log10v, min_log=MIN_LOG10, max_log=CAP_LOG10):
    """Map log10 guesses (clamped) to continuous 1.0-10.0 scale."""
    v = float(log10v) if np.isfinite(log10v) else min_log
    v = max(min_log, min(max_log, v))
    score = 1.0 + 9.0 * (v - min_log) / (max_log - min_log)
    return score

def robust_sanitize_features(X, log_like_cols=None, cap_log10=CAP_LOG10, min_log10=MIN_LOG10):
    """Sanitize feature DataFrame X:
       - coerce object columns to numeric where possible
       - replace ±inf -> NaN, then fill NaN with median
       - clip log-like columns to [min_log10, cap_log10]
       - clip other columns to a large safe range
    """
    if log_like_cols is None:
        log_like_cols = ['zxcvbn_log10_guesses', 'omen_log10', 'pcfg_neglog10_prob', 'target_log10_guesses']

    # convert non-numeric-like columns to numeric where possible
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors='coerce')

    # replace ±inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # fill NaN with median per column (robust)
    for c in X.columns:
        if X[c].notna().sum() == 0:
            # column all NaN -> fill with 0
            X[c] = X[c].fillna(0.0)
            continue
        med = X[c].median()
        X[c] = X[c].fillna(med)

    # clip values: keep large safe window for general features; clamp log-like features tighter
    clip_min_general = -1e6
    clip_max_general = 1e6
    for c in X.columns:
        if c in log_like_cols:
            # ensure numeric and clamp to [min_log10, cap_log10]
            if c in X.columns:
                X[c] = X[c].astype(float).clip(lower=min_log10, upper=cap_log10)
        else:
            X[c] = X[c].astype(float).clip(lower=clip_min_general, upper=clip_max_general)

    # final check: replace any remaining non-finite with column median
    for c in X.columns:
        col = X[c].to_numpy()
        if not np.all(np.isfinite(col)):
            med = np.nanmedian(col[np.isfinite(col)]) if np.any(np.isfinite(col)) else 0.0
            X[c] = np.where(np.isfinite(col), col, med)

    return X

# ---------------- LOAD DATA ----------------
print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH, encoding="utf-8", low_memory=False)

# ---------- ESTIMATOR COLUMNS ----------
est_cols = ['zxcvbn_log10_guesses', 'omen_log10', 'pcfg_neglog10_prob']
df = sanitize_numeric_cols(df, est_cols, cap_log10=CAP_LOG10)

# Create target if not present
if 'target_log10_guesses' not in df.columns:
    df['target_log10_guesses'] = df[est_cols].max(axis=1)

# Add continuous score (not used as feature)
df['target_score_continuous'] = df['target_log10_guesses'].apply(lambda v: log10_to_score_continuous(v, MIN_LOG10, CAP_LOG10))

# ------------- BUILD FEATURE MATRIX -------------
# drop non-feature columns
drop_cols = [c for c in ['password', 'target_log10_guesses', 'target_score_continuous'] if c in df.columns]
X = df.drop(columns=drop_cols)
y = df['target_log10_guesses'].astype(float)

# identify and drop columns that are obviously non-numeric (strings)
non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
if non_numeric:
    print("Dropping non-numeric columns from features:", non_numeric)
    X = X.drop(columns=non_numeric)

# Robust sanitize all features
X = robust_sanitize_features(X, log_like_cols=est_cols + ['target_log10_guesses'], cap_log10=CAP_LOG10, min_log10=MIN_LOG10)

# Diagnostic: columns with non-finite values (should be none)
bad_cols = []
for c in X.columns:
    col = X[c].to_numpy()
    n_inf = np.isposinf(col).sum()
    n_ninf = np.isneginf(col).sum()
    n_nan = np.isnan(col).sum()
    if n_inf + n_ninf + n_nan > 0:
        bad_cols.append((c, int(n_inf), int(n_ninf), int(n_nan)))
if bad_cols:
    print("Warning: non-finite values remain (col, +inf, -inf, NaN):")
    for info in bad_cols:
        print(info)
    raise SystemExit("Aborting due to non-finite values in features.")
else:
    print("All feature columns finite and numeric. Feature count:", X.shape[1])

feature_names = X.columns.tolist()

# ------------ TRAIN/TEST SPLIT (safe stratify) -------------
# Try stratified split using binned y; fallback to plain split if it fails
try:
    y_binned = pd.cut(y, bins=[-np.inf, 3, 5, 7, 9, np.inf], labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_binned
    )
except Exception as e:
    print("Stratified split failed (probably due to bins); falling back to random split. Error:", e)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ------------ PIPELINE -------------
xgb_reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbosity=1,
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('xgb', xgb_reg)
])

# ------------ TRAINING -------------
if QUICK_TRAIN:
    print("QUICK_TRAIN is True -> training a single XGBoost model (no hyperparam search).")
    pipeline.set_params(xgb__n_estimators=200, xgb__max_depth=5, xgb__learning_rate=0.05)
    pipeline.fit(X_train, y_train)
    best_model = pipeline
else:
    # randomized search configuration
    param_dist = {
        'xgb__n_estimators': [100, 300, 600],
        'xgb__max_depth': [3, 5, 7],
        'xgb__learning_rate': [0.01, 0.05, 0.1],
        'xgb__subsample': [0.6, 0.8, 1.0],
        'xgb__colsample_bytree': [0.5, 0.7, 1.0],
        'xgb__reg_alpha': [0, 0.5, 1.0],
        'xgb__reg_lambda': [1.0, 2.0, 5.0],
    }

    rs = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        n_iter=20,
        scoring='neg_root_mean_squared_error',
        cv=3,
        verbose=2,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        error_score=np.nan  # let failures produce nan but search continue; we'll handle if all fail
    )

    print("Starting RandomizedSearchCV (this may take a while)...")
    try:
        rs.fit(X_train, y_train)
        # if all candidates failed, rs.best_estimator_ will raise; handle below
        best_model = rs.best_estimator_
        print("RandomizedSearchCV finished. Best params:", rs.best_params_)
    except Exception as e:
        print("RandomizedSearchCV failed or all fits failed. Falling back to quick fit. Error:", e)
        pipeline.set_params(xgb__n_estimators=200, xgb__max_depth=5, xgb__learning_rate=0.05)
        pipeline.fit(X_train, y_train)
        best_model = pipeline

# ---------- EVALUATE ON TEST (fixed RMSE computation) ----------
y_pred = best_model.predict(X_test)

# mean_squared_error in some environments may not accept squared=False
mse = mean_squared_error(y_test, y_pred)   # returns MSE
rmse = float(np.sqrt(mse))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R2:   {r2:.4f}")

# ------------ SAVE MODEL PACKAGE -------------
model_path = os.path.join(MODEL_OUTPUT_DIR, "xgb_password_regressor.joblib")
joblib.dump({
    "model": best_model,
    "feature_names": feature_names,
    "cap_log10": CAP_LOG10,
    "min_log10": MIN_LOG10
}, model_path)
print("Saved model package to:", model_path)

# ------------ FEATURE IMPORTANCE (XGBoost) -------------
try:
    booster = best_model.named_steps['xgb'].get_booster()
    fmap = booster.get_score(importance_type='gain')
    fi = pd.DataFrame.from_dict(fmap, orient='index', columns=['gain'])
    fi.index.name = 'feature'
    fi = fi.reset_index().sort_values('gain', ascending=False)
    fi.to_csv(os.path.join(MODEL_OUTPUT_DIR, "feature_importances_xgb_gain.csv"), index=False)
    print("Saved feature importances to CSV.")
except Exception as e:
    print("Could not extract XGBoost booster feature importances:", e)

# ------------ PLOTS & SAMPLE PREDICTIONS -------------
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.xlabel("True target_log10_guesses")
plt.ylabel("Predicted target_log10_guesses")
plt.title("True vs Predicted (log10 guesses)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_DIR, "true_vs_pred_log10.png"))
# plt.show()  # uncomment if running interactively

y_pred_score = [log10_to_score_continuous(v, min_log=MIN_LOG10, max_log=CAP_LOG10) for v in y_pred]
sample_out = pd.DataFrame({
    "true_log10": y_test.values,
    "pred_log10": y_pred,
    "true_score": y_test.apply(lambda v: log10_to_score_continuous(v, min_log=MIN_LOG10, max_log=CAP_LOG10)).values,
    "pred_score": y_pred_score
})
sample_out.to_csv(os.path.join(MODEL_OUTPUT_DIR, "predictions_sample.csv"), index=False)
print("Wrote sample predictions CSV to output dir.")

print("Done.")
