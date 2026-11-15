
# ============================================================
# NVDA STOCK PREDICTION - MODEL TRAINING AND EVALUATION
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

warnings.filterwarnings("ignore")

# ============================================================
# 1. CONFIGURATION
# ============================================================

INPUT_FILE = "data/processed/NVDA_features.csv"

MODEL_DIR = "models"
RESULT_DIR = "results"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

RANDOM_STATE = 42

# Target classes
CLASS_NAMES = ["DOWN", "HOLD", "UP"]
CLASS_LABELS = [0, 1, 2]

# ============================================================
# 2. LOAD DATA
# ============================================================

print("\nLoading engineered dataset...")

df = pd.read_csv(INPUT_FILE)

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"], utc=True, errors="coerce"
)

df["Target_End_Timestamp"] = pd.to_datetime(
    df["Target_End_Timestamp"], utc=True, errors="coerce"
)

df["Target"] = pd.to_numeric(
    df["Target"], errors="coerce"
)

df = df.dropna(
    subset=[
        "Timestamp",
        "Target_End_Timestamp",
        "Target"
    ]
).copy()

df["Target"] = df["Target"].astype(int)

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

print("Dataset shape:", df.shape)

# ============================================================
# 3. SELECT MODEL FEATURES
# ============================================================

FEATURES = [
    "Return_1",
    "Return_3",
    "Return_6",
    "Return_12",
    "Return_24",
    "High_Low_Range",
    "Close_Open_Return",
    "EMA_9_Distance",
    "EMA_20_Distance",
    "EMA_50_Distance",
    "RSI_14",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Width",
    "BB_Percent_B",
    "ATR_Percent",
    "Log_Volume",
    "Relative_Volume",
    "Volume_Change",
    "OBV_Normalized",
    "Minutes_Since_Open",
    "Time_Sin",
    "Time_Cos",
    "Weekday_Sin",
    "Weekday_Cos"
]

# Optional features
for feature in ["VWAP_Distance", "Log_Transactions"]:
    if feature in df.columns:
        FEATURES.append(feature)

missing_features = [
    feature for feature in FEATURES
    if feature not in df.columns
]

if missing_features:
    raise ValueError(
        f"Missing model features: {missing_features}"
    )

# Convert features to numeric
for feature in FEATURES:
    df[feature] = pd.to_numeric(
        df[feature], errors="coerce"
    )

# Remove infinite values
df.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

# Keep only rows with complete model inputs
df = df.dropna(
    subset=FEATURES + [
        "Target",
        "Future_Return",
        "Target_End_Timestamp"
    ]
).copy()

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)

# Check target labels
df = df[
    df["Target"].isin(CLASS_LABELS)
].copy()

df["Target"] = df["Target"].astype(int)

print("\nNumber of model features:", len(FEATURES))
print("Dataset after removing missing values:", df.shape)

print("\nTarget distribution:")
print(
    df["Target"].value_counts()
    .reindex(CLASS_LABELS, fill_value=0)
    .rename(index={
        0: "DOWN",
        1: "HOLD",
        2: "UP"
    })
)

if df["Target"].nunique() < 2:
    raise ValueError(
        "The dataset contains fewer than two target classes."
    )

# ============================================================
# 4. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\nCreating chronological data splits...")

n = len(df)

train_end_index = int(n * 0.60)
val_end_index = int(n * 0.80)

# Boundaries are based on chronological observation timestamps
validation_start = df.iloc[
    train_end_index
]["Timestamp"]

test_start = df.iloc[
    val_end_index
]["Timestamp"]

# Training observations must have their complete future target
# strictly before the validation period.
train_mask = (
    (df["Timestamp"] < validation_start) &
    (df["Target_End_Timestamp"] < validation_start)
)

# Validation observations must have their complete future target
# strictly before the test period.
val_mask = (
    (df["Timestamp"] >= validation_start) &
    (df["Timestamp"] < test_start) &
    (df["Target_End_Timestamp"] < test_start)
)

# Test observations begin at the test boundary.
test_mask = (
    df["Timestamp"] >= test_start
)

train_df = df.loc[train_mask].copy()
val_df = df.loc[val_mask].copy()
test_df = df.loc[test_mask].copy()

if min(len(train_df), len(val_df), len(test_df)) == 0:
    raise ValueError(
        "One of the chronological splits is empty. "
        "Check the dataset and target timestamps."
    )

X_train = train_df[FEATURES]
y_train = train_df["Target"]

X_val = val_df[FEATURES]
y_val = val_df["Target"]

X_test = test_df[FEATURES]
y_test = test_df["Target"]

print("\nSplit sizes:")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))

print("\nSplit dates:")
print("Validation starts:", validation_start)
print("Test starts:", test_start)

print("\nTraining target distribution:")
print(y_train.value_counts().sort_index())

# ============================================================
# 5. DEFINE EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred, y_prob=None):

    metrics = {
        "Model": model_name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision_Macro": precision_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),
        "Recall_Macro": recall_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),
        "Macro_F1": f1_score(
            y_true,
            y_pred,
            labels=CLASS_LABELS,
            average="macro",
            zero_division=0
        ),
        "Balanced_Accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        )
    }

    # Multi-class ROC-AUC
    try:
        if y_prob is not None:
            metrics["ROC_AUC"] = roc_auc_score(
                y_true,
                y_prob,
                labels=CLASS_LABELS,
                multi_class="ovr",
                average="macro"
            )
        else:
            metrics["ROC_AUC"] = np.nan

    except (ValueError, TypeError):
        metrics["ROC_AUC"] = np.nan

    return metrics


# ============================================================
# 6. DEFINE MACHINE LEARNING MODELS
# ============================================================

print("\nDefining machine learning models...")

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ]),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ))
    ])
}

# Add XGBoost if installed
try:
    from xgboost import XGBClassifier

    models["XGBoost"] = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("XGBoost is available.")

except ImportError:
    print(
        "XGBoost is not installed. "
        "The other models will still be trained."
    )

# ============================================================
# 7. TRAIN MODELS AND EVALUATE ON VALIDATION DATA
# ============================================================

print("\nTraining models...")

trained_models = {}
validation_results = []

for name, model in models.items():

    print("\nTraining:", name)

    try:

        # XGBoost receives balanced sample weights
        if name == "XGBoost":

            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights
            )

        else:
            model.fit(
                X_train,
                y_train
            )

        trained_models[name] = model

        val_predictions = model.predict(X_val)

        val_probabilities = model.predict_proba(X_val)

        result = evaluate_model(
            name,
            y_val,
            val_predictions,
            val_probabilities
        )

        validation_results.append(result)

        print(
            "Validation Macro F1:",
            round(result["Macro_F1"], 4)
        )

    except Exception as error:
        print(
            f"Could not train {name}: {error}"
        )

if not trained_models:
    raise RuntimeError(
        "No machine learning model was trained successfully."
    )

# ============================================================
# 8. BASELINE MODELS
# ============================================================

print("\nEvaluating baseline models...")

# Majority class baseline
majority_class = y_train.mode().iloc[0]

majority_predictions = np.full(
    len(y_val),
    majority_class
)

validation_results.append(
    evaluate_model(
        "Majority Class Baseline",
        y_val,
        majority_predictions
    )
)

# Always UP baseline
always_up_predictions = np.full(
    len(y_val),
    2
)

validation_results.append(
    evaluate_model(
        "Always UP Baseline",
        y_val,
        always_up_predictions
    )
)

# Basic RSI Rule baseline (Section 4.3 of brief)
rsi_val = X_val["RSI_14"]
rsi_val_predictions = np.where(rsi_val < 30, 2, np.where(rsi_val > 70, 0, 1))

validation_results.append(
    evaluate_model(
        "Basic RSI Rule Baseline",
        y_val,
        rsi_val_predictions
    )
)

# ============================================================
# 9. SOFT VOTING ENSEMBLE
# ============================================================

ensemble_names = [
    "Logistic Regression",
    "Random Forest",
    "XGBoost"
]

if all(
    name in trained_models
    for name in ensemble_names
):

    print("\nCreating soft voting ensemble...")

    ensemble_val_probabilities = np.mean(
        [
            trained_models[name].predict_proba(X_val)
            for name in ensemble_names
        ],
        axis=0
    )

    ensemble_val_predictions = np.array(CLASS_LABELS)[
        np.argmax(ensemble_val_probabilities, axis=1)
    ]

    ensemble_result = evaluate_model(
        "Soft Voting Ensemble",
        y_val,
        ensemble_val_predictions,
        ensemble_val_probabilities
    )

    validation_results.append(
        ensemble_result
    )

    print(
        "Ensemble Validation Macro F1:",
        round(ensemble_result["Macro_F1"], 4)
    )

# ============================================================
# 10. COMPARE VALIDATION RESULTS
# ============================================================

validation_df = pd.DataFrame(
    validation_results
)

validation_df = validation_df.sort_values(
    "Macro_F1",
    ascending=False
).reset_index(drop=True)

validation_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "validation_metrics.csv"
    ),
    index=False
)

print("\n==========================================")
print("VALIDATION MODEL COMPARISON")
print("==========================================")

print(
    validation_df.round(4).to_string(index=False)
)

# Select best model using validation Macro F1
best_model_name = validation_df.iloc[0]["Model"]

if best_model_name in [
    "Majority Class Baseline",
    "Always UP Baseline"
]:
    print(
        "\nA baseline performed best. "
        "No trained ML model beats it on validation Macro F1."
    )

    ml_validation = validation_df[
        validation_df["Model"].isin(
            list(trained_models.keys()) +
            ["Soft Voting Ensemble"]
        )
    ]

    if ml_validation.empty:
        raise RuntimeError(
            "No trained model is available for final evaluation."
        )

    best_model_name = ml_validation.iloc[0]["Model"]

print("\nSelected model:", best_model_name)

# ============================================================
# 11. EVALUATE SELECTED MODEL ON TEST DATA
# ============================================================

print("\nEvaluating selected model on untouched test data...")

if best_model_name == "Soft Voting Ensemble":

    test_probabilities = np.mean(
        [
            trained_models[name].predict_proba(X_test)
            for name in ensemble_names
        ],
        axis=0
    )

    test_predictions = np.array(CLASS_LABELS)[
        np.argmax(test_probabilities, axis=1)
    ]

    best_model_object = {
        name: trained_models[name]
        for name in ensemble_names
    }

else:

    best_model_object = trained_models[
        best_model_name
    ]

    test_predictions = best_model_object.predict(
        X_test
    )

    test_probabilities = (
        best_model_object.predict_proba(X_test)
    )

test_metrics = evaluate_model(
    best_model_name,
    y_test,
    test_predictions,
    test_probabilities
)

test_results = [test_metrics]

# Also evaluate the baselines on test data
test_results.append(
    evaluate_model(
        "Majority Class Baseline",
        y_test,
        np.full(len(y_test), majority_class)
    )
)

test_results.append(
    evaluate_model(
        "Always UP Baseline",
        y_test,
        np.full(len(y_test), 2)
    )
)

# Basic RSI Rule baseline (Section 4.3 of brief)
rsi_test = X_test["RSI_14"]
rsi_test_predictions = np.where(rsi_test < 30, 2, np.where(rsi_test > 70, 0, 1))

test_results.append(
    evaluate_model(
        "Basic RSI Rule Baseline",
        y_test,
        rsi_test_predictions
    )
)

test_metrics_df = pd.DataFrame(
    test_results
)

test_metrics_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "test_metrics.csv"
    ),
    index=False
)

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(
    test_metrics_df.round(4).to_string(index=False)
)

# ============================================================
# 12. SAVE TEST PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({

    "Timestamp": test_df["Timestamp"].values,

    "Actual_Target": y_test.values,

    "Predicted_Target": test_predictions,

    "Actual_Future_Return": test_df[
        "Future_Return"
    ].values,

    "Predicted_Class": [
        CLASS_NAMES[int(x)]
        for x in test_predictions
    ],

    "Actual_Class": [
        CLASS_NAMES[int(x)]
        for x in y_test.values
    ],

    "Confidence": np.max(
        test_probabilities,
        axis=1
    )

})

# Add individual class probabilities
for i, class_name in enumerate(CLASS_NAMES):
    predictions_df[
        f"Probability_{class_name}"
    ] = test_probabilities[:, i]

predictions_df.to_csv(
    os.path.join(
        RESULT_DIR,
        "best_model_test_predictions.csv"
    ),
    index=False
)

# ============================================================
# 13. TRADING BACKTEST EVALUATION (AFTER COSTS)
# ============================================================

print("\nRunning simulated trading backtest (with transaction costs)...")

def simulate_trading(pred_labels, future_returns, cost_pct=0.02):
    pnl = []
    active_mask = []
    wins = 0
    active_count = 0
    
    for p, r in zip(pred_labels, future_returns):
        is_up = (p == 2 or p == "UP")
        is_down = (p == 0 or p == "DOWN")
        if is_up:
            pnl.append(r - cost_pct)
            active_mask.append(True)
            active_count += 1
            if r > 0:
                wins += 1
        elif is_down:
            pnl.append(-r - cost_pct)
            active_mask.append(True)
            active_count += 1
            if r < 0:
                wins += 1
        else:
            pnl.append(0.0)
            active_mask.append(False)
            
    pnl = pd.Series(pnl)
    equity = (1.0 + pnl / 100.0).cumprod()
    max_dd = ((equity - equity.cummax()) / equity.cummax()).min() * 100.0 if len(equity) > 0 else 0.0
    total_net_pnl = (equity.iloc[-1] - 1.0) * 100.0 if len(equity) > 0 else 0.0
    win_rate = (wins / active_count * 100.0) if active_count > 0 else 0.0
    coverage = (active_count / len(pred_labels) * 100.0) if len(pred_labels) > 0 else 0.0
    
    active_pnl = pnl[active_mask]
    if len(active_pnl) > 5 and active_pnl.std() > 0:
        sharpe = (active_pnl.mean() / active_pnl.std()) * np.sqrt(78 * 252)
    else:
        sharpe = 0.0

    return {
        "Signals": active_count,
        "Coverage_Pct": round(coverage, 1),
        "Win_Rate_Pct": round(win_rate, 1),
        "Net_PnL_Pct": round(total_net_pnl, 2),
        "Max_Drawdown_Pct": round(max_dd, 2),
        "Sharpe_Ratio": round(sharpe, 2)
    }

# Find optimal confidence threshold on validation set
val_probabilities = (
    np.mean([trained_models[name].predict_proba(X_val) for name in ensemble_names], axis=0)
    if best_model_name == "Soft Voting Ensemble"
    else trained_models[best_model_name].predict_proba(X_val)
)

best_val_thresh = 0.40
best_val_ret = -999.0
val_fut_ret = val_df["Future_Return"].values

for th in np.arange(0.35, 0.50, 0.02):
    th = round(float(th), 2)
    gated_val = np.where((val_probabilities[:, 2] >= th) & (val_probabilities[:, 2] > val_probabilities[:, 0]), "UP",
                np.where((val_probabilities[:, 0] >= th) & (val_probabilities[:, 0] > val_probabilities[:, 2]), "DOWN", "HOLD"))
    sim_res = simulate_trading(gated_val, val_fut_ret)
    if sim_res["Signals"] >= 30 and sim_res["Net_PnL_Pct"] > best_val_ret:
        best_val_ret = sim_res["Net_PnL_Pct"]
        best_val_thresh = th

print(f"Optimal validation confidence threshold: {best_val_thresh}")

# Apply confidence gating on test set
test_fut_ret = test_df["Future_Return"].values
gated_test_preds = np.where((test_probabilities[:, 2] >= best_val_thresh) & (test_probabilities[:, 2] > test_probabilities[:, 0]), "UP",
                   np.where((test_probabilities[:, 0] >= best_val_thresh) & (test_probabilities[:, 0] > test_probabilities[:, 2]), "DOWN", "HOLD"))

trading_rows = []

# 1. Best Model (Confidence Gated)
t_res = simulate_trading(gated_test_preds, test_fut_ret)
t_res["Strategy"] = f"{best_model_name} (Confidence Gated @ {best_val_thresh})"
trading_rows.append(t_res)

# 2. Best Model (Raw Argmax)
t_raw = simulate_trading(predictions_df["Predicted_Class"], test_fut_ret)
t_raw["Strategy"] = f"{best_model_name} (Raw Argmax)"
trading_rows.append(t_raw)

# 3. Basic RSI Rule Baseline
t_rsi = simulate_trading(np.where(rsi_test < 30, "UP", np.where(rsi_test > 70, "DOWN", "HOLD")), test_fut_ret)
t_rsi["Strategy"] = "Basic RSI Rule Baseline"
trading_rows.append(t_rsi)

# 4. Always UP Baseline
t_up = simulate_trading(np.full(len(test_df), "UP"), test_fut_ret)
t_up["Strategy"] = "Always UP Baseline"
trading_rows.append(t_up)

# 5. Majority Class Baseline
t_maj = simulate_trading(np.full(len(test_df), "HOLD"), test_fut_ret)
t_maj["Strategy"] = "Majority Class Baseline"
trading_rows.append(t_maj)

trading_metrics_df = pd.DataFrame(trading_rows)[
    ["Strategy", "Signals", "Coverage_Pct", "Win_Rate_Pct", "Net_PnL_Pct", "Max_Drawdown_Pct", "Sharpe_Ratio"]
]

trading_metrics_df.to_csv(
    os.path.join(RESULT_DIR, "trading_metrics.csv"),
    index=False
)

print("\n==========================================")
print("TRADING BACKTEST RESULTS (WITH TRANSACTION COSTS)")
print("==========================================")
print(trading_metrics_df.to_string(index=False))

# Save confidence threshold
with open(os.path.join(MODEL_DIR, "confidence_threshold.txt"), "w") as f:
    f.write(str(best_val_thresh))

# ============================================================
# 13. CONFUSION MATRIX
# ============================================================

print("\nGenerating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_predictions,
    labels=CLASS_LABELS
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(7, 6))

disp.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "best_model_confusion_matrix.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 14. VALIDATION MODEL COMPARISON GRAPH
# ============================================================

print("\nGenerating model comparison graph...")

plot_df = validation_df.sort_values(
    "Macro_F1",
    ascending=True
)

plt.figure(figsize=(10, 6))

plt.barh(
    plot_df["Model"],
    plot_df["Macro_F1"]
)

plt.xlabel("Validation Macro F1")
plt.ylabel("Model")

plt.title(
    "Machine Learning Model Comparison"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "model_comparison.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 15. SAVE TRAINED MODEL
# ============================================================

print("\nSaving trained model...")

joblib.dump(
    best_model_object,
    os.path.join(
        MODEL_DIR,
        "best_model.joblib"
    )
)

# Save selected model name
with open(
    os.path.join(
        MODEL_DIR,
        "best_model.txt"
    ),
    "w"
) as file:

    file.write(best_model_name)

# Save feature names for future inference
joblib.dump(
    FEATURES,
    os.path.join(
        MODEL_DIR,
        "feature_names.joblib"
    )
)

# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n==========================================")
print("MODEL TRAINING COMPLETED")
print("==========================================")

print("Selected model:", best_model_name)

print(
    "Test Accuracy:",
    round(test_metrics["Accuracy"], 4)
)

print(
    "Test Macro F1:",
    round(test_metrics["Macro_F1"], 4)
)

print(
    "Test Balanced Accuracy:",
    round(test_metrics["Balanced_Accuracy"], 4)
)

print(
    "Test ROC-AUC:",
    round(test_metrics["ROC_AUC"], 4)
    if pd.notna(test_metrics["ROC_AUC"])
    else "Not available"
)

print("\nSaved files:")

print("1. results/validation_metrics.csv")
print("2. results/test_metrics.csv")
print("3. results/trading_metrics.csv")
print("4. results/best_model_test_predictions.csv")
print("5. results/best_model_confusion_matrix.png")
print("6. results/model_comparison.png")
print("7. models/best_model.joblib")
print("8. models/best_model.txt")
print("9. models/confidence_threshold.txt")
print("10. models/feature_names.joblib")

print("\n==========================================")
