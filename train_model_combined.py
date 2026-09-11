import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_FILE = "features_combined.csv"
MODEL_FILE = "driver_behavior_model_combined.pkl"
METRICS_FILE = "model_metrics_combined.json"
IMPORTANCE_FILE = "feature_importance_combined.csv"
CONFUSION_FILE = "confusion_matrix_combined.csv"

FEATURES = [
    "ear",
    "mar",
    "jaw_ratio",
    "brightness"
]

TARGET = "behavior"

RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 65)
print(" COMBINED DRIVER BEHAVIOR MODEL TRAINING")
print("=" * 65)
print()

df = pd.read_csv(DATASET_FILE)

print(
    f"Dataset size: {len(df)}"
)

print()

print("Subjects:")
print(
    df["subject_id"].value_counts().to_string()
)

print()

print("Behaviors:")
print(
    df[TARGET].value_counts().to_string()
)

print()


# ============================================================
# VALIDATE
# ============================================================

required_columns = FEATURES + [TARGET]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:

    print(
        "ERROR: Missing columns:"
    )

    print(
        missing
    )

    raise SystemExit


# ============================================================
# PREPARE DATA
# ============================================================

X = df[FEATURES].copy()

y = df[TARGET].copy()


# Replace invalid numeric values

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples : {len(X_test)}"
)

print()


# ============================================================
# MODEL
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=18,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print(
    "Training Random Forest..."
)

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)

print()


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)


print("=" * 65)
print(" MODEL PERFORMANCE")
print("=" * 65)
print()

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print()

print("Classification Report:")
print()

print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    y.unique()
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

rf = model.named_steps[
    "classifier"
]

importance = rf.feature_importances_

importance_df = pd.DataFrame(
    {
        "feature": FEATURES,
        "importance": importance
    }
)

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)

importance_df.to_csv(
    IMPORTANCE_FILE,
    index=False
)

print("Feature importance:")
print()

print(
    importance_df.to_string(
        index=False
    )
)

print()


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)

print(
    f"Model saved: {MODEL_FILE}"
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {
    "dataset": DATASET_FILE,
    "samples": int(len(df)),
    "features": FEATURES,
    "accuracy": float(accuracy),
    "precision_weighted": float(precision),
    "recall_weighted": float(recall),
    "f1_weighted": float(f1),
    "subjects": sorted(
        df["subject_id"].unique().tolist()
    ),
    "classes": labels
}

with open(
    METRICS_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


print(
    f"Metrics saved: {METRICS_FILE}"
)

print()

print("=" * 65)
print(" TRAINING COMPLETED")
print("=" * 65)
print()