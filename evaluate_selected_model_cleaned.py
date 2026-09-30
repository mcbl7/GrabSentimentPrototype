from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_PATH = Path("data/splits_cleaned/train.csv")
TEST_PATH = Path("data/splits_cleaned/test.csv")

RESULTS_DIR = Path("results/platform_evaluation_cleaned")
MODEL_DIR = Path("models/final_cleaned")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FINAL_MODEL_PATH = MODEL_DIR / "tfidf_svm_selected.joblib"

LABELS = ["Negative", "Neutral", "Positive"]

RANDOM_STATE = 42


# ============================================================
# LOAD FIXED TRAIN / TEST DATA
# ============================================================

print("=" * 75)
print("SELECTED MODEL PLATFORM EVALUATION")
print("=" * 75)

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

required_columns = {
    "Review Text",
    "Sentiment",
    "Store",
}

for name, dataframe in [
    ("train", train_df),
    ("test", test_df),
]:
    missing = required_columns - set(dataframe.columns)

    if missing:
        raise ValueError(
            f"{name}.csv is missing columns: "
            f"{sorted(missing)}"
        )


# Normalize platform names
train_df["Store"] = (
    train_df["Store"]
    .astype(str)
    .str.lower()
    .str.strip()
)

test_df["Store"] = (
    test_df["Store"]
    .astype(str)
    .str.lower()
    .str.strip()
)


X_train = (
    train_df["Review Text"]
    .fillna("")
    .astype(str)
)

y_train = train_df["Sentiment"]

X_test = (
    test_df["Review Text"]
    .fillna("")
    .astype(str)
)

y_test = test_df["Sentiment"]


print(f"\nTraining rows: {len(train_df):,}")
print(f"Testing rows:  {len(test_df):,}")

print("\nTest platform distribution:")
print(test_df["Store"].value_counts())

print("\nTest sentiment distribution:")
print(test_df["Sentiment"].value_counts())


# ============================================================
# FINAL SELECTED MODEL
#
# Rebuild the winning TF-IDF + SVM configuration using
# the hyperparameters selected during cross-validation.
#
# Best TF-IDF SVM parameter:
# C = 0.1
# ============================================================

print("\n" + "=" * 75)
print("TRAINING FINAL SELECTED MODEL")
print("=" * 75)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=10000,
            sublinear_tf=True,
        ),
    ),
    (
        "clf",
        LinearSVC(
            C=0.1,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    ),
])

model.fit(
    X_train,
    y_train,
)

joblib.dump(
    model,
    FINAL_MODEL_PATH,
)

print(
    f"\nSaved selected model to: "
    f"{FINAL_MODEL_PATH}"
)


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_subset(name, subset_df):

    if subset_df.empty:
        raise ValueError(
            f"No rows found for {name}"
        )

    X = (
        subset_df["Review Text"]
        .fillna("")
        .astype(str)
    )

    y_true = subset_df["Sentiment"]

    y_pred = model.predict(X)

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    print("\n" + "=" * 75)
    print(f"{name.upper()} RESULTS")
    print("=" * 75)

    print(f"Reviews:         {len(subset_df):,}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=LABELS,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=LABELS,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"Actual {label}"
            for label in LABELS
        ],
        columns=[
            f"Predicted {label}"
            for label in LABELS
        ],
    )

    safe_name = (
        name
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    cm_df.to_csv(
        RESULTS_DIR
        / f"{safe_name}_confusion_matrix.csv"
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(
        report
    ).transpose().to_csv(
        RESULTS_DIR
        / f"{safe_name}_classification_report.csv"
    )

    return {
        "Evaluation Set": name,
        "Number of Reviews": len(subset_df),
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
    }


# ============================================================
# COMBINED TEST SET
# ============================================================

summary = []

summary.append(
    evaluate_subset(
        "Combined",
        test_df,
    )
)


# ============================================================
# ANDROID
# ============================================================

android_df = test_df[
    test_df["Store"] == "android"
].copy()

summary.append(
    evaluate_subset(
        "Android",
        android_df,
    )
)


# ============================================================
# IOS
# ============================================================

ios_df = test_df[
    test_df["Store"] == "ios"
].copy()

summary.append(
    evaluate_subset(
        "iOS",
        ios_df,
    )
)


# ============================================================
# SAVE REVIEW-LEVEL PREDICTIONS
# ============================================================

test_predictions = test_df.copy()

test_predictions[
    "Predicted Sentiment"
] = model.predict(
    X_test
)

test_predictions[
    "Correct Prediction"
] = (
    test_predictions["Sentiment"]
    == test_predictions["Predicted Sentiment"]
)

test_predictions.to_csv(
    RESULTS_DIR
    / "selected_model_test_predictions.csv",
    index=False,
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)

summary_path = (
    RESULTS_DIR
    / "platform_performance_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False,
)


print("\n" + "=" * 75)
print("PLATFORM PERFORMANCE SUMMARY")
print("=" * 75)

print(
    summary_df.to_string(
        index=False
    )
)

print("\nSaved results to:")
print(f"  {RESULTS_DIR}/")

print("\nSaved final selected model to:")
print(f"  {FINAL_MODEL_PATH}")

print(
    "\nPlatform evaluation completed successfully."
)