from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/reviews_cleaned.xlsx")

OUTPUT_DIR = Path("data/splits_cleaned")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = OUTPUT_DIR / "train.csv"
TEST_PATH = OUTPUT_DIR / "test.csv"

REPORT_PATH = Path(
    "results/language_audit/cleaned_split_report.json"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20

VALID_SENTIMENTS = {
    "Positive",
    "Neutral",
    "Negative",
}


# ============================================================
# LOAD CLEANED DATASET
# ============================================================

print("=" * 75)
print("PREPARE CLEANED TRAIN / TEST SPLIT")
print("=" * 75)

df = pd.read_excel(DATA_PATH)

print(f"\nLoaded cleaned reviews: {len(df):,}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = {
    "Review Text",
    "Sentiment",
    "Store",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"Missing required columns: {sorted(missing)}"
    )


# ============================================================
# NORMALIZE VALUES
# ============================================================

df["Review Text"] = (
    df["Review Text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["Store"] = (
    df["Store"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["Sentiment"] = (
    df["Sentiment"]
    .astype(str)
    .str.strip()
    .str.title()
)


# ============================================================
# VALIDATION
# ============================================================

empty_reviews = (
    df["Review Text"] == ""
).sum()

if empty_reviews:
    raise ValueError(
        f"Found {empty_reviews} empty reviews."
    )


invalid_sentiments = (
    set(df["Sentiment"].unique())
    - VALID_SENTIMENTS
)

if invalid_sentiments:
    raise ValueError(
        f"Invalid sentiment labels: "
        f"{sorted(invalid_sentiments)}"
    )


valid_stores = {
    "android",
    "ios",
}

invalid_stores = (
    set(df["Store"].unique())
    - valid_stores
)

if invalid_stores:
    raise ValueError(
        f"Unexpected Store values: "
        f"{sorted(invalid_stores)}"
    )


# Check duplicate Review IDs if available
if "Review ID" in df.columns:

    duplicate_ids = (
        df["Review ID"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate Review IDs: "
        f"{duplicate_ids:,}"
    )

    if duplicate_ids:
        raise ValueError(
            "Duplicate Review IDs detected."
        )


# ============================================================
# DISPLAY CLEANED DISTRIBUTION
# ============================================================

print("\nPlatform distribution:")
print(
    df["Store"]
    .value_counts()
)

print("\nSentiment distribution:")
print(
    df["Sentiment"]
    .value_counts()
)

print("\nPlatform × Sentiment distribution:")
print(
    pd.crosstab(
        df["Store"],
        df["Sentiment"],
    )
)


# ============================================================
# JOINT STRATIFICATION KEY
# ============================================================

df["_stratify_key"] = (
    df["Store"]
    + "__"
    + df["Sentiment"]
)


print("\nJoint stratification groups:")
print(
    df["_stratify_key"]
    .value_counts()
)


# ============================================================
# FIXED 80 / 20 SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["_stratify_key"],
)


# Remove temporary key
train_df = (
    train_df
    .drop(
        columns=["_stratify_key"]
    )
    .reset_index(drop=True)
)

test_df = (
    test_df
    .drop(
        columns=["_stratify_key"]
    )
    .reset_index(drop=True)
)


# ============================================================
# SAVE SPLITS
# ============================================================

train_df.to_csv(
    TRAIN_PATH,
    index=False,
)

test_df.to_csv(
    TEST_PATH,
    index=False,
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 75)
print("CLEANED SPLIT SUMMARY")
print("=" * 75)

print(
    f"\nTraining reviews: "
    f"{len(train_df):,}"
)

print(
    f"Testing reviews:  "
    f"{len(test_df):,}"
)


print("\nTRAIN — Sentiment:")
print(
    train_df["Sentiment"]
    .value_counts()
)

print("\nTEST — Sentiment:")
print(
    test_df["Sentiment"]
    .value_counts()
)


print("\nTRAIN — Platform:")
print(
    train_df["Store"]
    .value_counts()
)

print("\nTEST — Platform:")
print(
    test_df["Store"]
    .value_counts()
)


print("\nTRAIN — Platform × Sentiment:")
print(
    pd.crosstab(
        train_df["Store"],
        train_df["Sentiment"],
    )
)

print("\nTEST — Platform × Sentiment:")
print(
    pd.crosstab(
        test_df["Store"],
        test_df["Sentiment"],
    )
)


# ============================================================
# SAVE REPORT
# ============================================================

report = {
    "dataset": {
        "total_reviews":
            int(len(df)),

        "android_reviews":
            int(
                (df["Store"] == "android")
                .sum()
            ),

        "ios_reviews":
            int(
                (df["Store"] == "ios")
                .sum()
            ),
    },

    "split": {
        "random_state":
            RANDOM_STATE,

        "test_size":
            TEST_SIZE,

        "train_reviews":
            int(len(train_df)),

        "test_reviews":
            int(len(test_df)),
    },

    "train_sentiment": {
        key: int(value)
        for key, value
        in train_df[
            "Sentiment"
        ]
        .value_counts()
        .items()
    },

    "test_sentiment": {
        key: int(value)
        for key, value
        in test_df[
            "Sentiment"
        ]
        .value_counts()
        .items()
    },

    "train_platform": {
        key: int(value)
        for key, value
        in train_df[
            "Store"
        ]
        .value_counts()
        .items()
    },

    "test_platform": {
        key: int(value)
        for key, value
        in test_df[
            "Store"
        ]
        .value_counts()
        .items()
    },
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        report,
        f,
        indent=2,
    )


print("\nSaved:")
print(f"  {TRAIN_PATH}")
print(f"  {TEST_PATH}")
print(f"  {REPORT_PATH}")

print(
    "\nCleaned train/test split "
    "created successfully."
)