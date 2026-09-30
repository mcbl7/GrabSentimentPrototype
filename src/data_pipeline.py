"""Create one reproducible holdout split shared by every model experiment."""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split

REQUIRED_COLUMNS = {"Store", "Country", "Review ID", "Review Date", "Star Rating", "Review Text"}
VALID_STORES = {"android", "ios"}


def assign_sentiment(rating: int) -> str:
    if rating <= 2:
        return "Negative"
    if rating == 3:
        return "Neutral"
    return "Positive"


def prepare_dataframe(path: str | Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_excel(path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    report = {"rows_loaded": int(len(df))}
    df = df.copy()
    df["Store"] = df["Store"].astype(str).str.strip().str.lower()
    invalid_stores = sorted(set(df["Store"].dropna()) - VALID_STORES)
    if invalid_stores:
        raise ValueError(f"Unexpected Store values: {invalid_stores}")

    df["Star Rating"] = pd.to_numeric(df["Star Rating"], errors="coerce")
    before = len(df)
    df = df[df["Star Rating"].between(1, 5, inclusive="both")].copy()
    report["invalid_rating_rows_removed"] = int(before - len(df))

    before = len(df)
    df = df.dropna(subset=["Review Text"])
    df["Review Text"] = df["Review Text"].astype(str).str.strip()
    df = df[df["Review Text"].ne("")].copy()
    report["missing_or_blank_text_rows_removed"] = int(before - len(df))

    # Review ID is the safest available observation identifier. Do not delete repeated
    # short texts (e.g. "good") because different users can legitimately write them.
    report["duplicate_review_ids_found"] = int(df.duplicated(subset=["Review ID"]).sum())
    df = df.drop_duplicates(subset=["Review ID"], keep="first").copy()

    df["Sentiment"] = df["Star Rating"].astype(int).map(assign_sentiment)
    df["Review Date"] = pd.to_datetime(df["Review Date"], errors="coerce", utc=True)
    df["Stratify Key"] = df["Store"] + "__" + df["Sentiment"]

    report["rows_after_cleaning"] = int(len(df))
    report["platform_counts"] = {str(k): int(v) for k, v in df["Store"].value_counts().items()}
    report["sentiment_counts"] = {str(k): int(v) for k, v in df["Sentiment"].value_counts().items()}
    report["joint_counts"] = {str(k): int(v) for k, v in df["Stratify Key"].value_counts().items()}
    return df, report


def create_fixed_split(df: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    counts = df["Stratify Key"].value_counts()
    if (counts < 2).any():
        raise ValueError(f"Cannot stratify groups with fewer than 2 rows: {counts[counts < 2].to_dict()}")
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["Stratify Key"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def save_split(input_path="data/reviews.xlsx", output_dir="data/splits"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, report = prepare_dataframe(input_path)
    train_df, test_df = create_fixed_split(df)
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    report["train_rows"] = int(len(train_df))
    report["test_rows"] = int(len(test_df))
    report["test_fraction"] = 0.20
    report["random_state"] = 42
    with open(out / "split_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report
