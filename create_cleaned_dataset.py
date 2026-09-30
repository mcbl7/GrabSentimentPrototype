from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ORIGINAL_DATA_PATH = Path("data/reviews.xlsx")

LANGUAGE_AUDIT_PATH = Path(
    "results/language_audit/all_reviews_language_audit.csv"
)

MANUAL_REVIEW_PATH = Path(
    "results/language_audit/borderline_manual_review.xlsx"
)

OUTPUT_XLSX_PATH = Path(
    "data/reviews_cleaned.xlsx"
)

OUTPUT_CSV_PATH = Path(
    "data/reviews_cleaned.csv"
)

REPORT_PATH = Path(
    "results/language_audit/cleaning_summary.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

print("=" * 75)
print("CREATE CLEANED DATASET")
print("=" * 75)

original_df = pd.read_excel(
    ORIGINAL_DATA_PATH
)

audit_df = pd.read_csv(
    LANGUAGE_AUDIT_PATH
)

manual_df = pd.read_excel(
    MANUAL_REVIEW_PATH
)


print(
    f"\nOriginal reviews: "
    f"{len(original_df):,}"
)


# ============================================================
# NORMALIZE BOOLEAN FLAG
# ============================================================

audit_df[
    "Likely Indonesian/Malay"
] = (
    audit_df[
        "Likely Indonesian/Malay"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("true")
)


# ============================================================
# AUTOMATIC EXCLUSIONS
# ============================================================

auto_remove_df = audit_df[
    audit_df[
        "Likely Indonesian/Malay"
    ]
].copy()

auto_indices = set(
    auto_remove_df[
        "Original Row Index"
    ]
    .astype(int)
    .tolist()
)


print(
    f"Automatic removals: "
    f"{len(auto_indices):,}"
)


# ============================================================
# MANUAL EXCLUSIONS
# ============================================================

manual_df[
    "Manual Decision"
] = (
    manual_df[
        "Manual Decision"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.upper()
)


valid_decisions = {
    "REMOVE",
    "KEEP",
}

invalid_decisions = manual_df[
    ~manual_df[
        "Manual Decision"
    ].isin(
        valid_decisions
    )
]

if not invalid_decisions.empty:
    raise ValueError(
        "\nSome manual review rows do not contain "
        "REMOVE or KEEP.\n"
        f"Invalid rows: {len(invalid_decisions)}"
    )


manual_remove_df = manual_df[
    manual_df[
        "Manual Decision"
    ] == "REMOVE"
].copy()

manual_keep_df = manual_df[
    manual_df[
        "Manual Decision"
    ] == "KEEP"
].copy()


manual_remove_indices = set(
    manual_remove_df[
        "Original Row Index"
    ]
    .astype(int)
    .tolist()
)


print(
    f"Manual REMOVE decisions: "
    f"{len(manual_remove_indices):,}"
)

print(
    f"Manual KEEP decisions: "
    f"{len(manual_keep_df):,}"
)


# ============================================================
# VERIFY NO UNEXPECTED OVERLAP
# ============================================================

overlap = (
    auto_indices
    & manual_remove_indices
)

print(
    f"Automatic/manual overlap: "
    f"{len(overlap):,}"
)

if overlap:
    raise ValueError(
        "Automatic and manual removal sets overlap. "
        "This should normally be zero because the manual "
        "sheet contains only borderline cases."
    )


# ============================================================
# COMBINE REMOVALS
# ============================================================

all_remove_indices = (
    auto_indices
    | manual_remove_indices
)

print(
    f"\nTotal exclusions: "
    f"{len(all_remove_indices):,}"
)


# Validate indices
invalid_indices = [
    idx
    for idx in all_remove_indices
    if (
        idx < 0
        or idx >= len(original_df)
    )
]

if invalid_indices:
    raise ValueError(
        f"Invalid original row indices found: "
        f"{invalid_indices[:10]}"
    )


# ============================================================
# CREATE CLEANED DATASET
# ============================================================

cleaned_df = original_df.drop(
    index=list(
        all_remove_indices
    )
).copy()

cleaned_df = cleaned_df.reset_index(
    drop=True
)


print(
    f"Reviews remaining: "
    f"{len(cleaned_df):,}"
)


# ============================================================
# PLATFORM COUNTS
# ============================================================

print(
    "\nPlatform distribution "
    "after cleaning:"
)

if "Store" in cleaned_df.columns:

    cleaned_df["Store"] = (
        cleaned_df["Store"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    platform_counts = (
        cleaned_df[
            "Store"
        ]
        .value_counts()
    )

    print(
        platform_counts
    )

else:
    platform_counts = (
        pd.Series(
            dtype=int
        )
    )


# ============================================================
# SENTIMENT / RATING SUMMARY
# ============================================================

if "Sentiment" in cleaned_df.columns:

    print(
        "\nSentiment distribution "
        "after cleaning:"
    )

    print(
        cleaned_df[
            "Sentiment"
        ]
        .value_counts()
    )


if "Rating" in cleaned_df.columns:

    print(
        "\nRating distribution "
        "after cleaning:"
    )

    print(
        cleaned_df[
            "Rating"
        ]
        .value_counts()
        .sort_index()
    )


# ============================================================
# SAVE CLEANED DATA
# ============================================================

cleaned_df.to_excel(
    OUTPUT_XLSX_PATH,
    index=False
)

cleaned_df.to_csv(
    OUTPUT_CSV_PATH,
    index=False
)


# ============================================================
# SAVE CLEANING REPORT
# ============================================================

summary_rows = [
    {
        "Metric":
            "Original Reviews",
        "Count":
            len(original_df),
    },
    {
        "Metric":
            "Automatic Language Exclusions",
        "Count":
            len(auto_indices),
    },
    {
        "Metric":
            "Manual Borderline Exclusions",
        "Count":
            len(manual_remove_indices),
    },
    {
        "Metric":
            "Manual Borderline Keeps",
        "Count":
            len(manual_keep_df),
    },
    {
        "Metric":
            "Total Exclusions",
        "Count":
            len(all_remove_indices),
    },
    {
        "Metric":
            "Final Cleaned Reviews",
        "Count":
            len(cleaned_df),
    },
]


if "android" in platform_counts.index:
    summary_rows.append({
        "Metric":
            "Final Android Reviews",
        "Count":
            int(
                platform_counts[
                    "android"
                ]
            ),
    })


if "ios" in platform_counts.index:
    summary_rows.append({
        "Metric":
            "Final iOS Reviews",
        "Count":
            int(
                platform_counts[
                    "ios"
                ]
            ),
    })


summary_df = pd.DataFrame(
    summary_rows
)

summary_df.to_csv(
    REPORT_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n" + "=" * 75
)

print(
    "CLEANING COMPLETE"
)

print(
    "=" * 75
)

print(
    f"\nSaved cleaned Excel file:\n"
    f"  {OUTPUT_XLSX_PATH}"
)

print(
    f"\nSaved cleaned CSV file:\n"
    f"  {OUTPUT_CSV_PATH}"
)

print(
    f"\nSaved cleaning report:\n"
    f"  {REPORT_PATH}"
)