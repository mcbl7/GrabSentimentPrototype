from pathlib import Path
import re

import pandas as pd
from lingua import Language, LanguageDetectorBuilder


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = Path("data/reviews.xlsx")

RESULTS_DIR = Path("results/language_audit")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MIN_LETTERS = 15
MIN_LANGUAGE_CONFIDENCE = 0.65
MIN_CONFIDENCE_MARGIN = 0.15


# ============================================================
# LANGUAGES
# ============================================================

languages = [
    Language.ENGLISH,
    Language.TAGALOG,
    Language.INDONESIAN,
    Language.MALAY,
]

detector = (
    LanguageDetectorBuilder
    .from_languages(*languages)
    .build()
)


# ============================================================
# HIGH-PRECISION INDONESIAN / MALAY MARKERS
# ============================================================

INDONESIAN_MARKERS = {
    "tidak",
    "saya",
    "yang",
    "dan",
    "dengan",
    "untuk",
    "sudah",
    "udah",
    "banget",
    "gak",
    "nggak",
    "bisa",
    "karena",
    "krn",
    "jadi",
    "kalau",
    "kalo",
    "aplikasi",
    "pesanan",
    "uang",
    "sekarang",
    "makin",
    "tolong",
    "seharusnya",
    "jelek",
    "malah",
    "cuma",
    "pake",
    "pilih",
    "sesuai",
    "belum",
    "belom",
    "masih",
    "driver",
    "tarif",
    "diskon",
}


def tokenize(text):
    return re.findall(
        r"[a-zA-ZÀ-ÿ]+",
        str(text).lower()
    )


def count_letters(text):
    return len(
        re.findall(
            r"[a-zA-ZÀ-ÿ]",
            str(text)
        )
    )


def marker_matches(text):
    tokens = tokenize(text)

    matches = sorted(
        set(tokens)
        & INDONESIAN_MARKERS
    )

    return matches


def detect_language(text):
    text = str(text).strip()

    if count_letters(text) < MIN_LETTERS:
        return {
            "Detected Language": "TOO_SHORT",
            "Language Confidence": 0.0,
            "Second Language": "",
            "Confidence Margin": 0.0,
        }

    values = (
        detector
        .compute_language_confidence_values(text)
    )

    if not values:
        return {
            "Detected Language": "UNKNOWN",
            "Language Confidence": 0.0,
            "Second Language": "",
            "Confidence Margin": 0.0,
        }

    top = values[0]

    second = (
        values[1]
        if len(values) > 1
        else None
    )

    top_score = float(top.value)

    second_score = (
        float(second.value)
        if second
        else 0.0
    )

    return {
        "Detected Language":
            top.language.name,

        "Language Confidence":
            top_score,

        "Second Language":
            second.language.name
            if second
            else "",

        "Confidence Margin":
            top_score - second_score,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("LANGUAGE CONTAMINATION AUDIT")
print("=" * 75)

df = pd.read_excel(DATA_PATH)

required = {
    "Review Text",
    "Store",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

df["Review Text"] = (
    df["Review Text"]
    .fillna("")
    .astype(str)
)

df["Store"] = (
    df["Store"]
    .astype(str)
    .str.lower()
    .str.strip()
)

print(f"\nReviews loaded: {len(df):,}")

if "Country" in df.columns:

    print("\nCountry metadata:")
    print(
        df["Country"]
        .value_counts(
            dropna=False
        )
    )


# ============================================================
# LANGUAGE DETECTION
# ============================================================

print(
    "\nDetecting languages..."
)

records = []

for number, text in enumerate(
    df["Review Text"],
    start=1,
):

    detection = detect_language(
        text
    )

    markers = marker_matches(
        text
    )

    detection[
        "Indonesian Marker Count"
    ] = len(markers)

    detection[
        "Indonesian Markers"
    ] = ", ".join(markers)

    records.append(
        detection
    )

    if number % 1000 == 0:
        print(
            f"  Processed "
            f"{number:,}/{len(df):,}"
        )


audit_df = pd.concat(
    [
        df.reset_index(
            drop=False
        ).rename(
            columns={
                "index":
                "Original Row Index"
            }
        ),
        pd.DataFrame(records),
    ],
    axis=1,
)


# ============================================================
# CONSERVATIVE CONTAMINATION FLAG
#
# We flag Indonesian/Malay only when:
#
# A) detector says Indonesian/Malay
#    with reasonable confidence and margin
#
# OR
#
# B) review contains at least 3
#    distinctive Indonesian markers.
#
# ============================================================

detected_id_ms = (
    audit_df[
        "Detected Language"
    ]
    .isin(
        [
            "INDONESIAN",
            "MALAY",
        ]
    )
)

strong_detection = (
    detected_id_ms
    &
    (
        audit_df[
            "Language Confidence"
        ]
        >= MIN_LANGUAGE_CONFIDENCE
    )
    &
    (
        audit_df[
            "Confidence Margin"
        ]
        >= MIN_CONFIDENCE_MARGIN
    )
)

strong_lexical_evidence = (
    audit_df[
        "Indonesian Marker Count"
    ]
    >= 3
)

audit_df[
    "Likely Indonesian/Malay"
] = (
    strong_detection
    | strong_lexical_evidence
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("DETECTED LANGUAGE SUMMARY")
print("=" * 75)

print(
    audit_df[
        "Detected Language"
    ]
    .value_counts(
        dropna=False
    )
)


print("\n" + "=" * 75)
print("LIKELY INDONESIAN / MALAY CONTAMINATION")
print("=" * 75)

flagged = audit_df[
    audit_df[
        "Likely Indonesian/Malay"
    ]
].copy()

print(
    f"\nFlagged reviews: "
    f"{len(flagged):,}"
)

print(
    f"Percentage of dataset: "
    f"{len(flagged) / len(audit_df) * 100:.2f}%"
)


print(
    "\nFlagged reviews by platform:"
)

print(
    flagged["Store"]
    .value_counts()
)


print(
    "\nFlagged percentage within each platform:"
)

platform_summary = (
    audit_df
    .groupby("Store")
    .agg(
        Total_Reviews=(
            "Review Text",
            "size",
        ),
        Flagged_Reviews=(
            "Likely Indonesian/Malay",
            "sum",
        ),
    )
)

platform_summary[
    "Flagged_Percentage"
] = (
    platform_summary[
        "Flagged_Reviews"
    ]
    / platform_summary[
        "Total_Reviews"
    ]
    * 100
)

print(
    platform_summary
)


# ============================================================
# LANGUAGE × PLATFORM TABLE
# ============================================================

language_platform = pd.crosstab(
    audit_df["Store"],
    audit_df[
        "Detected Language"
    ],
)


# ============================================================
# SAVE OUTPUTS
# ============================================================

audit_df.to_csv(
    RESULTS_DIR
    / "all_reviews_language_audit.csv",
    index=False,
)

flagged.to_csv(
    RESULTS_DIR
    / "likely_indonesian_malay_reviews.csv",
    index=False,
)

platform_summary.to_csv(
    RESULTS_DIR
    / "platform_language_contamination_summary.csv",
)

language_platform.to_csv(
    RESULTS_DIR
    / "language_by_platform.csv",
)


# ============================================================
# SAMPLE FLAGGED REVIEWS
# ============================================================

print(
    "\nSample flagged reviews:\n"
)

sample_columns = [
    "Store",
    "Detected Language",
    "Language Confidence",
    "Indonesian Marker Count",
    "Review Text",
]

for _, row in (
    flagged[
        sample_columns
    ]
    .head(20)
    .iterrows()
):

    print("-" * 75)

    print(
        f"Store: "
        f"{row['Store']}"
    )

    print(
        f"Detected: "
        f"{row['Detected Language']}"
    )

    print(
        f"Confidence: "
        f"{row['Language Confidence']:.3f}"
    )

    print(
        f"Marker count: "
        f"{row['Indonesian Marker Count']}"
    )

    print(
        f"Review: "
        f"{row['Review Text']}"
    )


print("\n" + "=" * 75)
print("LANGUAGE AUDIT COMPLETE")
print("=" * 75)

print(
    f"\nSaved outputs to: "
    f"{RESULTS_DIR}/"
)
