from pathlib import Path
import re
from collections import Counter

import joblib
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# CONFIGURATION

DATA_PATH = Path("data/reviews_cleaned.xlsx")
MODEL_PATH = Path("models/final_cleaned/tfidf_svm_selected.joblib")

RESULTS_DIR = Path("results/dissatisfaction_cleaned")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_TOPICS = 6
TOP_WORDS = 12
RANDOM_STATE = 42

# OPERATIONAL KWIC LEXICON

KWIC_CATEGORIES = {

    "Pricing and Fare": [
        "price",
        "pricing",
        "fare",
        "expensive",
        "mahal",
        "overpriced",
        "surge",
        "fee",
        "fees",
        "charge",
        "charged",
    ],

    "Waiting Time and Delay": [
        "wait",
        "waiting",
        "late",
        "delay",
        "delayed",
        "slow",
        "tagal",
        "matagal",
        "mabagal",
        "antagal",
    ],

    "Order Fulfillment and Accuracy": [
        "order",
        "wrong order",
        "missing",
        "missing item",
        "kulang",
        "incorrect",
        "cancelled",
        "canceled",
        "cancel",
        "food",
        "item",
    ],

    "Driver Rider Courier Conduct": [
        "driver",
        "rider",
        "courier",
        "rude",
        "attitude",
        "behavior",
        "behaviour",
        "unprofessional",
        "service",
    ],

    "Application Performance": [
        "app",
        "application",
        "crash",
        "crashes",
        "lag",
        "laggy",
        "bug",
        "bugs",
        "loading",
        "login",
        "otp",
        "gps",
        "location",
        "update",
    ],

    "Payment and Refund": [
        "payment",
        "pay",
        "paid",
        "card",
        "cash",
        "gcash",
        "wallet",
        "refund",
        "refunded",
        "money",
        "balance",
    ],

    "Customer Support": [
        "support",
        "customer service",
        "help",
        "hotline",
        "agent",
        "complaint",
        "response",
        "respond",
        "contact",
    ],
}


# ============================================================
# LOAD DATA + FINAL MODEL
# ===========================================

print("=" * 75)
print("CUSTOMER DISSATISFACTION FACTOR ANALYSIS")
print("=" * 75)

df = pd.read_excel(DATA_PATH)

required_columns = {
    "Review Text",
    "Store",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(
        f"reviews_cleaned.xlsx is missing columns: {sorted(missing)}"
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


model = joblib.load(MODEL_PATH)

print(f"\nLoaded reviews: {len(df):,}")
print(f"Loaded model: {MODEL_PATH}")

# PREDICT ALL REVIEWS

print("\nGenerating sentiment predictions...")

df["Predicted Sentiment"] = model.predict(
    df["Review Text"]
)

negative_df = df[
    df["Predicted Sentiment"] == "Negative"
].copy()


print(
    f"Predicted Negative reviews: "
    f"{len(negative_df):,}"
)

print("\nNegative reviews by platform:")
print(
    negative_df["Store"]
    .value_counts()
)

# SAVE NEGATIVE REVIEW

negative_df.to_csv(
    RESULTS_DIR / "predicted_negative_reviews.csv",
    index=False,
)

# KWIC ANALYSIS

print("\n" + "=" * 75)
print("KWIC OPERATIONAL FACTOR ANALYSIS")
print("=" * 75)


def contains_keyword(text, keyword):

    text = str(text).lower()

    # Phrase / token-aware search
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"

    return bool(
        re.search(
            pattern,
            text,
        )
    )


kwic_rows = []
kwic_summary = []


for category, keywords in KWIC_CATEGORIES.items():

    category_review_ids = set()

    keyword_counter = Counter()

    for idx, row in negative_df.iterrows():

        text = row["Review Text"]

        matched_keywords = [
            keyword
            for keyword in keywords
            if contains_keyword(
                text,
                keyword,
            )
        ]

        if matched_keywords:

            category_review_ids.add(idx)

            for keyword in matched_keywords:
                keyword_counter[keyword] += 1

            kwic_rows.append({
                "Review Index": idx,
                "Store": row["Store"],
                "Category": category,
                "Matched Keywords":
                    ", ".join(matched_keywords),
                "Review Text": text,
            })


    count = len(
        category_review_ids
    )

    percentage = (
        count / len(negative_df) * 100
        if len(negative_df) > 0
        else 0
    )

    top_keywords = ", ".join(
        [
            f"{word} ({freq})"
            for word, freq
            in keyword_counter.most_common(10)
        ]
    )

    kwic_summary.append({
        "Category": category,
        "Matched Negative Reviews": count,
        "Percentage of Negative Reviews":
            percentage,
        "Top Matched Keywords":
            top_keywords,
    })


kwic_summary_df = pd.DataFrame(
    kwic_summary
).sort_values(
    by="Matched Negative Reviews",
    ascending=False,
)

kwic_rows_df = pd.DataFrame(
    kwic_rows
)


kwic_summary_df.to_csv(
    RESULTS_DIR / "kwic_factor_summary.csv",
    index=False,
)

kwic_rows_df.to_csv(
    RESULTS_DIR / "kwic_review_matches.csv",
    index=False,
)


print(
    "\nKWIC Factor Summary:\n"
)

print(
    kwic_summary_df[
        [
            "Category",
            "Matched Negative Reviews",
            "Percentage of Negative Reviews",
        ]
    ].to_string(
        index=False
    )
)

# TEXT CLEANING FOR LDA

CUSTOM_STOP_WORDS = [
    # Project-specific / generic English terms
    "grab",
    "app",
    "apps",
    "please",
    "im",
    "ive",
    "really",
    "still",
    "even",
    "also",
    "get",
    "got",
    "use",
    "using",
    "used",
    "one",
    "like",
    "would",
    "could",

    # Filipino / Tagalog function words
    "na",
    "ng",
    "sa",
    "ang",
    "yung",
    "yong",
    "iyong",
    "ko",
    "mo",
    "ka",
    "ako",
    "siya",
    "niya",
    "nya",
    "kami",
    "tayo",
    "sila",
    "namin",
    "natin",
    "nila",
    "pa",
    "lang",
    "mga",
    "ay",
    "at",
    "para",
    "ito",
    "yan",
    "yun",
    "yon",
    "nyo",
    "ninyo",
    "naman",
    "din",
    "rin",
    "daw",
    "raw",
    "nga",
    "kasi",
    "pero",
    "nag",
    "mag",
        # for LDA topic clarity only
    "wala",
    "walang",
    "hindi",
    "di",
    "niyo",
    "kayo",
    "tapos",
    "pag",
    "kahit",
    "sana",
]


def clean_for_lda(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text,
    )

    text = re.sub(
        r"[^a-zA-ZÀ-ÿ\s]",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


lda_texts = (
    negative_df["Review Text"]
    .apply(clean_for_lda)
    .tolist()
)

# LDA VECTORIZATION

print("\n" + "=" * 75)
print("LDA TOPIC MODELING")
print("=" * 75)


vectorizer = CountVectorizer(
    lowercase=True,
    stop_words="english",
    min_df=5,
    max_df=0.85,
    max_features=5000,
)

document_term_matrix = (
    vectorizer.fit_transform(
        lda_texts
    )
)


# Remove project-specific generic terms from vocabulary
feature_names = vectorizer.get_feature_names_out()

keep_mask = [
    word not in CUSTOM_STOP_WORDS
    for word in feature_names
]

keep_indices = [
    i
    for i, keep
    in enumerate(keep_mask)
    if keep
]

document_term_matrix = (
    document_term_matrix[
        :,
        keep_indices
    ]
)

feature_names = (
    feature_names[
        keep_indices
    ]
)


print(
    f"Negative review documents: "
    f"{document_term_matrix.shape[0]:,}"
)

print(
    f"LDA vocabulary size: "
    f"{document_term_matrix.shape[1]:,}"
)

# FIT LDA

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=RANDOM_STATE,
    learning_method="batch",
    max_iter=30,
)

topic_distributions = lda.fit_transform(
    document_term_matrix
)

# EXTRACT TOP WORDS

topic_rows = []

for topic_number, topic_weights in enumerate(
    lda.components_,
    start=1,
):

    top_indices = (
        topic_weights
        .argsort()[-TOP_WORDS:]
        [::-1]
    )

    top_words = [
        feature_names[index]
        for index in top_indices
    ]

    topic_rows.append({
        "Topic": topic_number,
        "Top Words":
            ", ".join(top_words),
    })


topic_df = pd.DataFrame(
    topic_rows
)

topic_df.to_csv(
    RESULTS_DIR / "lda_topics.csv",
    index=False,
)


print("\nLDA Topics:\n")

print(
    topic_df.to_string(
        index=False
    )
)

# ASSIGN DOMINANT TOPIC TO EACH NEGATIVE REVIEW

dominant_topics = (
    topic_distributions
    .argmax(axis=1)
    + 1
)

topic_strength = (
    topic_distributions
    .max(axis=1)
)


lda_review_df = negative_df.copy()

lda_review_df[
    "Dominant Topic"
] = dominant_topics

lda_review_df[
    "Topic Strength"
] = topic_strength


lda_review_df.to_csv(
    RESULTS_DIR / "negative_reviews_with_topics.csv",
    index=False,
)

# TOPIC DISTRIBUTION

topic_counts = (
    lda_review_df[
        "Dominant Topic"
    ]
    .value_counts()
    .sort_index()
)

topic_summary_rows = []

for topic in range(
    1,
    N_TOPICS + 1,
):

    count = int(
        topic_counts.get(
            topic,
            0,
        )
    )

    percentage = (
        count
        / len(lda_review_df)
        * 100
        if len(lda_review_df) > 0
        else 0
    )

    words = topic_df.loc[
        topic_df["Topic"] == topic,
        "Top Words",
    ].iloc[0]

    topic_summary_rows.append({
        "Topic": topic,
        "Number of Reviews": count,
        "Percentage":
            percentage,
        "Top Words":
            words,
    })


topic_summary_df = pd.DataFrame(
    topic_summary_rows
)


topic_summary_df.to_csv(
    RESULTS_DIR
    / "lda_topic_distribution.csv",
    index=False,
)


print(
    "\nTopic Distribution:\n"
)

print(
    topic_summary_df.to_string(
        index=False
    )
)

# PLATFORM & TOPIC DISTRIBUTION

platform_topic = pd.crosstab(
    lda_review_df["Store"],
    lda_review_df["Dominant Topic"],
)

platform_topic.to_csv(
    RESULTS_DIR
    / "lda_platform_topic_distribution.csv"
)

# SAVE FULL PREDICTION DATASET

df.to_csv(
    RESULTS_DIR
    / "all_reviews_with_predictions.csv",
    index=False,
)


print("\n" + "=" * 75)
print("DISSATISFACTION ANALYSIS COMPLETED")
print("=" * 75)

print("\nSaved outputs to:")
print(f"  {RESULTS_DIR}/")