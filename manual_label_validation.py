import pandas as pd

# Load the label-noise results
df = pd.read_excel("data/reviews-labeled.xlsx")

# Recalculate VADER scores
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()


def get_vader_score(text):
    return analyzer.polarity_scores(str(text))["compound"]


df["VADER_Score"] = df["Review Text"].apply(get_vader_score)


def classify_vader(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


df["VADER_Sentiment"] = df["VADER_Score"].apply(classify_vader)


# Find potential inconsistencies
potential_noise = df[
    (
        (df["Sentiment"] == "Neutral")
        & (df["VADER_Sentiment"] != "Neutral")
    )
    |
    (
        (df["Sentiment"] == "Positive")
        & (df["VADER_Sentiment"] == "Negative")
    )
    |
    (
        (df["Sentiment"] == "Negative")
        & (df["VADER_Sentiment"] == "Positive")
    )
].copy()


# Select samples from each original label
negative_sample = potential_noise[
    potential_noise["Sentiment"] == "Negative"
].sample(
    min(25, len(potential_noise[potential_noise["Sentiment"] == "Negative"])),
    random_state=42
)

neutral_sample = potential_noise[
    potential_noise["Sentiment"] == "Neutral"
].sample(
    min(25, len(potential_noise[potential_noise["Sentiment"] == "Neutral"])),
    random_state=42
)

positive_sample = potential_noise[
    potential_noise["Sentiment"] == "Positive"
].sample(
    min(25, len(potential_noise[potential_noise["Sentiment"] == "Positive"])),
    random_state=42
)


# Combine samples
validation_sample = pd.concat(
    [
        negative_sample,
        neutral_sample,
        positive_sample
    ]
)


# Add a manual validation column
validation_sample["Manual_Label"] = ""


# Select useful columns
validation_sample = validation_sample[
    [
        "Star Rating",
        "Sentiment",
        "VADER_Sentiment",
        "VADER_Score",
        "Review Text",
        "Manual_Label"
    ]
]


# Save for manual inspection
output_file = "data/manual-validation-sample.xlsx"

validation_sample.to_excel(
    output_file,
    index=False
)


print("Potential inconsistencies:", len(potential_noise))

print("\nManual validation sample created!")

print("Negative sample:", len(negative_sample))
print("Neutral sample:", len(neutral_sample))
print("Positive sample:", len(positive_sample))

print("\nTotal validation sample:", len(validation_sample))

print("\nSaved as:")
print(output_file)