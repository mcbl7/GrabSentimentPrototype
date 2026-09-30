import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 1. Load Data
df = pd.read_excel("data/reviews-labeled.xlsx")

# 2. Replicate standard 80/20 train/test split
train_df, test_df = train_test_split(df, test_size=0.20, random_state=42, stratify=df["Sentiment"])

# 3. Fit Champion Tuned TF-IDF + Balanced SVM (C=0.1)
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vec.fit_transform(train_df["Review Text"].astype(str))
y_train = train_df["Sentiment"]

clf = LinearSVC(class_weight="balanced", random_state=42, C=0.1)
clf.fit(X_train, y_train)

# 4. Evaluate Test Set 1: Android Only (Google Play Store)
android_test = test_df[test_df["Store"] == "android"]
X_android = vec.transform(android_test["Review Text"].astype(str))
y_android = android_test["Sentiment"]
preds_android = clf.predict(X_android)

print("=== TEST SET 1: GOOGLE PLAY STORE (ANDROID) ===")
print(f"Sample Count: {len(android_test)}")
print(f"Accuracy: {accuracy_score(y_android, preds_android):.4f}")
print(f"Macro F1: {f1_score(y_android, preds_android, average='macro'):.4f}\n")
print(classification_report(y_android, preds_android))

# 5. Evaluate Test Set 2: iOS Only (Apple App Store)
ios_test = test_df[test_df["Store"] == "ios"]
X_ios = vec.transform(ios_test["Review Text"].astype(str))
y_ios = ios_test["Sentiment"]
preds_ios = clf.predict(X_ios)

print("=== TEST SET 2: APPLE APP STORE (IOS) ===")
print(f"Sample Count: {len(ios_test)}")
print(f"Accuracy: {accuracy_score(y_ios, preds_ios):.4f}")
print(f"Macro F1: {f1_score(y_ios, preds_ios, average='macro'):.4f}\n")
print(classification_report(y_ios, preds_ios))
