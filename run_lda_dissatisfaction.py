import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_excel("data/reviews-labeled.xlsx")
neg_df = df[df["Sentiment"] == "Negative"].dropna(subset=["Review Text"])

# Custom stopword list filtering English & Tagalog conversational noise
custom_stops = [
    'ang', 'ng', 'sa', 'na', 'mga', 'ko', 'mo', 'nga', 'naman', 'pa', 'lang', 'ba', 
    'po', 'pero', 'at', 'ni', 'kay', 'lahat', 'grab', 'app', 'the', 'and', 'to', 
    'is', 'in', 'it', 'you', 'that', 'my', 'of', 'for', 'with', 'this', 'on'
]

cv = CountVectorizer(max_features=2000, stop_words=custom_stops, ngram_range=(1, 2), min_df=5)
dtm = cv.fit_transform(neg_df["Review Text"].astype(str))

lda = LatentDirichletAllocation(n_components=4, random_state=42)
lda.fit(dtm)

print("=== TOP DISSATISFACTION TOPICS EXTRACTED (LDA) ===")
feature_names = cv.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
    print(f"Topic #{idx + 1}: {', '.join(top_words)}")
