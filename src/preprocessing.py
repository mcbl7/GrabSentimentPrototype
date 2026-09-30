"""Shared text preprocessing for all sentiment experiments.

Keep this module deterministic: the same raw review must produce the same cleaned
text for TF-IDF, Word2Vec, evaluation, and the Streamlit demo.
"""
import re
import unicodedata

# Conservative normalization: preserve negators and code-switched sentiment words.
PHRASE_MAP = {
    r"\boks lang\b": "okay lang",
    r"\bk lang\b": "okay lang",
    r"\bsaks lang\b": "sakto lang",
    r"\boks\b": "okay",
}


def preprocess_taglish(text: object) -> str:
    """Normalize a Filipino/English app review without deleting sentiment-bearing words."""
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower().strip()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)

    for pattern, replacement in PHRASE_MAP.items():
        text = re.sub(pattern, replacement, text)

    # Keep letters/numbers plus Filipino characters; turn punctuation into spaces.
    text = re.sub(r"[^\w\sáéíóúñü]", " ", text, flags=re.UNICODE)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
