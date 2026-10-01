from itertools import combinations
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Normalize text while keeping word boundaries."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()


def analyze_documents(documents: dict[str, str]) -> dict:
    """Calculate pairwise TF-IDF cosine similarity for uploaded documents."""
    names = list(documents.keys())
    cleaned = [clean_text(documents[name]) for name in names]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
    )
    matrix_vectors = vectorizer.fit_transform(cleaned)
    similarities = cosine_similarity(matrix_vectors)

    percent_matrix = np.round(similarities * 100, 2)
    matrix_df = pd.DataFrame(percent_matrix, index=names, columns=names)

    pairs = []
    for i, j in combinations(range(len(names)), 2):
        pairs.append(
            {
                "document_a": names[i],
                "document_b": names[j],
                "similarity": float(percent_matrix[i, j]),
            }
        )

    return {"matrix": matrix_df, "pairs": pairs}
