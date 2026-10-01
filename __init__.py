from itertools import combinations
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_text(text: str) -> str:
    """Clean and normalize document text."""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.strip()


def analyze_documents(documents: dict[str, str]) -> dict:
    """
    Compare multiple documents using TF-IDF and cosine similarity.

    Returns:
        matrix: Similarity matrix as percentages.
        pairs: Pairwise similarity results.
    """
    names = list(documents.keys())

    cleaned_documents = [
        clean_text(documents[name])
        for name in names
    ]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    vectors = vectorizer.fit_transform(cleaned_documents)

    similarity_matrix = cosine_similarity(vectors)

    percentage_matrix = np.round(
        similarity_matrix * 100,
        2
    )

    matrix_df = pd.DataFrame(
        percentage_matrix,
        index=names,
        columns=names
    )

    pairs = []

    for i, j in combinations(range(len(names)), 2):
        pairs.append({
            "document_a": names[i],
            "document_b": names[j],
            "similarity": float(
                percentage_matrix[i, j]
            )
        })

    return {
        "matrix": matrix_df,
        "pairs": pairs
    }
