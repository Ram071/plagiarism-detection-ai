from src.detector import analyze_documents, clean_text


def test_clean_text():
    result = clean_text("Hello, WORLD!")

    assert result == "hello world"


def test_identical_documents_have_high_similarity():
    documents = {
        "document1.txt": "Machine learning is useful for data analysis.",
        "document2.txt": "Machine learning is useful for data analysis.",
    }

    result = analyze_documents(documents)

    similarity = result["pairs"][0]["similarity"]

    assert similarity > 99


def test_different_documents_are_compared():
    documents = {
        "document1.txt": "Python programming and algorithms.",
        "document2.txt": "The weather is sunny and the garden has flowers.",
    }

    result = analyze_documents(documents)

    assert len(result["pairs"]) == 1

    similarity = result["pairs"][0]["similarity"]

    assert 0 <= similarity <= 100
