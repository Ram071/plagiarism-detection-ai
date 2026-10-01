import streamlit as st
from src.detector import analyze_documents

st.set_page_config(page_title="AI Plagiarism Detector", page_icon="🔎", layout="wide")

st.title("🔎 AI Plagiarism Detector")
st.caption("Compare documents using lexical and semantic similarity.")

uploaded = st.file_uploader(
    "Upload two or more text documents",
    type=["txt"],
    accept_multiple_files=True,
)

threshold = st.slider(
    "Plagiarism / high-similarity threshold (%)",
    min_value=0,
    max_value=100,
    value=70,
    step=5,
)

if uploaded:
    if len(uploaded) < 2:
        st.warning("Please upload at least two .txt files.")
    else:
        documents = {}
        for file in uploaded:
            documents[file.name] = file.read().decode("utf-8", errors="ignore")

        result = analyze_documents(documents)

        st.subheader("Similarity Matrix")
        st.dataframe(result["matrix"], use_container_width=True)

        st.subheader("Pairwise Results")
        for item in result["pairs"]:
            status = "HIGH SIMILARITY" if item["similarity"] >= threshold else "LOW/MODERATE"
            st.write(
                f"**{item['document_a']} ↔ {item['document_b']}** — "
                f"**{item['similarity']:.2f}%** — `{status}`"
            )

        st.info(
            "Similarity is an indicator, not proof of plagiarism. "
            "Review the matched content and context before making an academic decision."
        )
else:
    st.markdown(
        """
        ### How to use
        1. Upload two or more `.txt` files.
        2. Choose the similarity threshold.
        3. Review the similarity matrix and pairwise results.
        """
    )
