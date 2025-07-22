import streamlit as st
import re
import fitz
import numpy as np
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

# Clean text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Read uploaded PDFs
def read_uploaded_pdfs(uploaded_files):
    cv_texts = []
    cv_filenames = []
    for uploaded_file in uploaded_files:
        try:
            with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
                text = "".join([page.get_text() for page in doc])
            text = clean_text(text)
            if text:
                cv_texts.append(text)
                cv_filenames.append(uploaded_file.name)
                # st.success(f"✅ Loaded {uploaded_file.name} [lang={detect(text)}]")
        except Exception as e:
            st.warning(f"⚠️ Failed to read {uploaded_file.name}: {e}")
    st.success(f"✅ Loaded {len(cv_texts)} CVs")
    return cv_texts, cv_filenames

# Chunk text
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# UI layout
st.set_page_config(page_icon="📄",page_title="Multilingual CV Matcher", layout="wide")
st.title("📄 Multilingual CV Similarity Matcher")

st.markdown("### 📝 Step 1: Paste the Job Description")
jd_text = st.text_area("Enter the job description here:", height=250)

st.markdown("### 📤 Step 2: Upload CVs (PDF format)")
uploaded_cvs = st.file_uploader("Upload one or more CVs", type=["pdf"], accept_multiple_files=True)

top_k = st.slider("How many top matches to show?", min_value=1, max_value=len(uploaded_cvs), value=5)

match_clicked = st.button("🔍 Match CVs", key="match_button")

if match_clicked:
    if not jd_text:
        st.warning("⚠️ Please paste a job description before matching.")
    elif not uploaded_cvs:
        st.warning("⚠️ Please upload at least one CV.")
    else:
        with st.spinner("🔍 Loading embedding model..."):
            model = SentenceTransformer("intfloat/multilingual-e5-large")

        with st.spinner("📂 Reading uploaded CVs..."):
            cv_texts, cv_filenames = read_uploaded_pdfs(uploaded_cvs)

        if not cv_texts:
            st.error("❌ No valid CVs extracted.")
        else:
            with st.spinner("🔗 Encoding and indexing CVs..."):
                all_chunks = []
                chunk_map = []
                for i, text in enumerate(cv_texts):
                    chunks = chunk_text(text)
                    all_chunks.extend([f"passage: {ch}" for ch in chunks])
                    chunk_map.extend([(i, j) for j in range(len(chunks))])

                chunk_embeddings = model.encode(all_chunks, convert_to_numpy=True)
                chunk_embeddings = normalize(chunk_embeddings, axis=1)

                index = faiss.IndexFlatIP(chunk_embeddings.shape[1])
                index.add(chunk_embeddings)

                query_embedding = model.encode([f"query: {jd_text}"], convert_to_numpy=True)
                query_embedding = normalize(query_embedding, axis=1)

                scores, indices = index.search(query_embedding, len(all_chunks))

            # Best chunk per CV
            seen = {}
            for i in indices[0]:
                cv_idx, chunk_idx = chunk_map[i]
                score = float(np.dot(query_embedding[0], chunk_embeddings[i]))
                if cv_idx not in seen or score > seen[cv_idx][0]:
                    seen[cv_idx] = (score, chunk_idx)

            ranked = sorted([(cv_filenames[i], score, chunk_text(cv_texts[i])[j])
                             for i, (score, j) in seen.items()],
                            key=lambda x: x[1], reverse=True)

            st.markdown("## 🎯 Top Matching CVs")
            for rank, (filename, score, best_chunk) in enumerate(ranked[:top_k], 1):
                st.markdown(f"**Rank {rank} — `{filename}` — Score: `{score:.4f}`**")
                with st.expander("Show best-matching chunk"):
                    st.write(best_chunk.strip())
