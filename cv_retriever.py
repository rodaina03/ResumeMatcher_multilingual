import os
import re
import hashlib
import pdfplumber
import faiss
import numpy as np
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==== Config ====
PDF_FOLDER = "cvs/"
MODEL_NAME = "intfloat/multilingual-e5-large"
TOP_K = 5
EMBEDDING_WEIGHT = 0.95
TFIDF_WEIGHT = 0.05
MIN_FINAL_SCORE = 0.70
MIN_EMBEDDING_SCORE = 0.75



print(f"Loading model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# ==== Text utilities ====
def clean_text(text: str) -> str:
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def chunk_text(text: str, max_words: int = 120) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def embed_texts(texts: List[str], is_query: bool = False) -> np.ndarray:
    prefix = "query: " if is_query else "passage: "
    embeddings = []
    for text in texts:
        chunks = chunk_text(text)
        if not chunks:
            chunks = [""]
        prefixed_chunks = [prefix + chunk for chunk in chunks]
        chunk_embeds = model.encode(prefixed_chunks, normalize_embeddings=True)
        avg_embed = np.mean(chunk_embeds, axis=0)
        embeddings.append(avg_embed)
    return np.vstack(embeddings)

# ==== Read PDFs ====
def read_pdfs(folder: str) -> List[Tuple[str, str]]:
    docs = []
    for filename in os.listdir(folder):
        if filename.lower().endswith(".pdf"):
            with pdfplumber.open(os.path.join(folder, filename)) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            docs.append((filename, clean_text(text)))
    return docs

# ==== FAISS index ====
def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index

# ==== Main matcher ====
def hybrid_match(job_desc: str, cv_texts: List[Tuple[str, str]], top_k: int = TOP_K):
    use_tfidf = len(job_desc.strip().split()) >= 3

    filenames, raw_texts = zip(*cv_texts)

    print("Embedding CVs and job description...")
    cv_embeddings = embed_texts(raw_texts)
    job_embedding = embed_texts([job_desc], is_query=True)

    faiss_index = build_faiss_index(cv_embeddings)
    embed_scores, embed_indices = faiss_index.search(job_embedding, len(cv_texts))
    embed_scores = embed_scores[0]
    embed_order = embed_indices[0]

    print("Computing TF-IDF similarity...")
    tfidf = TfidfVectorizer().fit(raw_texts + (job_desc,))
    tfidf_matrix = tfidf.transform(raw_texts)
    tfidf_query = tfidf.transform([job_desc])
    tfidf_scores = cosine_similarity(tfidf_query, tfidf_matrix)[0]

    combined = []
    for i in range(len(cv_texts)):
        idx = embed_order[i]
        filename = filenames[idx]
        content = raw_texts[idx]
        embed_score = embed_scores[i]
        tfidf_score = tfidf_scores[idx]
        if use_tfidf:
            final_score = EMBEDDING_WEIGHT * embed_score + TFIDF_WEIGHT * tfidf_score
        else:
            final_score = embed_score
        combined.append((filename, final_score, embed_score, tfidf_score, content))

    # Remove duplicates based on content hash
    seen = set()
    unique = []
    for item in combined:
        _, _, _, _, content = item
        digest = hashlib.md5(content.encode()).hexdigest()
        if digest not in seen:
            unique.append(item)
            seen.add(digest)

    # Filter by final + embedding threshold
    filtered = []
    for filename, final_score, embed_score, tfidf_score, content in unique:
        if final_score >= MIN_FINAL_SCORE and embed_score >= MIN_EMBEDDING_SCORE:
            filtered.append((filename, final_score, embed_score, tfidf_score, content))
        else:
            print(f"❌ Skipped: {filename} | Final={final_score:.3f}, E5={embed_score:.3f}, TF-IDF={tfidf_score:.3f}")

    filtered.sort(key=lambda x: x[1], reverse=True)
    return filtered[:top_k]

# ==== Entry Point ====
if __name__ == "__main__":
    print("Reading CVs...")
    cv_texts = read_pdfs(PDF_FOLDER)

    job_desc = input("\nEnter job description: ").strip()
    job_desc_clean = clean_text(job_desc)

    print("\nMatching resumes...")
    results = hybrid_match(job_desc_clean, cv_texts)

    if not results:
        print("\u26a0\ufe0f No resumes matched above the threshold.")
    else:
        print("\nTop matching CVs:")
        for rank, (filename, final, embed, tfidf, content) in enumerate(results, 1):
            print(f"{rank}. {filename} — Final: {final:.3f} | E5: {embed:.3f}, TF-IDF: {tfidf:.3f}")
            print(content[:250].strip() + "...")
            print("-" * 60)
