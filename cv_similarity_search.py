import os
import re
import fitz
import numpy as np
import faiss
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import argparse

# Clean text
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Read PDFs and extract text
def read_pdfs_from_folder(folder_path):
    cv_texts = []
    cv_filenames = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            try:
                with fitz.open(pdf_path) as doc:
                    text = "".join([page.get_text() for page in doc])
                text = clean_text(text)
                if text:
                    cv_texts.append(text)
                    cv_filenames.append(filename)
                    print(f"✅ Loaded {filename} [lang={detect(text)}]")
            except Exception as e:
                print(f"⚠️ Failed to read {filename}: {e}")
    return cv_texts, cv_filenames

# Split text into fixed-size chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Main logic
def main():
    parser = argparse.ArgumentParser(description="CV Similarity Search with Chunking")
    parser.add_argument("--cv_folder", type=str, default="cvs", help="Folder with CV PDFs")
    parser.add_argument("--jd_file", type=str, required=True, help="Path to job description .txt file")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top matches to return")
    args = parser.parse_args()

    print("🔍 Loading embedding model...")
    model = SentenceTransformer("intfloat/multilingual-e5-large")

    print(f"\n📂 Reading CVs from: {args.cv_folder}")
    cv_texts, cv_filenames = read_pdfs_from_folder(args.cv_folder)
    if not cv_texts:
        print("❌ No valid CVs found.")
        return

    print(f"\n🔗 Chunking and encoding {len(cv_texts)} CVs...")
    all_chunks = []
    chunk_map = []  # (cv_index, chunk_index)
    for i, text in enumerate(cv_texts):
        chunks = chunk_text(text)
        all_chunks.extend([f"passage: {ch}" for ch in chunks])
        chunk_map.extend([(i, j) for j in range(len(chunks))])

    chunk_embeddings = model.encode(all_chunks, convert_to_numpy=True)
    chunk_embeddings = normalize(chunk_embeddings, axis=1)

    dim = chunk_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(chunk_embeddings)

    # Load JD
    with open(args.jd_file, 'r', encoding='utf-8') as f:
        job_description = clean_text(f.read())
    print("\n📄 Job description loaded.")

    query_embedding = model.encode([f"query: {job_description}"], convert_to_numpy=True)
    query_embedding = normalize(query_embedding, axis=1)

    print("🚀 Running similarity search...")
    scores, indices = index.search(query_embedding, len(all_chunks))

    # Collect best-scoring chunk per CV
    seen = {}
    for i in indices[0]:
        cv_idx, chunk_idx = chunk_map[i]
        score = float(np.dot(query_embedding[0], chunk_embeddings[i]))
        if cv_idx not in seen or score > seen[cv_idx][0]:
            seen[cv_idx] = (score, chunk_idx)

    ranked = sorted([(cv_filenames[i], score, chunk_text(cv_texts[i])[j])
                     for i, (score, j) in seen.items()],
                    key=lambda x: x[1], reverse=True)

    print("\n🎯 Top Matching CVs (Best-Matching Chunk Preview):\n")
    for rank, (filename, score, best_chunk) in enumerate(ranked[:args.top_k], 1):
        print(f"Rank {rank} | Score: {score:.4f} | File: {filename}")
        # print("-" * 60)
        # print(best_chunk.strip())
        # print("...\n")

if __name__ == "__main__":
    main()
