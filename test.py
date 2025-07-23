import os
import re
import fitz  # PyMuPDF
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

def build_index(model, all_chunks):
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    embeddings = normalize(embeddings, axis=1)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings

def main():
    parser = argparse.ArgumentParser(description="CV Similarity Search with Dual-Model Score Fusion")
    parser.add_argument("--cv_folder", type=str, default="cvs", help="Folder with CV PDFs")
    parser.add_argument("--jd_file", type=str, required=True, help="Path to job description .txt file")
    parser.add_argument("--top_k", type=int, default=8, help="Number of top matches to return")
    parser.add_argument("--alpha", type=float, default=0.5, help="Weight for model A in score fusion (0-1)")
    args = parser.parse_args()

    print("🔍 Loading embedding models...")
    model_a = SentenceTransformer("intfloat/multilingual-e5-large")  # Model A
    model_b = SentenceTransformer("BAAI/bge-m3")                     # Model B

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
        all_chunks.extend(chunks)
        chunk_map.extend([(i, j) for j in range(len(chunks))])

    print("⚙️ Building indexes for both models...")
    encoded_chunks_a = [f"passage: {chunk}" for chunk in all_chunks]
    encoded_chunks_b = [chunk for chunk in all_chunks]

    index_a, emb_a = build_index(model_a, encoded_chunks_a)
    index_b, emb_b = build_index(model_b, encoded_chunks_b)

    with open(args.jd_file, 'r', encoding='utf-8') as f:
        job_description = clean_text(f.read())
    print("\n📄 Job description loaded.")

    print("🔍 Encoding job description...")
    query_a = model_a.encode([f"query: {job_description}"], convert_to_numpy=True)
    query_b = model_b.encode([job_description], convert_to_numpy=True)
    query_a = normalize(query_a, axis=1)
    query_b = normalize(query_b, axis=1)

    print("🚀 Performing similarity search on both models...")
    scores_a, idx_a = index_a.search(query_a, len(all_chunks))
    scores_b, idx_b = index_b.search(query_b, len(all_chunks))

    fused_scores = {}
    for rank in range(len(all_chunks)):
        idx = idx_a[0][rank]
        cv_idx, chunk_idx = chunk_map[idx]
        score_a = float(np.dot(query_a[0], emb_a[idx]))
        score_b = float(np.dot(query_b[0], emb_b[idx]))
        fused = args.alpha * score_a + (1 - args.alpha) * score_b
        if cv_idx not in fused_scores or fused > fused_scores[cv_idx][0]:
            fused_scores[cv_idx] = (fused, chunk_idx)

    ranked = sorted(
        [(cv_filenames[i], score, chunk_text(cv_texts[i])[j])
         for i, (score, j) in fused_scores.items()],
        key=lambda x: x[1], reverse=True
    )

    print(f"\n🎯 Top {args.top_k} Matching CVs (Score Fusion α={args.alpha}):\n")
    for rank, (filename, score, best_chunk) in enumerate(ranked[:args.top_k], 1):
        print(f"Rank {rank} | Score: {score:.4f} | File: {filename}")
        # print("-" * 60)
        # print(best_chunk.strip())
        # print("...\n")

if __name__ == "__main__":
    main()
