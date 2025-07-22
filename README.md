# 🌍 ResumeMatcher\_multilingual

A multilingual, domain-agnostic semantic search engine that matches candidate CVs (PDFs) to any job description using state-of-the-art transformer embeddings and FAISS vector search.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-success)
![FAISS](https://img.shields.io/badge/Search-FAISS-critical)
![Multilingual](https://img.shields.io/badge/Languages-100%2B-lightgrey)

---

## 🔍 What It Does

This tool ranks and retrieves the most relevant CVs for a given job description—regardless of role, language, or domain.

* ✅ Supports **100+ languages**
* ✅ Uses **chunked CV search** for higher semantic resolution
* ✅ Offers both **Streamlit UI** and **terminal CLI**
* ✅ Works across unrelated domains (e.g., tech vs hospitality)
* ✅ Pluggable support for **E5** and **MiniLM** models

---

## 📦 Project Structure

```bash
ResumeMatcher_multilingual/
├── main.py              # CLI interface
├── app.py               # Streamlit web app
├── jds/                 # Job descriptions (.txt)
├── cvs/                 # CVs (.pdf)
├── requirements.txt     # Dependencies
└── README.md
```

---

## ⚙️ Supported Embedding Models

You can switch between the following models in `main.py` or `app.py`:

|  Model  | Description  |
| ------- | ------------ | 
|[`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) | Fast, multilingual, trained for retrieval 
|[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base) | Best balance of speed and quality 
|[`intfloat/multilingual-e5-large`](https://huggingface.co/intfloat/multilingual-e5-large) | Highest accuracy (larger and slower)
|[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | General-purpose multilingual embeddings
|[`anass1209/resume-job-matcher-all-MiniLM-L6-v2`](https://huggingface.co/anass1209/resume-job-matcher-all-MiniLM-L6-v2)| Resume-specific multilingual MiniLM variant


---

## 🚀 Usage

### ▶️ Command-line (CLI)

```bash
python main.py --jd_file ./jds/software_engineer.txt --cv_folder ./cvs 
```

### 💻 Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser at: [http://localhost:8501](http://localhost:8501)

---

## 🔧 Installation
```bash
pip install -r requirements.txt
```

