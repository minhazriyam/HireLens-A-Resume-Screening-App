import os
import re
from typing import List
from werkzeug.utils import secure_filename

from flask import Flask, request, render_template
import docx2txt
import PyPDF2
import numpy as np

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# -------------------- Config --------------------
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

# weights for hybrid scoring
ALPHA = 0.6   # embeddings
BETA  = 0.3   # skills
GAMMA = 0.1   # BM25

CHUNK_WORDS = 250
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

SKILLS = {
    "python","java","c","c++","c#",".net","javascript","typescript","react","node","flask",
    "django","fastapi","pandas","numpy","scikit-learn","sklearn","pytorch","tensorflow",
    "keras","nlp","machine learning","deep learning","computer vision",
    "sql","postgres","mysql","mongodb","docker","kubernetes","aws","gcp","azure","linux",
    "git","rest","graphql","spark","airflow","bash","shell"
}

ALIASES = {
    "js": "javascript",
    "ts": "typescript",
    "tf": "tensorflow",
    "cv": "computer vision",
    "ml": "machine learning",
    "dl": "deep learning",
    "k8s": "kubernetes",
    "py": "python"
}

# -------------------- App --------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

EMB_MODEL = SentenceTransformer(MODEL_NAME)

# -------------------- File → text --------------------
def extract_text_from_pdf(path: str) -> str:
    text = ""
    try:
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for p in reader.pages:
                t = p.extract_text()
                if t: text += "\n" + t
    except Exception:
        pass
    return text.strip()

def extract_text_from_docx(path: str) -> str:
    try:
        return (docx2txt.process(path) or "").strip()
    except Exception:
        return ""

def extract_text_from_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
    except Exception:
        return ""

def extract_text(path: str) -> str:
    p = path.lower()
    if p.endswith(".pdf"):  return extract_text_from_pdf(path)
    if p.endswith(".docx"): return extract_text_from_docx(path)
    if p.endswith(".txt"):  return extract_text_from_txt(path)
    return ""

# -------------------- NLP helpers --------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"(https?://\S+)|(\w+@\w+\.\w+)", " ", text)
    text = re.sub(r"[^a-z0-9\s\-\+\.#/]", " ", text)  # keep tech chars like c++/.net
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str) -> List[str]:
    return preprocess(text).split()

def chunk_words(text: str, n: int = CHUNK_WORDS) -> List[str]:
    toks = preprocess(text).split()
    return [" ".join(toks[i:i+n]) for i in range(0, len(toks), n)] or [""]

def embed(texts: List[str]) -> np.ndarray:
    return EMB_MODEL.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

def max_chunk_sim(resume_text: str, jd_emb: np.ndarray) -> float:
    chunks = chunk_words(resume_text, CHUNK_WORDS)
    vecs = embed(chunks)
    # cosine since normalized
    return float(np.max(vecs @ jd_emb)) if len(vecs) else 0.0

def extract_skills(text: str) -> set:
    tl = preprocess(text)
    words = set(tl.split())
    expanded = {v for k, v in ALIASES.items() if k in tl}
    multi = {s for s in SKILLS if " " in s and s in tl}
    single = {s for s in SKILLS if " " not in s and s in words}
    return multi | single | expanded

def normalize_minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0: return arr
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-12: return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

# -------------------- Scoring --------------------
def hybrid_scores(job_description: str, resumes_text: List[str]) -> np.ndarray:
    jd_text = job_description or ""
    res_texts = [t or "" for t in resumes_text]

    # skills
    jd_sk = extract_skills(jd_text)
    sk_scores = np.array([
        (len(jd_sk & extract_skills(r)) / max(1, len(jd_sk))) if jd_sk else 0.0
        for r in res_texts
    ], dtype=float)

    # bm25
    bm25 = BM25Okapi([tokenize(r) for r in res_texts])
    bm25_raw = np.array(bm25.get_scores(tokenize(jd_text)), dtype=float)
    bm25_norm = normalize_minmax(bm25_raw)

    # embeddings (chunked max)
    jd_emb = embed([preprocess(jd_text)])[0]
    emb_raw = np.array([max_chunk_sim(r, jd_emb) for r in res_texts], dtype=float)
    emb_norm = (emb_raw + 1.0) / 2.0  # [-1,1] → [0,1]

    # hybrid
    final = ALPHA * emb_norm + BETA * sk_scores + GAMMA * bm25_norm
    return final

# -------------------- Flask routes --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/matcher", methods=["POST"])
def matcher():
    job_description = request.form.get("job_description", "").strip()
    resume_files = request.files.getlist("resumes")

    if not job_description or not resume_files:
        return render_template("index.html", message="Please upload resumes and enter a job description.")

    saved_names: List[str] = []
    texts: List[str] = []
    for f in resume_files:
        if not f or not f.filename or not allowed_file(f.filename):
            continue
        name = secure_filename(f.filename)
        path = os.path.join(app.config["UPLOAD_FOLDER"], name)
        f.save(path)
        saved_names.append(name)
        texts.append(extract_text(path))

    if not texts:
        return render_template("index.html", message="No valid resume files uploaded (pdf/docx/txt only).")

    scores = hybrid_scores(job_description, texts)

    k = min(3, len(scores))
    top_idx = np.argsort(scores)[-k:][::-1]

    top_resumes = [saved_names[i] for i in top_idx]
    # convert 0..1 → percent with 2 decimals
    similarity_scores = [float(round(scores[i] * 100, 2)) for i in top_idx]

    return render_template(
        "index.html",
        message="Top matching resumes:",
        top_resumes=top_resumes,
        similarity_scores=similarity_scores,
    )

if __name__ == "__main__":
    app.run(debug=True)
