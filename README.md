# HireLens-A-Resume-Screening-App

This project can help recruiters and hiring managers **filter out resumes (CVs)** by automatically matching them against a given **job description**. Instead of going through hundreds of resumes manually, the system highlights the **top 3 most relevant candidates** based on textual similarity.  

---

## Features  
-  Upload multiple resumes (PDF, DOCX, TXT supported).  
-  Paste a job description into the application.  
-  Hybrid scoring: sentence embeddings + BM25 + skill overlap  
-  Returns the **top 3 resumes** with similarity scores (in %).  
-  Clean drag-and-drop UI
-  Built with **Flask (Python)** for the backend.  

---

## Tech Stack  
- **Backend**: Flask (Python)  
- **NLP/ML**: sentence-transformers, rank-bm25, numpy  
- **Frontend**: HTML, CSS, JavaScript 
- **File Handling**: PyPDF2, docx2txt  

---

## Project Structure  

```bash
Resume-Matcher/
├─ main.py
├─ templates/
│  └─ index.html
├─ static/
│  ├─ style.css
│  └─ script.js
├─ uploads/           # created at runtime
└─ requirements.txt   # optional, see below
```
# requirements.txt

```bash
flask
docx2txt
PyPDF2
numpy
rank-bm25
sentence-transformers
```

## Setup
1. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```
2. Install dependencies
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```
Open http://127.0.0.1:5000

* Paste the job description
* Drag & drop resumes
* Submit to see the ranked list (percent scores)

## How it works

### Preprocessing
* Lowercasing, remove URLs/emails
* Keep tech tokens like c++, .net, c#

### Scoring(Hybrid)

* **Embeddings**: cosine similarity between JD and the best matching chunk of each resume (chunking prevents long CVs from diluting signal)
* **BM25**: classic ranking for exact/frequent term matches
* **Skill overlap**: intersection of JD skills and resume skills (using a small skills lexicon + aliases)

### Final Score
```bash
final = α * embedding_score + β * skill_overlap + γ * bm25_score
```
Default weights in main.py:
* ALPHA = 0.6
* BETA = 0.3
* GAMMA = 0.1
### Top-K
* Returns up to 3 resumes

## Configuration (in main.py)

* MODEL_NAME: embedding model (sentence-transformers/all-MiniLM-L6-v2)
* ALPHA, BETA, GAMMA: hybrid weights
* CHUNK_WORDS: resume chunk size (default 250)
* SKILLS and ALIASES: extend for your domain (add tools, frameworks, etc.)
* ALLOWED_EXTENSIONS: allowed file types (pdf, docx, txt)

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

## License
MIT License
