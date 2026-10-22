# Hallucination Detection using Explainable AI - Chrome Extension

```
extension/    Chrome extension (Manifest V3): reads the chat page, shows results in a side panel
backend/      Python FastAPI server: retrieval -> NLI verification -> XGBoost -> SHAP explanation
experiments/  Notebooks for later: real training data + evaluation
```

## 1. Run the backend (Windows PowerShell)

```powershell
cd backend
python -m venv venv
# If this next line is blocked, run first:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn app.main:app --port 8000
```

First start downloads ~250 MB of models (one time). Wait for the log line `Ready.`
Check it works: open http://localhost:8000/health  and  http://localhost:8000/docs

Run the tests (no downloads needed):  `python -m pytest app/tests -v`

## 2. Load the extension

1. Chrome -> `chrome://extensions` -> turn on **Developer mode**
2. **Load unpacked** -> select the `extension` folder
3. Open a ChatGPT / Claude / Gemini chat (reload the tab once), click the extension icon
4. Click **Analyze**, or select some text first, or paste text into the panel

## What is real and what is placeholder

- Real: claim splitting, hybrid retrieval (BM25 + FAISS + Wikipedia), NLI verification, 5 more
  signals, SHAP explanations, side-panel UI, page highlighting.
- PLACEHOLDER: the XGBoost classifier is bootstrapped on synthetic data so the pipeline runs end to
  end. Train it on real labelled data (HaluEval) before trusting any score - see classifier.py.
- Not wired in yet: semantic entropy (code + tests exist in pipeline/semantic_entropy.py).
