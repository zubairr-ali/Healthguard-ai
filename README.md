# HealthGuard AI

**Explainable clinical risk prediction for heart disease and diabetes.**
Computer Science project, module 6WCM0029, University of Hertfordshire.

HealthGuard AI predicts a patient's risk of heart disease or diabetes from
clinical measurements, explains every prediction feature-by-feature using
SHAP, and — beyond the prediction tool itself — investigates a broader
question: **does a more complex model actually earn its cost, or does the
right architecture depend entirely on the shape of the data?**

That question is tested four times, across four different kinds of data:
structured clinical records, ECG waveforms, free-text symptom descriptions,
and an independent second patient cohort. The findings don't always agree
with each other, and that disagreement is the most interesting part of the
project — see [Research findings](#research-findings) below.

---

## What's in this repository

| Folder | What it is |
|---|---|
| `backend/` | FastAPI application: model training, SHAP explanations, LLM advisory notes, SQLite prediction history, and the full pytest suite |
| `frontend/` | React + Tailwind CSS web app — the live prediction interface |
| `notebooks/ecg/` | Colab notebook and results for the ECG arrhythmia classification phase |
| `notebooks/nlp/` | Colab notebook and results for the symptom-text classification phase |

Large trained model weights (TabPFN's cache, the ECG `.keras` files, the
DistilBERT checkpoint) are intentionally excluded from version control —
see `.gitignore` in each folder. Every result is still fully reproducible
by re-running the relevant script or notebook.

---

## Live prediction tool

The core deliverable: enter a patient's clinical values for heart disease
or diabetes and get back a risk probability, a SHAP breakdown of which
factors drove it, and a plain-language advisory note grounded in the
actual submitted values.

**Best models:** TabPFN for heart disease (95.84% ROC-AUC), XGBoost for
diabetes (82.33% ROC-AUC) — selected from a lineup of **11 models per
condition**, spanning classical baselines, gradient-boosting ensembles, a
deep neural network, a pretrained tabular foundation model (TabPFN), and
three tabular transformer architectures (FT-Transformer, TabTransformer,
TabNet).

A separate in-app page (`/models`) shows the full 11-model comparison for
both conditions, colour-coded by model family, with an explanation of
*why* the weakest model underperforms rather than just reporting the
winner.

---

## Research findings

Beyond the live tool, four experimental phases each ask the same question
in a different data setting: does the more sophisticated model win?

| Phase | Data | Winner | Why |
|---|---|---|---|
| **Structured records** | 918 rows, 11 tabular models | TabPFN / XGBoost | Small sample, no spatial/sequential structure — the pretrained foundation model and gradient boosting both edge out deep learning and transformers, though FT-Transformer and TabNet are competitive |
| **ECG signal** | 109,446 heartbeats, 3 architectures | 1D CNN (97.58% acc, 0.8776 macro F1) | Local waveform morphology; a BiLSTM was both slower to train and less accurate, confirmed via a convergence check, not just a single run |
| **Symptom text** | 1,153 descriptions, 4 models | Statistical tie | A linear SVM on TF-IDF features (0.9675 macro F1) matched a fine-tuned DistilBERT (~0.964) at roughly 1,600× faster training |
| **Cross-dataset validation** | Heart (918) ↔ independent cardiovascular cohort (70,000) | — | The key finding: a model's internal cross-validated ranking does **not** reliably predict its ranking on an independent population (Spearman ρ = 0.943 one direction, 0.257 — statistically no better than random — the other) |

**A methodology finding along the way:** a second cross-dataset experiment
was aborted after row-level fingerprinting revealed that a supposedly
independent "second hospital" diabetes dataset was actually the original
Pima dataset resampled with replacement (95.6% row overlap). The
validation pipeline now includes an automatic independence check that
refuses to score any dataset pair sharing more than 5% of rows.

Full write-ups, figures, and the underlying data live in `notebooks/` and
the results chapter of the project report.

---

## Tech stack

- **Backend:** Python, FastAPI, scikit-learn, XGBoost / LightGBM / CatBoost, TabPFN, PyTorch (FT-Transformer, TabTransformer, TabNet), SHAP, SQLite
- **Frontend:** React 19, Tailwind CSS v4, React Router, Recharts, Framer Motion
- **Modelling:** TensorFlow/Keras (ECG CNN/BiLSTM), Hugging Face Transformers (DistilBERT), trained on Google Colab (T4 GPU)
- **Testing:** pytest (17 tests — models, SHAP, database, API)

---

## Running it locally

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
uvicorn main:app --reload
```

Runs at `http://127.0.0.1:8000`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Runs at `http://localhost:5173` and proxies `/api/*` requests to the
backend above — both need to be running for live predictions.

### Tests

```bash
cd backend
pytest
```

---

## Datasets

| Dataset | Rows | Source |
|---|---|---|
| Heart disease | 918 | Combined Cleveland, Hungarian, Switzerland, Long Beach VA (fedesoriano, Kaggle) |
| Diabetes | 768 | Pima Indians Diabetes Database |
| Cardiovascular disease | 70,000 | sulianova, Kaggle (used for cross-dataset validation only) |
| MIT-BIH ECG Arrhythmia | 109,446 heartbeats | shayanfazeli/heartbeat, Kaggle |
| Symptom-to-disease text | 1,153 descriptions | niyarrbarman/symptom2disease, Kaggle |

All datasets are publicly available; none contain real patient-identifiable data.

---

## Disclaimer

HealthGuard AI produces statistical risk estimates for decision-support
purposes only. It is not a diagnostic tool, does not account for
information outside the fields it is given, and should never replace
professional medical advice.

---

## Acknowledgements

Built as part of a Computer Science final-year project, with guidance
from an academic supervisor across multiple review meetings.
