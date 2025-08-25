
# Binary Classification: End-to-End ML Project

A clean, internship-ready machine learning project for **binary classification** using scikit-learn.  
It includes data loading, preprocessing, training with hyperparameter tuning, evaluation, a **Streamlit** demo app, unit tests, and **GitHub Actions** CI.

---

## 🧠 Problem
Given tabular features, predict one of two classes (e.g., disease vs. no disease).  
By default, this repo uses the built-in **Breast Cancer** dataset from scikit-learn.

---

## 📁 Project Structure
```
binary-classification-ml/
├── app/
│   └── app.py                # Streamlit demo app (loads trained model)
├── artifacts/
│   ├── model.joblib          # Saved best model (created after training)
│   ├── scaler.joblib         # Saved StandardScaler (created after training)
│   ├── metrics.json          # Evaluation metrics (created after evaluate)
│   └── plots/
│       ├── roc_curve.png
│       └── confusion_matrix.png
├── notebooks/
│   └── 01_eda_and_baseline.ipynb  # Optional notebook starter
├── src/
│   ├── __init__.py
│   ├── data.py               # load & preprocess data
│   ├── models.py             # model definitions + param grids
│   ├── train.py              # train + tune + save best model
│   └── evaluate.py           # evaluate, save metrics & plots
├── tests/
│   ├── test_data.py
│   └── test_train.py
├── .github/workflows/ci.yml  # CI: flake8 + pytest
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

## 🚀 Quickstart (Local)

### 1) Create and activate a virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Train and save the best model
```bash
python -m src.train --dataset breast_cancer --test-size 0.2 --seed 42
```
This creates:
- `artifacts/model.joblib`
- `artifacts/scaler.joblib`

### 4) Evaluate and create plots/metrics
```bash
python -m src.evaluate
```
This creates:
- `artifacts/metrics.json`
- `artifacts/plots/roc_curve.png`
- `artifacts/plots/confusion_matrix.png`

### 5) Run the Streamlit app
```bash
streamlit run app/app.py
```

---

## 🌐 Push to GitHub (Step-by-step)

> You can also use the web UI to create a new repo and upload files, but here’s the CLI flow.

1. **Sign in to GitHub** → Create a new empty repo (no README/license) named, e.g., `binary-classification-ml`.
2. In your terminal, inside this project folder:
```bash
git init
git add .
git commit -m "Initial commit: binary classification ML project"
# Replace <YOUR_GITHUB_USERNAME> and ensure the repo exists
git branch -M main
git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/binary-classification-ml.git
git push -u origin main
```
3. GitHub Actions CI will run automatically (flake8 + pytest).

---

## 🧪 Run tests locally
```bash
pytest -q
```

---

## 🧰 Switch dataset (optional)
By default, we use scikit-learn’s **Breast Cancer** dataset.  
To use your own CSV, adapt `src/data.py` in `load_dataset()`—there’s a TODO section showing how.

---

## 📄 License
MIT — do anything, just keep the copyright notice.
