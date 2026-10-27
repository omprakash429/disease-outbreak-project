# Disease Outbreak Transmission Predictor

College ML project (BE CSE). Uses the CDC **NORS** outbreak dataset to predict
**how an outbreak mainly spread** (`Primary Mode`: Person-to-person, Food, Water, Animal contact)
from the month, state, germ (etiology), setting and number of people ill. Includes a Streamlit web app.

> **Disclaimer:** educational project only. It is NOT a real-world public-health warning system.

## Why this task?
The dataset only contains outbreaks that already happened, so it cannot honestly predict
*whether* an outbreak will occur. Predicting the transmission mode is a valid task the data supports.

## Folder structure
```
outbreak_project/
├── app.py               # Streamlit web interface
├── train_model.py       # data report + training + evaluation
├── requirements.txt
├── README.md
├── data/NORS_20261027.csv      <- your dataset goes here
├── src/data_utils.py    # cleaning (shared by training and app)
├── src/predict.py       # prediction helper
├── models/              # created by train_model.py
└── reports/             # charts + metrics, created by train_model.py
```

## Setup on Windows (VS Code)
1. Install Python 3.10+ from https://www.python.org/downloads/ (tick **"Add python.exe to PATH"**).
2. In VS Code: **File > Open Folder** > choose `outbreak_project`, then **Terminal > New Terminal**.
3. Run these commands one by one:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

If activation is blocked, run `Set-ExecutionPolicy -Scope Process Bypass` first and activate again.
The app opens at http://localhost:8501. Next time you only need `.venv\Scripts\activate` then `streamlit run app.py`.

## How it avoids data leakage
- `Food Vehicle`, `Food Contaminated Ingredient`, `IFSAC Category`, `Water Exposure`, `Water Type`,
  `Animal Type` are only filled in after the mode is known -> **not used**.
- `Hospitalizations` / `Deaths` are later outcomes -> **not used**.
- Exact duplicates are removed **before** the train/test split; the split is stratified (80/20).

## Models
Baseline (always guesses the most common class), Logistic Regression, and Random Forest (main model).
Metrics: accuracy, macro-F1, per-class precision/recall, confusion matrix.
Run `python train_model.py` to see your own numbers (saved in `reports/`).

## Limitations
- US-only, non-random reporting; the model mostly learns reporting patterns (setting + germ).
- "Animal contact" has few examples (~650) and is predicted unreliably.
- "Indeterminate/unknown" and "Environmental contamination" modes were dropped.
