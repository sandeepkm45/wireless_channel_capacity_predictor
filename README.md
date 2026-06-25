# 📡 Wireless Channel Capacity Predictor

An end-to-end ML project that predicts wireless channel capacity (Mbps) from physical link parameters using five regression algorithms, topped by an interactive Streamlit dashboard.

## Live Demo
Deploy in one click via **[Streamlit Cloud](#-deploy-to-streamlit-cloud)** — no server required.

---

## Project Pipeline

```
Phase 1 — Setup         pip install + folder structure
Phase 2 — Dataset       10,000 synthetic samples (Shannon physics)
Phase 3 — Preprocess    StandardScaler · 80/20 train-test split
Phase 4 — Train         5 algorithms trained & timed
Phase 5 — Evaluate      R² · RMSE · MAE comparison
Phase 6 — Save          best_model.pkl · scaler.pkl
Phase 7 — Dashboard     Streamlit interactive app
```

### Model Results

| Model | R² | RMSE (Mbps) |
|---|---|---|
| **Neural Network (MLP)** ⭐ | **0.858** | — |
| Random Forest | 0.853 | — |
| SVM (SVR) | 0.761 | — |
| Decision Tree | 0.752 | — |
| Linear Regression | 0.377 | — |

---

## Project Structure

```
wireless-predictor/
├── app.py                  # Phase 7 — Streamlit dashboard
├── generate_dataset.py     # Phase 2 — synthetic dataset
├── preprocess.py           # Phase 3 — preprocessing pipeline
├── train_models.py         # Phase 4–6 — training & saving
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml         # amber dark theme
└── models/                 # pre-trained artifacts (committed)
    ├── best_model.pkl
    ├── best_model_name.pkl
    ├── feature_columns.pkl
    ├── scaler.pkl
    ├── target_scaler.pkl
    ├── model_comparison.csv
    └── model_comparison.png
```

---

## 🖥️ Run Locally

### 1 — Clone & create environment
```bash
git clone https://github.com/<your-username>/wireless-predictor.git
cd wireless-predictor

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### 3 — Launch the dashboard
```bash
streamlit run app.py
```
Opens at **http://localhost:8501** automatically.

### (Optional) Retrain from scratch
```bash
python generate_dataset.py   # regenerates data/wireless_dataset.csv
python train_models.py       # retrains all 5 models, saves best to models/
streamlit run app.py         # pick up new artifacts
```

---

## ☁️ Deploy to Streamlit Cloud

The fastest zero-cost path to a public URL:

1. **Fork / push** this repo to your GitHub account (keep it public, or use Streamlit Cloud's private-repo feature on a paid plan).
2. Go to **[share.streamlit.io](https://share.streamlit.io)** → **New app**.
3. Fill in:
   - **Repository** — `<your-username>/wireless-predictor`
   - **Branch** — `main`
   - **Main file path** — `app.py`
4. Click **Deploy** — Streamlit Cloud installs `requirements.txt` and starts the app.
5. Share the generated `https://<your-app>.streamlit.app` link.

> **Note:** The `models/` folder (≈ 260 KB total) is committed to the repo so Streamlit Cloud can load the pre-trained artifacts directly without retraining.

---

## Physics Behind the Model

```
Log-distance path loss  =  PL(d₀) + 10·n·log₁₀(d/d₀) + X_σ
Received power (dBm)    =  Pₜₓ + Gₐₙₜ − PL
SNR (dB)                =  Pᵣₓ − (−174 + 10·log₁₀(B) + NF)
Shannon capacity (Mbps) =  B · log₂(1 + SNR_linear) × fading_factor
```

Environment-dependent path-loss exponents:

| Environment | n | σ (dB) |
|---|---|---|
| Urban | 3.8 | 9.0 |
| Suburban | 3.2 | 7.0 |
| Rural | 2.3 | 4.5 |

---

## Dashboard Features

- **Real-time prediction** — sliders update all outputs instantly
- **Capacity vs. Distance** sweep chart (ML vs Shannon theory)
- **Link Budget** breakdown (Tx → FSPL → Rx → SNR → Capacity)
- **Model Comparison** bar chart (all 5 algorithms, R² scores)
- **Link Quality** indicator (Excellent / Good / Fair / Poor)

---

## Tech Stack

`Python 3.10+` · `Streamlit` · `scikit-learn` · `Plotly` · `NumPy` · `Pandas` · `Joblib`

---

## License

MIT — free to use, modify, and distribute.
