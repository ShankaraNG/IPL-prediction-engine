# 🏏 IPL Match Winner Prediction — MLOps Pipeline

A machine learning application that predicts the winner of an Indian Premier League (IPL) match using a Random Forest model — developed by Shankara Narayana N G.

🌐 **Live Demo:** [https://ipl-prediction-engine.onrender.com](https://ipl-prediction-engine.onrender.com)

> ⏳ The application is hosted on Render's free tier. It may take approximately **1 minute to wake up** when you first hit the URL. Please be patient!

---

> ⚠️ **Important Disclaimer**
> This application is built **strictly for research and testing purposes only.**
> Any use of this application for actual gambling or betting is entirely at the user's own risk.
> The creator — **Shankara Narayana N G** — holds **no responsibility** for any such usage.

---

## 📖 Overview

The IPL Match Winner Prediction tool uses a **Random Forest classifier** trained on IPL match data from **2008 through the latest available leagues** to predict which team will win a match given the match conditions.

The project is split into two parts:

- 🤖 **ML Build** — Trains and saves the machine learning model locally
- 🚀 **FastAPI Application** — Serves the model via a web interface and REST API for predictions

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| ML Framework | scikit-learn (Random Forest) |
| API Framework | FastAPI + Uvicorn |
| Data Processing | Pandas, NumPy, SciPy |
| Visualisation | Matplotlib, Seaborn |
| Templating | Jinja2 |
| Config | PyYAML |
| Dependency Management | Poetry |

---

## ⚙️ Prerequisites

- Python 3.11 or above
- [Poetry](https://python-poetry.org/docs/#installation) installed on your machine

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd ipl-mlops
```

### 2. Install dependencies using Poetry

```bash
poetry install
```

---

## 🤖 Step 1 — Build the ML Model

The model must be built locally before running the application. This trains the Random Forest classifier and saves the model to the `/model` directory.

```bash
poetry run python ml_build.main
```

> ⚠️ This step is required before starting the application. The application will not work without the trained model saved in `/model`.

---

## ▶️ Step 2 — Run the Application

Once the model is built, start the FastAPI web application:

```bash
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The application will be available at `http://localhost:8000`.

---

## 🔗 API & UI Endpoints

### REST API

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Send match details as a JSON payload via Postman or any REST client to get a prediction |

### Web UI Pages

| Endpoint | Description |
|---|---|
| `/about` | Information about the IPL Prediction application |
| `/upload` | Upload a CSV file containing match data for batch prediction |
| `/form_predict` | Fill in the match details through a form and get a prediction |
| `/upload_predict` | Run predictions on a previously uploaded CSV file |

---

## 🏏 About the Model

- **Algorithm:** Random Forest Classifier
- **Training Data:** IPL match data from **2008 to the latest available season**
- **Prediction:** Given two competing IPL teams and match conditions, the model predicts which team is likely to win
- **Purpose:** Research and testing only

---

## 👤 Author

**Shankara Narayana N G**
📧 shankarnarayana92@gmail.com

For further details on setup, configuration, or usage — please contact Shankara directly.

---

## 📄 License

> This application is for **research and testing purposes only.** The creator holds no responsibility for any use of this application for gambling or betting of any kind.
