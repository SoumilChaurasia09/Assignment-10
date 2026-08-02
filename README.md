# End-to-End Machine Learning Model Deployment: Heart Disease Prediction

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.3-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.0-orange)
![Render Deployment](https://img.shields.io/badge/Render-Deployed-brightgreen)

This repository contains an end-to-end Machine Learning deployment solution for predicting patient heart disease risk based on clinical parameters. It includes complete data preprocessing, candidate classification model evaluation, trained model serialization, a Flask REST API service with interactive Web UI, and cloud deployment configuration for Render.

---

## 🌐 Live Render Deployment URL

- **GitHub Repository:** [https://github.com/SoumilChaurasia09/Assignment-10](https://github.com/SoumilChaurasia09/Assignment-10)
- **Render Live Service URL:** [https://heart-disease-prediction-4z7m.onrender.com](https://heart-disease-prediction-4z7m.onrender.com)
- **API Health Endpoint:** `GET https://heart-disease-prediction-4z7m.onrender.com/health`
- **API Prediction Endpoint:** `POST https://heart-disease-prediction-4z7m.onrender.com/predict`

---

## 📁 Repository Structure

```text
HeartDiseaseDeployment/
│
├── app.py              # Flask Web Application & REST API
├── model.pkl           # Serialized trained Random Forest model artifact
├── requirements.txt    # Python package dependencies for deployment
├── README.md           # Project documentation and submission report
├── train_model.py      # ML data loading, EDA, training, and evaluation script
├── heart.csv           # Heart Disease clinical dataset (14 columns)
├── Procfile            # Deployment process file for Render (Gunicorn)
├── templates/
│   └── index.html      # Responsive glassmorphism web UI form template
└── static/
    └── style.css       # Custom CSS design system & micro-animations
```

---

## 📋 Task Breakdown & Solution Implementation

### Task 1: Data Understanding and Preprocessing (2 Marks)
- **Dataset:** Loaded using Pandas (`heart.csv`).
- **Features & Target Variable:**
  - **Numerical Features (13):** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`
  - **Target Variable (1):** `target` (0 = No Heart Disease Detected, 1 = Heart Disease Detected)
- **Data Quality:** Checked for missing values (`0` missing values after standard mode imputation).
- **Train/Test Split:** 80% training set (242 samples) and 20% testing set (61 samples) with stratified split.

### Task 2: Model Development & Evaluation (2 Marks)
Evaluated multiple classification algorithms on the test set:
1. **Random Forest Classifier:** **88.52% Accuracy** *(Selected Best Model)*
2. **Logistic Regression:** 86.89% Accuracy
3. **Decision Tree Classifier:** 73.77% Accuracy
4. **Support Vector Machine (SVM):** 67.21% Accuracy

The trained **Random Forest Classifier** model was serialized to `model.pkl` using `joblib`.

### Task 3: API Development (2 Marks)
Developed a Flask REST API with `/predict` endpoint:
- **Input:** JSON payload containing 13 clinical parameters.
- **Output:** Standardized JSON response returning prediction status and confidence score.

#### Example Request:
```bash
curl -X POST https://heart-disease-prediction-4z7m.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67,
    "sex": 1,
    "cp": 3,
    "trestbps": 160,
    "chol": 286,
    "fbs": 0,
    "restecg": 2,
    "thalach": 108,
    "exang": 1,
    "oldpeak": 1.5,
    "slope": 1,
    "ca": 3,
    "thal": 2
  }'
```

#### Example Response:
```json
{
  "confidence": 0.87,
  "input_received": {
    "age": 67.0,
    "ca": 3.0,
    "chol": 286.0,
    "cp": 3.0,
    "exang": 1.0,
    "fbs": 0.0,
    "oldpeak": 1.5,
    "restecg": 2.0,
    "sex": 1.0,
    "slope": 1.0,
    "thal": 2.0,
    "thalach": 108.0,
    "trestbps": 160.0
  },
  "prediction": "Heart Disease Detected",
  "prediction_code": 1,
  "status": "success"
}
```

---

## 🚀 Step-by-Step GitHub & Render Deployment Guide (Task 4)

### 1. Push Code to GitHub
```bash
git init
git add .
git commit -m "Initial commit: Heart Disease ML REST API & Render Deployment"
git branch -M main
git remote add origin https://github.com/SoumilChaurasia09/Assignment-10.git
git push -u origin main
```

### 2. Deploy on Render
1. Sign in to [Render](https://render.com/).
2. Click **New +** -> **Web Service**.
3. Connect your public GitHub repository (`HeartDiseaseDeployment`).
4. Configure service settings:
   - **Name:** `heart-disease-prediction-api`
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
5. Click **Create Web Service**. Render will build and deploy the app live.

---

## 📝 Task 5: Conclusion & MLOps Insights (1 Mark)

The Random Forest classification model achieved an outstanding **88.52% accuracy** on the test dataset, proving highly effective at capturing non-linear clinical relationships between parameters like ST depression (`oldpeak`), chest pain type (`cp`), and heart disease risk. Key challenges encountered during cloud deployment included configuring environment-agnostic dependency versions in `requirements.txt`, handling Web Server Gateway Interface (WSGI) timeouts using Gunicorn, and managing cold-start latencies inherent to free-tier cloud platforms like Render. This project underscores the vital importance of MLOps in modern data science: building an accurate machine learning model is only half the battle. Establishing reproducible automated pipelines, robust REST API interfaces, continuous deployment integration, and health monitoring ensures machine learning solutions transition seamlessly from experimental Jupyter notebooks into reliable, accessible, and life-saving clinical healthcare applications.

---

## 💻 Local Running Instructions

1. **Clone & Setup Environment:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/HeartDiseaseDeployment.git
   cd HeartDiseaseDeployment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train Model:**
   ```bash
   python train_model.py
   ```

3. **Run Web App locally:**
   ```bash
   python app.py
   ```
   Open `http://127.0.0.1:5000` in your web browser.
