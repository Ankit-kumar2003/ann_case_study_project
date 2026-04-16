# Bank Customer Churn Prediction — ANN Case Study

A complete end-to-end Machine Learning web application that predicts whether a bank customer will **leave (churn)** or **stay** using an **Artificial Neural Network (ANN)**.

---

## About the Project

Banks lose customers every year due to churn. This project uses Deep Learning (ANN) to predict customer churn based on their demographics, account information, and activity patterns. The model is trained on a dataset of **10,000 bank customers** and deployed as a web application using Flask.

### Key Features
- Trained ANN model with **86% accuracy**
- Clean, modern web interface for predictions
- Real-time churn probability display
- Flask-based backend with preprocessed ML pipeline

---

## Dataset

The dataset contains **10,000 records** with the following features:

| Feature | Description |
|---------|-------------|
| CreditScore | Customer's credit score (300–900) |
| Geography | Country (France, Germany, Spain) |
| Gender | Male / Female |
| Age | Customer's age |
| Tenure | Years with the bank (0–10) |
| Balance | Account balance |
| NumOfProducts | Number of bank products used |
| HasCrCard | Has credit card (Yes/No) |
| IsActiveMember | Is active member (Yes/No) |
| EstimatedSalary | Annual estimated salary |
| **Exited** | **Target — Left the bank (1) or Stayed (0)** |

---

## ANN Model Architecture

| Layer | Neurons | Activation | Dropout |
|-------|---------|------------|---------|
| Input + Hidden 1 | 64 | ReLU | 20% |
| Hidden 2 | 32 | ReLU | 20% |
| Output | 1 | Sigmoid | — |

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Epochs:** 100 | **Batch Size:** 32
- **Test Accuracy:** ~86%

---

## Project Structure

```
ann_case_study_project/
├── app.py                          # Flask backend
├── train_model.py                  # Model training script
├── model.h5                        # Trained ANN model
├── preprocessing.pkl               # Saved encoders & scaler
├── requirements.txt                # Python dependencies
├── Procfile                        # Deployment config
├── Artificial_Neural_Network_Case_Study_data.csv
├── templates/
│   └── index.html                  # Web UI
└── static/
    └── style.css                   # Styling
```

---

## How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Ankit-kumar2003/ann_case_study_project.git
cd ann_case_study_project

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train the model
python train_model.py

# 4. Start the Flask app
python app.py

# 5. Open browser
# Visit http://127.0.0.1:5000
```

---

## Technologies Used

- **Python** — Core programming language
- **TensorFlow / Keras** — ANN model building & training
- **Flask** — Web framework for backend
- **Scikit-learn** — Preprocessing (encoding, scaling)
- **Pandas / NumPy** — Data manipulation
- **HTML / CSS** — Frontend interface
- **Gunicorn** — Production WSGI server

---

## Results

- **Accuracy:** 86.10%
- **Confusion Matrix:**

|  | Predicted Stay | Predicted Leave |
|--|---------------|----------------|
| **Actual Stay** | 1548 | 59 |
| **Actual Leave** | 219 | 174 |

---

## Deployment

This project is deployment-ready for platforms like **Render**, **Railway**, or **PythonAnywhere**.

**Procfile** and **requirements.txt** are included for one-click deployment.

---

## Author

**Ankit Kumar**

---
