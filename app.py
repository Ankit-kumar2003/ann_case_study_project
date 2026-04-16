"""
app.py
------
Flask backend for the Bank Customer Churn Prediction Web Application.
Uses pure NumPy for model inference (no TensorFlow dependency).

Routes:
  - "/" : Homepage with prediction form
  - "/predict" : POST endpoint to get churn prediction
"""

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# ============================================================
# INITIALIZE FLASK APP
# ============================================================
app = Flask(__name__)

# ============================================================
# PURE NUMPY ANN INFERENCE
# ============================================================

def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def predict_ann(input_data, weights):
    """
    Forward pass through the ANN using pure NumPy.
    Replicates: Input -> Dense(64,ReLU) -> Dense(32,ReLU) -> Dense(1,Sigmoid)
    Note: Dropout is not applied during inference.
    """
    x = input_data
    for i, layer in enumerate(weights):
        w = np.array(layer['weights'])
        b = np.array(layer['bias'])
        x = np.dot(x, w) + b
        # Apply activation: ReLU for hidden layers, Sigmoid for output
        if i < len(weights) - 1:
            x = relu(x)
        else:
            x = sigmoid(x)
    return x

# ============================================================
# LOAD MODEL WEIGHTS AND SETUP PREPROCESSING
# ============================================================
print("Loading model weights and setting up preprocessing...")

# Load ANN weights from JSON (extracted from trained Keras model)
with open('model_weights.json', 'r') as f:
    model_weights = json.load(f)
print(f"[OK] Model weights loaded: {len(model_weights)} layers")

# Rebuild preprocessing objects from the dataset
dataset = pd.read_csv('Artificial_Neural_Network_Case_Study_data.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Label encode Gender
le_gender = LabelEncoder()
X[:, 2] = le_gender.fit_transform(X[:, 2])

# OneHot encode Geography
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1])],
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)

# Train-test split (same random_state as training)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit StandardScaler on training data
sc = StandardScaler()
sc.fit(X_train)

print("[OK] Preprocessing ready!")

# ============================================================
# ROUTES
# ============================================================

@app.route('/')
def home():
    """Render the homepage with the prediction form."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction request.
    Gets form data, preprocesses it (same as training),
    and returns the churn prediction.
    """
    try:
        # Get form data
        credit_score = int(request.form['credit_score'])
        geography = request.form['geography']
        gender = request.form['gender']
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])
        is_active_member = int(request.form['is_active_member'])
        estimated_salary = float(request.form['estimated_salary'])

        # Encode Gender using the saved LabelEncoder
        gender_encoded = le_gender.transform([gender])[0]

        # Build the raw input array (before OneHotEncoding)
        input_data = np.array([[
            credit_score,
            geography,
            gender_encoded,
            age,
            tenure,
            balance,
            num_of_products,
            has_cr_card,
            is_active_member,
            estimated_salary
        ]], dtype=object)

        # Apply the same ColumnTransformer (OneHotEncode Geography)
        input_data = np.array(ct.transform(input_data), dtype=float)

        # Apply the same StandardScaler
        input_data = sc.transform(input_data)

        # Make prediction using pure NumPy ANN
        prediction_prob = float(predict_ann(input_data, model_weights)[0][0])
        prediction = int(prediction_prob > 0.5)

        # Determine result message
        if prediction == 1:
            result = "Will Leave the Bank"
            result_class = "negative"
        else:
            result = "Will Stay with the Bank"
            result_class = "positive"

        probability = round(prediction_prob * 100, 2)

        return render_template(
            'index.html',
            prediction=result,
            result_class=result_class,
            probability=probability,
            credit_score=credit_score,
            geography=geography,
            gender=gender,
            age=age,
            tenure=tenure,
            balance=balance,
            num_of_products=num_of_products,
            has_cr_card=has_cr_card,
            is_active_member=is_active_member,
            estimated_salary=estimated_salary
        )

    except Exception as e:
        error_msg = f"Error: {str(e)}. Please check your inputs and try again."
        return render_template('index.html', error=error_msg)


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
