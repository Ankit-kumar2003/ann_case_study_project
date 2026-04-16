"""
app.py
------
Flask backend for the Bank Customer Churn Prediction Web Application.

Routes:
  - "/" : Homepage with prediction form
  - "/predict" : POST endpoint to get churn prediction
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ============================================================
# INITIALIZE FLASK APP
# ============================================================
app = Flask(__name__)

# ============================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ============================================================
print("Loading model and preprocessing objects...")

# Load the trained ANN model
model = load_model('model.h5')
print("[OK] Model loaded successfully!")

# Load preprocessing objects
with open('preprocessing.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

le_gender = preprocessing['label_encoder_gender']
ct = preprocessing['column_transformer']
sc = preprocessing['scaler']
feature_columns = preprocessing['feature_columns']

print("[OK] Preprocessing objects loaded successfully!")
print(f"Feature columns: {feature_columns}")

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
        geography = request.form['geography']           # France, Germany, or Spain
        gender = request.form['gender']                 # Male or Female
        age = int(request.form['age'])
        tenure = int(request.form['tenure'])
        balance = float(request.form['balance'])
        num_of_products = int(request.form['num_of_products'])
        has_cr_card = int(request.form['has_cr_card'])       # 0 or 1
        is_active_member = int(request.form['is_active_member'])  # 0 or 1
        estimated_salary = float(request.form['estimated_salary'])

        # Create input array in the SAME order as training features:
        # [CreditScore, Geography, Gender, Age, Tenure, Balance,
        #  NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]

        # Encode Gender using the saved LabelEncoder
        gender_encoded = le_gender.transform([gender])[0]

        # Build the raw input array (before OneHotEncoding)
        input_data = np.array([[
            credit_score,
            geography,         # Will be OneHot encoded by ColumnTransformer
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

        # Make prediction
        prediction_prob = model.predict(input_data)[0][0]
        prediction = int(prediction_prob > 0.5)

        # Determine result message
        if prediction == 1:
            result = "Will Leave the Bank 😟"
            result_class = "negative"
        else:
            result = "Will Stay with the Bank 😊"
            result_class = "positive"

        probability = round(prediction_prob * 100, 2)

        return render_template(
            'index.html',
            prediction=result,
            result_class=result_class,
            probability=probability,
            # Pass back form values to retain them after submission
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
        # Handle errors gracefully
        error_msg = f"Error: {str(e)}. Please check your inputs and try again."
        return render_template('index.html', error=error_msg)


# ============================================================
# RUN THE APP
# ============================================================
if __name__ == '__main__':
    app.run(debug=True, port=5000)
