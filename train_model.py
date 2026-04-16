"""
train_model.py
--------------
Script to train an Artificial Neural Network (ANN) for predicting
bank customer churn (Exited column).

Steps:
  1. Load dataset
  2. Preprocess (encode, scale)
  3. Train ANN
  4. Evaluate
  5. Save model + preprocessing objects
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import os

# ============================================================
# 1. LOAD DATASET
# ============================================================
print("=" * 60)
print("Step 1: Loading dataset...")
print("=" * 60)

dataset = pd.read_csv('Artificial_Neural_Network_Case_Study_data.csv')
print(f"Dataset shape: {dataset.shape}")
print(f"\nFirst 5 rows:\n{dataset.head()}")
print(f"\nColumn names: {list(dataset.columns)}")
print(f"\nDataset info:")
print(dataset.info())
print(f"\nMissing values:\n{dataset.isnull().sum()}")
print(f"\nTarget distribution:\n{dataset['Exited'].value_counts()}")

# ============================================================
# 2. DATA PREPROCESSING
# ============================================================
print("\n" + "=" * 60)
print("Step 2: Data Preprocessing...")
print("=" * 60)

# Separate features (X) and target (y)
# Drop irrelevant columns: RowNumber, CustomerId, Surname
X = dataset.iloc[:, 3:13].values   # CreditScore to EstimatedSalary
y = dataset.iloc[:, 13].values     # Exited

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature columns: {list(dataset.columns[3:13])}")

# Encode categorical variables
# Geography (column index 1 in X) and Gender (column index 2 in X)

# Label encode Gender (Male/Female -> 0/1)
le_gender = LabelEncoder()
X[:, 2] = le_gender.fit_transform(X[:, 2])
print(f"\nGender encoding: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")

# One-Hot encode Geography (France/Germany/Spain)
# Using ColumnTransformer for Geography column (index 1)
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(drop='first'), [1])],  # drop='first' to avoid dummy variable trap
    remainder='passthrough'
)
X = np.array(ct.fit_transform(X), dtype=float)
print(f"Shape after encoding: {X.shape}")

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================
print("\n" + "=" * 60)
print("Step 3: Train-Test Split...")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ============================================================
# 4. FEATURE SCALING
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Feature Scaling...")
print("=" * 60)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print("StandardScaler fitted and applied.")
print(f"Mean of features (train): {sc.mean_[:5]}...")
print(f"Scale of features (train): {sc.scale_[:5]}...")

# ============================================================
# 5. BUILD ANN MODEL
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Building ANN Model...")
print("=" * 60)

# Import TensorFlow/Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Build the ANN
model = Sequential()

# Input layer + First hidden layer (ReLU activation)
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dropout(0.2))  # Dropout for regularization

# Second hidden layer (ReLU activation)
model.add(Dense(units=32, activation='relu'))
model.add(Dropout(0.2))

# Output layer (Sigmoid for binary classification)
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# ============================================================
# 6. TRAIN THE MODEL
# ============================================================
print("\n" + "=" * 60)
print("Step 6: Training the ANN...")
print("=" * 60)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=100,
    validation_split=0.1,
    verbose=1
)

# ============================================================
# 7. EVALUATE THE MODEL
# ============================================================
print("\n" + "=" * 60)
print("Step 7: Evaluating the Model...")
print("=" * 60)

# Predictions on test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:\n{cm}")

# Classification Report
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# ============================================================
# 8. SAVE MODEL AND PREPROCESSING OBJECTS
# ============================================================
print("\n" + "=" * 60)
print("Step 8: Saving Model and Preprocessing Objects...")
print("=" * 60)

# Save the trained ANN model
model.save('model.h5')
print("[OK] Model saved as 'model.h5'")

# Save preprocessing objects (LabelEncoder, ColumnTransformer, StandardScaler)
preprocessing = {
    'label_encoder_gender': le_gender,
    'column_transformer': ct,
    'scaler': sc,
    'feature_columns': list(dataset.columns[3:13])
}

with open('preprocessing.pkl', 'wb') as f:
    pickle.dump(preprocessing, f)
print("[OK] Preprocessing objects saved as 'preprocessing.pkl'")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
print("=" * 60)
