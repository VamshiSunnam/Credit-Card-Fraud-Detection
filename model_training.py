import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Used for saving the model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc

# --- 1. Configuration & Setup ---
pd.set_option('display.max_columns', None)
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')
DATASET_PATH = 'data/creditcard.csv'
MODEL_PATH = 'fraud_model.joblib'
SCALER_PATH = 'scaler.joblib'

# --- 2. Data Loading ---
def load_data(path):
    """Loads the dataset from the given path."""
    print(f"Loading dataset from: {path}...")
    try:
        data = pd.read_csv(path)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file was not found at {path}")
        print("Please download the dataset and place it in the 'data' directory.")
        return None

# --- 3. Data Preprocessing ---
def preprocess_data(df):
    """Scales the 'Amount' and 'Time' features and returns X, y, and the scaler."""
    if df is None:
        return None, None

    print("\n--- Preprocessing Data ---")
    processed_df = df.copy()
    
    scaler = StandardScaler()
    
    # CORRECTED: Fit the scaler on 'Amount' and 'Time' together and transform them
    # The order here ['Amount', 'Time'] is important and must be consistent.
    processed_df[['scaled_amount', 'scaled_time']] = scaler.fit_transform(processed_df[['Amount', 'Time']])
    
    # Drop the original 'Time' and 'Amount' columns
    processed_df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    print("'Amount' and 'Time' columns scaled.")
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")

    # Reorder columns to ensure 'Class' is at the end before splitting, for consistency
    if 'Class' in processed_df.columns:
        cols = [col for col in processed_df.columns if col != 'Class']
        processed_df = processed_df[cols + ['Class']]

    X = processed_df.drop('Class', axis=1)
    y = processed_df['Class']

    return X, y

# --- 4. Model Training and Evaluation ---
def train_and_evaluate(X, y):
    """Splits data, trains a Random Forest model, evaluates it, and saves it."""
    if X is None or y is None:
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("\n--- Data Splitting ---")
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # We will focus on Random Forest as it generally performs well on this task
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    print("\n--- Training Random Forest ---")
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # --- Evaluation ---
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"\nPrecision-Recall AUC: {pr_auc:.4f}")

# --- 5. Main Execution ---
if __name__ == "__main__":
    print("====== Credit Card Fraud Model Training ======")
    credit_card_data = load_data(DATASET_PATH)

    if credit_card_data is not None:
        X_features, y_target = preprocess_data(credit_card_data)
        train_and_evaluate(X_features, y_target)

    print("\n====== Model Training Finished ======")
