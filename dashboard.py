import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Configuration ---
MODEL_PATH = 'fraud_model.joblib'
SCALER_PATH = 'scaler.joblib'

# --- Load Model and Scaler ---
@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler from disk."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError:
        st.error(f"Error: Model or scaler file not found. Please run 'model_training.py' first.")
        return None, None

model, scaler = load_model_and_scaler()

# --- Page Setup ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Credit Card Fraud Detection Dashboard")
st.write(
    "This interactive dashboard uses a trained Random Forest model to predict "
    "whether a credit card transaction is fraudulent or legitimate. "
    "Enter the transaction details in the sidebar to get a prediction."
)

# --- Sidebar for User Input ---
st.sidebar.header("Enter Transaction Data")

def get_user_input():
    """Creates sidebar elements to get transaction data from the user."""
    # The original dataset has 'Time' and 'Amount' followed by V1-V28
    # We will ask for Time and Amount first.
    time = st.sidebar.number_input("Time (seconds since first transaction)", min_value=0.0, value=1000.0, step=1.0)
    amount = st.sidebar.number_input("Transaction Amount ($)", min_value=0.0, value=50.0, step=1.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Anonymized Features (V1-V28)")
    
    # Create sliders for all V features
    v_features = {}
    for i in range(1, 29):
        feature_name = f'V{i}'
        v_features[feature_name] = st.sidebar.slider(
            label=feature_name,
            min_value=-50.0,
            max_value=50.0,
            value=0.0,
            step=0.1
        )
    
    # Store all inputs in a dictionary
    input_data = {'Time': time, 'Amount': amount, **v_features}
    
    # Convert to a DataFrame
    input_df = pd.DataFrame([input_data])
    return input_df

user_input_df = get_user_input()

# --- Main Panel for Displaying Results ---
st.header("Prediction Result")
st.write("The model will analyze the data you entered on the left.")

# Prediction button
if st.sidebar.button("Predict"):
    if model is not None and scaler is not None:
        st.subheader("Input Transaction Data")
        st.write(user_input_df)

        # --- Preprocessing the Input Data ---
        # Create a copy to avoid modifying the original input df
        processed_df = user_input_df.copy()

        # Scale 'Time' and 'Amount' using the loaded scaler
        # The scaler expects a 2D array for each feature
        scaled_features = scaler.transform(processed_df[['Time', 'Amount']])
        processed_df['scaled_time'] = scaled_features[:, 0]
        processed_df['scaled_amount'] = scaled_features[:, 1]
        
        # Drop original Time and Amount
        processed_df.drop(['Time', 'Amount'], axis=1, inplace=True)

        # Reorder columns to match the model's training data
        # The training data had scaled_amount and scaled_time at the end
        # V1 -> V28 -> scaled_amount -> scaled_time
        v_cols = [f'V{i}' for i in range(1, 29)]
        final_cols_order = v_cols + ['scaled_amount', 'scaled_time']
        processed_df = processed_df[final_cols_order]

        # --- Making Prediction ---
        prediction = model.predict(processed_df)
        prediction_proba = model.predict_proba(processed_df)

        # --- Displaying the Prediction ---
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.error("🚨 This transaction is likely **FRAUDULENT**.")
        else:
            st.success("✅ This transaction is likely **LEGITIMATE**.")

        st.subheader("Prediction Probability")
        st.write(f"**Probability of being Legitimate (Class 0):** {prediction_proba[0][0]:.2%}")
        st.write(f"**Probability of being Fraudulent (Class 1):** {prediction_proba[0][1]:.2%}")
    else:
        st.warning("Model is not loaded. Please check the logs.")

else:
    st.info("Enter transaction data in the sidebar and click 'Predict'.")

