import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- Load the saved model and columns ---
try:
    with open('house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('model_columns.pkl', 'rb') as file:
        model_columns = pickle.load(file)
except FileNotFoundError:
    st.error("Model files not found. Please run your Jupyter notebook to train and save the model first.")
    st.stop()

# --- Create the web app interface ---
st.title("🏡 House Price Prediction App")
st.write("Enter the details of the house to get a price prediction.")

# Create input fields for the user based on your dataset's features
# IMPORTANT: This is a simplified example. You should add all important features.
area = st.slider("Area (in sq. ft.)", min_value=500, max_value=5000, value=1500)
bedrooms = st.selectbox("No. of Bedrooms", options=[1, 2, 3, 4, 5], index=2)
bathrooms = st.selectbox("No. of Bathrooms", options=[1, 2, 3, 4], index=1)
city = st.selectbox("City", options=['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Hyderabad', 'Kolkata'])

# --- Preprocess user input and make a prediction ---
def predict_price():
    # Create a dictionary from user inputs
    input_data = {
        'Area': area,
        'No. of Bedrooms': bedrooms,
        'Bathrooms': bathrooms,
        'city': city
        # Add other necessary features here, possibly with default values of 0 or their median
    }

    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    # One-hot encode the categorical 'city' column
    input_df = pd.get_dummies(input_df)
    
    # Align the input DataFrame with the columns the model was trained on
    # This ensures the model receives data in the exact same format
    final_df = input_df.reindex(columns=model_columns, fill_value=0)

    # --- Make a prediction ---
    prediction_log = model.predict(final_df)
    
    # Revert the log transformation to get the actual price
    prediction = np.expm1(prediction_log)

    # --- Display the result ---
    st.success(f"Predicted House Price: ₹ {prediction[0]:,.2f}")

# Add a button to trigger the prediction
if st.button("Predict Price"):
    predict_price()
    