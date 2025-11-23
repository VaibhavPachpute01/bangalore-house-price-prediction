import streamlit as st
import pickle
import json
import numpy as np

# ------------------------------
# LOAD MODEL & COLUMNS
# ------------------------------

# Load trained model
with open("bengaluru_home_prices_model.pickle", "rb") as f:
    model = pickle.load(f)

# Load columns.json
with open("columns.json", "r") as f:
    data_columns = json.load(f)["data_columns"]

# Extract location names (all columns after the first 4)
locations = data_columns[4:]

# ------------------------------
# STREAMLIT UI
# ------------------------------

st.set_page_config(page_title="Bangalore Home Price Predictor", layout="centered")

st.title("🏡 Bangalore House Price Prediction App")
st.write("Enter the property details below to estimate the home price.")

# ------------------------------
# USER INPUTS
# ------------------------------

location = st.selectbox("📍 Select Location", locations)

sqft = st.number_input("📏 Total Square Feet", min_value=100, max_value=10000, value=1000)

bath = st.number_input("🚿 Number of Bathrooms", min_value=1, max_value=10, value=2)

balcony = st.number_input("🌤 Number of Balconies", min_value=0, max_value=5, value=1)

bhk = st.number_input("🛏️ BHK", min_value=1, max_value=10, value=2)

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------

def predict_price(location, sqft, bath, balcony, bhk):
    # Create input vector
    x = np.zeros(len(data_columns))

    x[0] = sqft
    x[1] = bath
    x[2] = balcony
    x[3] = bhk

    # Set location index to 1
    if location.lower() in data_columns:
        loc_index = data_columns.index(location.lower())
    else:
        loc_index = -1

    if loc_index >= 0:
        x[loc_index] = 1

    return round(model.predict([x])[0], 2)

# ------------------------------
# BUTTON
# ------------------------------

if st.button("🔮 Predict Price"):
    result = predict_price(location, sqft, bath, balcony, bhk)
    st.success(f"🏷️ **Estimated Price: ₹ {result} Lakhs**")
