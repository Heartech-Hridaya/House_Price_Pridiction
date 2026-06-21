import streamlit as st
import pandas as pd
import pickle

# Load saved model and helpers
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('location_price_map.pkl', 'rb') as f:
    location_price_map = pickle.load(f)

with open('model_columns.pkl', 'rb') as f:
    model_columns = pickle.load(f)

st.title("🏠 House Price Predictor")
st.write("Estimate the total price of a house based on its features.")

# --- User inputs ---
location = st.selectbox("Location", sorted(location_price_map.index.tolist()))
carpet_area = st.number_input("Carpet Area (sqft)", min_value=100, max_value=10000, value=1000)
bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
balcony = st.number_input("Balconies", min_value=0, max_value=10, value=1)
floor_number = st.number_input("Floor Number", min_value=0, max_value=50, value=2)
total_floors = st.number_input("Total Floors in Building", min_value=1, max_value=50, value=10)
furnishing = st.selectbox("Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished", "Unknown"])
ownership = st.selectbox("Ownership", ["Freehold", "Leasehold", "Co-operative Society", "Power Of Attorney", "Unknown"])
transaction = st.selectbox("Transaction Type", ["New Property", "Resale", "Rent/Lease", "Other"])

if st.button("Predict Price"):
    # Build a single-row dataframe matching training format
    input_data = pd.DataFrame(columns=model_columns)
    input_data.loc[0] = 0  # start everything at 0

    input_data['Carpet_Area'] = carpet_area
    input_data['Bathroom'] = bathroom
    input_data['Balcony'] = balcony
    input_data['Floor_Number'] = floor_number
    input_data['Total_Floors'] = total_floors
    input_data['location_encoded'] = location_price_map.get(location, location_price_map.mean())

    # Set the matching one-hot columns to 1 (True), if they exist in model_columns
    furnishing_col = f'Furnishing_{furnishing}'
    if furnishing_col in input_data.columns:
        input_data[furnishing_col] = 1

    ownership_col = f'Ownership_{ownership}'
    if ownership_col in input_data.columns:
        input_data[ownership_col] = 1

    transaction_col = f'Transaction_{transaction}'
    if transaction_col in input_data.columns:
        input_data[transaction_col] = 1

    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Price: ₹{prediction:,.0f}")