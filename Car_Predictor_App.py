import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model = joblib.load("car_prediction_model.joblib")

# Title
st.title("Vehicle Price Prediction (Cars and Bikes)")

# Combined list of cars and bikes without duplicates
vehicle_name = st.selectbox("Enter the vehicle name", [
    "Ritz", "SX4", "Ciaz", "Wagon R", "Swift", "Vitara Brezza", 
    "Alto 800", "Ertiga", "Alto K10", "Ignis", "Fortuner", 
    "S-Cross", "Omni", "Innova", "Corolla Altis", "Etios Cross", 
    "Etios G", "Etios Liva", "Etios GD", "Camry", "Land Cruiser", 
    "Corolla", "Elantra", "Creta", "Verna", "i20", "Grand i10", 
    "i10", "Eon", "Xcent", "City", "Brio", "Amaze", "Jazz",
    "Royal Enfield Thunder 500", "UM Renegade Mojave", "KTM RC200", 
    "Bajaj Dominar 400", "Royal Enfield Classic 350", "KTM RC390", 
    "Hyosung GT250R", "Royal Enfield Thunder 350", "KTM 390 Duke", 
    "Mahindra Mojo XT300", "Bajaj Pulsar RS200", "Royal Enfield Bullet 350", 
    "Royal Enfield Classic 500", "Bajaj Avenger 220", "Bajaj Avenger 150", 
    "Honda CB Hornet 160R", "Yamaha FZ S V 2.0", "Yamaha FZ 16", 
    "TVS Apache RTR 160", "Bajaj Pulsar 150", "Honda CBR 150", 
    "Hero Extreme", "Bajaj Avenger 220 DTSi", "Bajaj Avenger 150 Street", 
    "TVS Apache RTR 180", "Hero Passion X Pro"
])

year = st.number_input("Enter the year", min_value=2000, max_value=2024, step=1)
kms_driven = st.number_input("Enter the kilometers driven", min_value=0, step=100)
fuel_type = st.selectbox("Select fuel type", ["Petrol", "Diesel", "CNG", "LPG"])
transmission = st.selectbox("Select transmission type", ["Manual", "Automatic"])
owner = st.selectbox("Select owner type", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
seller_type = st.selectbox("Select seller type", ["Dealer", "Individual", "Trustmark Dealer"])

# Preprocess the data
input_data = np.array([[vehicle_name, year, kms_driven, fuel_type, transmission, owner, seller_type]])

# Ensure categorical data is encoded properly
le_vehicle_name = LabelEncoder()
le_fuel_type = LabelEncoder()
le_transmission = LabelEncoder()
le_owner = LabelEncoder()
le_seller_type = LabelEncoder()

# Example encoding process (fit the encoders on your training data)
input_data[:, 0] = le_vehicle_name.fit_transform(input_data[:, 0])  # Vehicle name
input_data[:, 3] = le_fuel_type.fit_transform(input_data[:, 3])  # Fuel type
input_data[:, 4] = le_transmission.fit_transform(input_data[:, 4])  # Transmission
input_data[:, 5] = le_owner.fit_transform(input_data[:, 5])  # Owner type
input_data[:, 6] = le_seller_type.fit_transform(input_data[:, 6])  # Seller type

# Convert the input data to the correct format
input_data = input_data.astype(float)

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted price is: {prediction[0]}")


