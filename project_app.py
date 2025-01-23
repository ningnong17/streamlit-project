import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv('resale.csv')

expected_columns = [
    'town_ANG MO KIO', 'town_BEDOK', 'town_BISHAN', 'town_BUKIT BATOK', 'town_BUKIT MERAH',
    'town_BUKIT PANJANG', 'town_BUKIT TIMAH', 'town_CENTRAL AREA', 'town_CHOA CHU KANG',
    'town_CLEMENTI', 'town_GEYLANG', 'town_HOUGANG', 'town_JURONG EAST', 'town_JURONG WEST',
    'town_KALLANG/WHAMPOA', 'town_MARINE PARADE', 'town_PASIR RIS', 'town_PUNGGOL',
    'town_QUEENSTOWN', 'town_SEMBAWANG', 'town_SENGKANG', 'town_SERANGOON', 'town_TAMPINES',
    'town_TOA PAYOH', 'town_WOODLANDS', 'town_YISHUN',
    'flat_type_1 ROOM', 'flat_type_2 ROOM', 'flat_type_3 ROOM', 'flat_type_4 ROOM',
    'flat_type_5 ROOM', 'flat_type_EXECUTIVE', 'flat_type_MULTI-GENERATION',
    'storey_range_01 TO 03', 'storey_range_04 TO 06', 'storey_range_07 TO 09',
    'storey_range_10 TO 12', 'storey_range_13 TO 15', 'storey_range_16 TO 18',
    'storey_range_19 TO 21', 'storey_range_22 TO 24', 'storey_range_25 TO 27',
    'storey_range_28 TO 30', 'storey_range_31 TO 33', 'storey_range_34 TO 36',
    'storey_range_37 TO 39', 'storey_range_40 TO 42', 'storey_range_43 TO 45',
    'storey_range_46 TO 48', 'storey_range_49 TO 51',
    "flat_model_2-room", "flat_model_3Gen", "flat_model_Adjoined flat", "flat_model_Apartment",
    "flat_model_DBSS", "flat_model_Improved", "flat_model_Improved-Maisonette",
    "flat_model_Maisonette", "flat_model_Model A", "flat_model_Model A-Maisonette",
    "flat_model_Model A2", "flat_model_Multi Generation", "flat_model_New Generation",
    "flat_model_Premium Apartment", "flat_model_Premium Apartment Loft",
    "flat_model_Premium Maisonette", "flat_model_Simplified", "flat_model_Standard",
    "flat_model_Terrace", "flat_model_Type S1", "flat_model_Type S2",
    'floor_area_sqm', 'lease_commence_date'
]

st.write("""
# Resale Price Prediction App
This app predicts the **resale price** of a flat based on various features!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    town = st.sidebar.selectbox('Town', df['town'].unique())
    flat_type = st.sidebar.selectbox('Flat Type', df['flat_type'].unique())
    storey_range = st.sidebar.selectbox('Storey Range', df['storey_range'].unique())
    floor_area_sqm = st.sidebar.slider('Floor Area (sqm)', 30, 370, 100)
    lease_commence_date = st.sidebar.slider('Lease Commence Date', 1930, 2020, 2000)

    data = {
        'town': [town],
        'flat_type': [flat_type],
        'storey_range': [storey_range],
        'floor_area_sqm': [floor_area_sqm],
        'lease_commence_date': [lease_commence_date],
    }
    features = pd.DataFrame(data)
    
    features = pd.get_dummies(features)

    features = features.reindex(columns=expected_columns, fill_value=0)
    
    return features, town, floor_area_sqm

user_input, user_town, user_floor_area = user_input_features()

model = joblib.load('trained_resale_price_decision_tree_model.pkl')

prediction = model.predict(user_input)


st.subheader("Predicted Resale Price")
st.write(f"The predicted resale price is: ${prediction[0]:,.2f}")

# Filter data to show 5 closest listings to user input
filtered_data = df[df['town'] == user_town].copy()
filtered_data['difference'] = abs(filtered_data['floor_area_sqm'] - user_floor_area)
closest_listings = filtered_data.nsmallest(5, 'difference')

# Display latest listings
st.subheader(f"5 Latest Listings in {user_town}")

# Filter the data for the selected town
latest_listings = df[df['town'] == user_town]
latest_listings = latest_listings.sort_values(by='month', ascending=False).head(5)

# Display the latest listings
st.write(latest_listings[['town', 'flat_type', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'resale_price']])

# Plot prediction distribution
st.subheader("Prediction Distribution")
plt.hist(df['resale_price'], bins=20, color='blue', edgecolor='black')
plt.axvline(prediction[0], color='red', linestyle='dashed', linewidth=2)
plt.title('Resale Price Distribution')
plt.xlabel('Resale Price')
plt.ylabel('Frequency')
st.pyplot(plt)
