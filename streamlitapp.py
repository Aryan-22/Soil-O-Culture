import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Crop Recommendation App", layout="wide")

# Load the scaler from the pickle file
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the transformation function


def transform_input(data):
    d = data[0]
    for i in range(len(d)):
        if idx_to_skew[i] == 0:
            d[i] = np.square(d[i])
        else:
            d[i] = np.sqrt(d[i])
    x = np.array([d])
    x = scaler.transform(x)
    return x


# Add a title and description
st.title("🌾 Crop Recommendation App")
st.write("""
This app predicts the **crop type** based on soil and weather conditions! 🌱
Upload a CSV file with your data or use the sliders and text inputs below to enter your data manually.
""")

# Sidebar for user input
st.sidebar.header('📝 User Input Features')
st.sidebar.markdown("""
Upload a CSV file with the features or use the sliders and text inputs below for manual input.

[Example CSV input file](https://example.com/example.csv)
""")

# Define the skew transformation
idx_to_skew = [1, 0, 1, 0, 0, 0, 1]

# File uploader for CSV input
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    st.sidebar.write("Uploaded CSV File:")
    st.sidebar.write(input_df)
else:
    st.sidebar.subheader('🔄 Manual Input (or use sliders)')

    st.sidebar.write("**Use the sliders to adjust values:**")

    nitrogen = st.sidebar.slider('Nitrogen', 0, 100, 50)
    phosphorus = st.sidebar.slider('Phosphorus', 0, 100, 50)
    potassium = st.sidebar.slider('Potassium', 0, 100, 50)
    temperature = st.sidebar.slider('Temperature (°C)', 0.0, 50.0, 25.0)
    humidity = st.sidebar.slider('Humidity (%)', 0.0, 100.0, 50.0)
    ph = st.sidebar.slider('pH', 0.0, 14.0, 7.0)
    rainfall = st.sidebar.slider('Rainfall (mm)', 0.0, 500.0, 100.0)

    # Define text inputs for manual input
    nitrogen_text = st.sidebar.number_input(
        'Nitrogen (0-100)', min_value=0, max_value=100, value=nitrogen)
    phosphorus_text = st.sidebar.number_input(
        'Phosphorus (0-100)', min_value=0, max_value=100, value=phosphorus)
    potassium_text = st.sidebar.number_input(
        'Potassium (0-100)', min_value=0, max_value=100, value=potassium)
    temperature_text = st.sidebar.number_input(
        'Temperature (°C) (0.0-50.0)', min_value=0.0, max_value=50.0, step=0.1, value=temperature)
    humidity_text = st.sidebar.number_input(
        'Humidity (%) (0.0-100.0)', min_value=0.0, max_value=100.0, step=0.1, value=humidity)
    ph_text = st.sidebar.number_input(
        'pH (0.0-14.0)', min_value=0.0, max_value=14.0, step=0.1, value=ph)
    rainfall_text = st.sidebar.number_input(
        'Rainfall (mm) (0.0-500.0)', min_value=0.0, max_value=500.0, step=0.1, value=rainfall)

    data_slider = {
        'NITROGEN': nitrogen,
        'PHOSPHORUS': phosphorus,
        'POTASSIUM': potassium,
        'TEMPERATURE': temperature,
        'HUMIDITY': humidity,
        'PH': ph,
        'RAINFALL': rainfall
    }
    input_df_slider = pd.DataFrame(data_slider, index=[0])

    data_text = {
        'NITROGEN': nitrogen_text,
        'PHOSPHORUS': phosphorus_text,
        'POTASSIUM': potassium_text,
        'TEMPERATURE': temperature_text,
        'HUMIDITY': humidity_text,
        'PH': ph_text,
        'RAINFALL': rainfall_text
    }
    input_df_text = pd.DataFrame(data_text, index=[0])

    use_sliders = st.sidebar.radio(
        "Choose input method:", ("Use Sliders", "Use Text Inputs"))

    if use_sliders == "Use Sliders":
        st.sidebar.write("**Slider Values:**")
        st.sidebar.write(input_df_slider)
        input_df = input_df_slider
    else:
        st.sidebar.write("**Text Input Values:**")
        st.sidebar.write(input_df_text)
        input_df = input_df_text

# Display user input features
st.subheader('🌟 User Input Features')
st.write(input_df)

# Transform the input data
input_data = transform_input(input_df.values)

# Load the trained model
load_clf = pickle.load(open('best_RF.pkl', 'rb'))

# Make predictions
prediction = load_clf.predict(input_data)
prediction_proba = load_clf.predict_proba(input_data)

# Display the prediction
st.subheader('🔍 Prediction')
crop_types = ['banana', 'blackgram', 'chickpea', 'coconut', 'coffee', 'cotton', 'jute',
              'kidneybeans', 'lentil', 'maize', 'mango', 'mothbeans', 'mungbean', 'muskmelon',
              'orange', 'papaya', 'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

predicted_crop = prediction[0]
st.write(f"**Predicted Crop Type:** :green[{predicted_crop}]")

# Display prediction probability
st.subheader('📊 Prediction Probability')
prob_df = pd.DataFrame(prediction_proba, columns=crop_types)
st.bar_chart(prob_df.T, height=300, use_container_width=True)

# Display the class with the highest probability
highest_prob_class_index = np.argmax(prediction_proba, axis=1)
highest_prob_class = crop_types[highest_prob_class_index[0]]
highest_prob_value = prediction_proba[0, highest_prob_class_index[0]]

st.subheader('🏆 Class with Highest Probability')
st.write(f"**Class:** :blue[{highest_prob_class}]")
st.write(f"**Probability:** :blue[{highest_prob_value:.4f}]")

st.markdown("---")
st.markdown("Thank you for using the Crop Recommendation App! 🌾")
