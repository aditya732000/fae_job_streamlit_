import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import joblib
import streamlit as st
from PIL import Image
import base64

# Initializing preprocessing tools
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Text cleaning function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    tokens = text.split()
    tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    tokens = [lemmatizer.lemmatize(token) for token in tokens]  # Lemmatization
    return " ".join(tokens)

# Function to apply transformations to user input
def preprocess_user_input(title, description, requirements, company_profile, benefits, employment_type,
                          required_experience, required_education, industry, function):
    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
        'employment_type': [employment_type],
        'required_experience': [required_experience],
        'required_education': [required_education],
        'industry': [industry],
        'function': [function],
        'company': [title + ' ' + description + ' ' + requirements + ' ' + company_profile + ' ' + benefits]
    })

    # Apply text preprocessing
    input_data['company'] = input_data['company'].apply(preprocess_text)

    # Apply TF-IDF on the 'company' column (using pre-trained TF-IDF model)
    tfidf = joblib.load('tfidf_transformer.pkl')  # Load the previously trained transformer
    tfidf_matrix = tfidf.transform(input_data['company'])
    input_data_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

    # Encode categorical columns (using pre-trained target_encoder)
    target_encoder = joblib.load('target_encoder.pkl')  # Load the encoder
    encoded_data = target_encoder.transform(input_data[['required_education', 'industry', 'function']])

    # Combine the encoded data with other columns
    input_data_encoded = pd.concat(
        [input_data.drop(['company', 'required_education', 'industry', 'function'], axis=1), encoded_data], axis=1)

    # Add the TF-IDF transformed data
    input_data_encoded = pd.concat([input_data_encoded, input_data_tfidf], axis=1)

    return input_data_encoded

# Load the LightGBM model
model = joblib.load('lightgbm_best_model.pkl')

# Prediction function
def predict_fraud(input_data_encoded):
    prediction = model.predict(input_data_encoded)
    return "⚠ The job post is likely fraudulent!" if prediction == 1 else "✅ The job post is legitimate."

# Streamlit application
def main():
    # Custom CSS for creative and professional design
    st.markdown("""
        <style>
            body {
                background-image: url('imago.jpg'); /* Background image as 'imago.jpg' */
                background-size: cover; /* Ensure the image covers the entire screen */
                background-position: center center; /* Center the image */
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                color: #333;
            }
            .header {
                 /* Gradient overlay for better contrast */
                color: white;
                padding: 40px 0;
                text-align: center;
                margin-bottom: 40px;
            }
            .header h1 {
                font-size: 50px;
                text-transform: uppercase;
                color:black;
                font-weight: bold;
                letter-spacing: 2px;
                
            }
            .navbar {
                display: flex;
                justify-content: center;
                background-color: #a0c4ff;
                padding: 15px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            }
            .navbar a {
                color: white;
                padding: 14px 20px;
                text-decoration: none;
                font-size: 18px;
                font-weight: bold;
                transition: background 0.3s ease;
                margin: 0 15px;
                position: relative;
            }
            .navbar a:hover {
                background-color: #ff7e5f;
                border-radius: 5px;
            }
            .navbar a:hover .hover-text {
                display: block; /* Show additional info on hover */
            }
            .hover-text {
                display: none;
                position: absolute;
                top: 100%;
                left: 0;
                padding: 10px;
                background-color: rgba(0, 0, 0, 0.8);
                color: white;
                border-radius: 5px;
                font-size: 14px;
                max-width: 200px;
                text-align: center;
            }
            .content {
                padding: 40px 20px;
            }
            .content .form-section {
                background-color: rgba(255, 255, 255, 0.8); /* Semi-transparent background for readability */
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                margin-bottom: 30px;
            }
            .content h2 {
                font-size: 36px;
                margin-bottom: 20px;
                color: #333;
                text-align: center;
            }
            .content p {
                font-size: 16px;
                line-height: 1.5;
                text-align: center;
                color: #555;
            }
            .stButton>button {
                background-color: #ff7e5f;
                color: white;
                font-size: 18px;
                padding: 15px 40px;
                border-radius: 50px;
                border: none;
                cursor: pointer;
                transition: all 0.3s ease;
                margin: 0 auto;
                display: block;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            }
            .stButton>button:hover {
                background-color: #feb47b;
                transform: scale(1.1);
            }
            .footer {
                background-color: #333;
                color: white;
                padding: 20px;
                text-align: center;
                margin-top: 40px;
            }
            .footer a {
                color: white;
                text-decoration: none;
                margin: 0 10px;
            }
            .footer a:hover {
                text-decoration: underline;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header with title and description
    st.markdown("""
        <div class="header">
            <h1>Job Offer Fraud Detection</h1
        </div>
    """, unsafe_allow_html=True)

    # Navigation bar with hover effect on "About"
    st.markdown("""
        <div class="navbar">
            <a href="#home">Home</a>
            <a href="#predict">Predict Fraud</a>
            <a href="#about" id="about-link">About</a>
        </div>
    """, unsafe_allow_html=True)



    # Main content section
    st.markdown("""<div class="content" id="predict">""", unsafe_allow_html=True)

    # Form fields using Streamlit's native widgets
    title = st.text_input('Job Title')
    description = st.text_area('Job Description')
    requirements = st.text_area('Job Requirements')
    company_profile = st.text_area('Company Profile')
    benefits = st.text_area('Job Benefits')

    employment_type = st.selectbox('Employment Type', ['Full-time', 'Part-time', 'Contract', 'Temporary'])
    required_experience = st.selectbox('Required Experience', ['None', 'Entry-level', 'Mid-level', 'Senior'])
    required_education = st.selectbox('Required Education', ['Not Specified', 'High School', 'Bachelor', 'Master', 'PhD'])
    industry = st.text_input('Industry')
    function = st.text_input('Job Function')

    # Button to trigger prediction
    if st.button("🔍 Predict Fraud"):
        input_data_encoded = preprocess_user_input(title, description, requirements, company_profile, benefits,
                                                   employment_type, required_experience, required_education, industry,
                                                   function)

        # Check model features and align the columns of input data
        model_features = model.feature_name_
        input_data_features = input_data_encoded.columns

        # Add missing columns
        missing_features = set(model_features) - set(input_data_features)
        for feature in missing_features:
            input_data_encoded[feature] = 0  # Add missing columns with default values

        # Remove extra columns
        extra_features = set(input_data_features) - set(model_features)
        input_data_encoded = input_data_encoded.drop(columns=extra_features)

        # Reorganize columns to match exactly with the model's features
        input_data_encoded = input_data_encoded[model_features]

        # Predict fraud
        result = predict_fraud(input_data_encoded)

        # Display the result
        st.write(result)

        # Display an appropriate image based on prediction result
        if result == "✅ The job post is legitimate.":
            st.image("techno.png", width=50)  # Use the "techno" image for legitimate result
        else:
            st.image("alerte.png", width=50)  # Use the "alerte" image for fraudulent result

    # Footer section
    st.markdown("""
        <div class="footer">
            <p>&copy; 2024 Job Offer Fraud Detection. All rights reserved.</p>
            <p><a href="#privacy">Privacy Policy</a> | <a href="#terms">Terms of Service</a></p>
        </div>
    """, unsafe_allow_html=True)

def add_bg_from_local(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local('imago.png')


if __name__ == "__main__":
    main()
