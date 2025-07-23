import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the trained Random Forest model
rfc = load('movie_success_model.joblib')

# Create a Streamlit app
st.title("Movie Success Predictor")
st.markdown("Predict whether a movie will be successful based on key production features.")


# User Inputs
budget = st.number_input("Budget (in million USD)", min_value=0.0, format="%.2f")
duration = st.number_input("Enter Movie Duration (minutes)", min_value=30, max_value=240)
num_voted_users = st.number_input("Number Of User Votes", min_value=0)
cast_total_facebook_likes = st.number_input("Total Cast Facebook Likes", min_value=0)
movie_facebook_likes = st.number_input("Movie Facebook Likes", min_value=0)
country = st.selectbox("Country", ["USA", "France", "UK", "India", "Others"])
language = st.selectbox("Language", ["English", "Hindi", "Korean", "French", "Others"])
genre = st.selectbox("Genre", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi"])


# Encode Genre (simple label encoding)
country_dict = {"USA": 0, "France": 1, "UK": 2, "India": 3, "Others": 4}
language_dict = {"English": 0, "Hindi": 1, "Korean": 2, "French": 3,  "Others": 4}
genre_dict = {"Action": 0, "Comedy": 1, "Drama": 2, "Horror": 3, "Sci-Fi": 4}


country_encoded = country_dict[country]
language_encoded = language_dict[language]
genre_encoded = genre_dict[genre]


input_data = np.array([[budget, duration, num_voted_users, cast_total_facebook_likes, movie_facebook_likes, country_encoded, language_encoded, genre_encoded]])

# Make a prediction using the model
prediction = rfc.predict(input_data)

# Display the prediction result on the main screen
st.header("Prediction Result")

if prediction[0] == 0:
    st.success("Success! The movie is likely to be successful.")
else:
    st.error("Not Successful. The movie may not perform well.")

# Add any additional Streamlit components or UI elements as needed.
