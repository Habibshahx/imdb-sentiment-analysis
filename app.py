import streamlit as st
import pickle

# Import SAME cleaning function
from utils import clean_text

# Load Pipeline
pipeline = pickle.load(open("model/pipeline.pkl", "rb"))

# Streamlit UI Config
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 IMDB Sentiment Analysis")
st.markdown("Analyze whether a movie review is **Positive** or **Negative** using Machine Learning.")

# Input
user_input = st.text_area("Enter your movie review:", height=150)

# Prediction
if st.button("🔍 Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")

    elif len(user_input.split()) < 2:
        st.warning("⚠️ Please enter a slightly longer review.")

    else:
        cleaned = clean_text(user_input)

        # Predict
        prediction = pipeline.predict([cleaned])[0]
        probability = pipeline.predict_proba([cleaned])[0]

        confidence = max(probability) * 100

        # Output Result
        if prediction == 1:
            st.success("Sentiment: Positive ")
        else:
            st.error("Sentiment: Negative ")

        st.info(f"Confidence: {confidence:.2f}%")

        st.write(f"Positive Probability: {probability[1]*100:.2f}%")
        st.write(f"Negative Probability: {probability[0]*100:.2f}%")

        st.progress(int(confidence))

# Info Section
st.markdown("---")

st.subheader("About This Model")
st.write("""
- Model: Logistic Regression  
- Feature Extraction: TF-IDF (with bigrams)  
- Dataset: IMDB 50K Reviews  
- Pipeline ensures consistent preprocessing  
""")

st.caption("Built using Streamlit")