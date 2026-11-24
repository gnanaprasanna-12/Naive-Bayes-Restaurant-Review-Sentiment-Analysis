import streamlit as st
import joblib
import re
import string
from nltk.stem import PorterStemmer

# Load saved files
model = joblib.load('naive_bayes_model.pkl')
cv = joblib.load('count_vectorizer.pkl')
preprocessing = joblib.load('preprocessing.pkl')

stop_words = preprocessing['stop_words']
ps = PorterStemmer()

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [ps.stem(word) for word in words]

    return ' '.join(words)

# Streamlit application
st.title("Restaurant Review Sentiment Analysis")
st.write("Enter a restaurant review to predict its sentiment.")

review = st.text_area("Enter your review")

if st.button("Predict Sentiment"):
    if review.strip():
        cleaned_review = preprocess_text(review)
        review_vector = cv.transform([cleaned_review])

        prediction = model.predict(review_vector)[0]
        probability = model.predict_proba(review_vector)[0]

        if prediction == 1:
            st.success("Positive Review 😊")
        else:
            st.error("Negative Review 😞")

        st.write("Negative Probability:", round(probability[0] * 100, 2), "%")
        st.write("Positive Probability:", round(probability[1] * 100, 2), "%")
    else:
        st.warning("Please enter a review.")
        