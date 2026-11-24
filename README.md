# Naive-Bayes-Restaurant-Review-Sentiment-Analysis
Restaurant Review Sentiment Classification using Naive Bayes and Streamlit.
# Restaurant Review Sentiment Analysis using Naïve Bayes

## Project Overview
This project uses Natural Language Processing (NLP) and Naïve Bayes machine learning algorithms to classify restaurant reviews as Positive or Negative.

A Streamlit web application is developed to allow users to enter a restaurant review and predict its sentiment.

## Technologies Used
- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Streamlit
- Joblib

## Project Workflow
1. Data Loading and Exploratory Data Analysis
2. Text Cleaning and Preprocessing
3. Stopword Removal (preserving negation words)
4. Stemming using Porter Stemmer
5. Train-Test Split (80:20)
6. Text Vectorization
7. Naïve Bayes Model Building
8. Model Evaluation
9. Sentiment Prediction
10. Streamlit Application

## Machine Learning Models
- Multinomial Naïve Bayes with CountVectorizer
- Multinomial Naïve Bayes with TF-IDF
- Bernoulli Naïve Bayes with Binary Features

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|
| MultinomialNB + CountVectorizer | 81.5% | 81.19% | 82% | 81.59% |
| MultinomialNB + TF-IDF | 79.5% | 78.10% | 82% | 80% |
| BernoulliNB + Binary Features | 81% | 78.70% | 85% | 81.73% |

## Streamlit Application
The application accepts a restaurant review and displays:
- Positive or Negative sentiment
- Negative probability
- Positive probability

## How to Run the Project

1. Clone this repository:
   `git clone YOUR_GITHUB_REPOSITORY_URL`

2. Install the required libraries:
   `pip install -r requirements.txt`

3. Run the Streamlit application:
   `streamlit run app.py`

## Project Files
- `app.py` – Streamlit application
- `naive_bayes_model.pkl` – Trained Naïve Bayes model
- `count_vectorizer.pkl` – Saved CountVectorizer
- `preprocessing.pkl` – Saved preprocessing settings
- `naive_bayes_model.ipynb` – Model implementation notebook
- `requirements.txt` – Required Python libraries

## Conclusion
This project demonstrates how Naïve Bayes algorithms and NLP techniques can be used to classify restaurant reviews. The trained model is integrated with Streamlit to provide an interactive sentiment prediction application.
