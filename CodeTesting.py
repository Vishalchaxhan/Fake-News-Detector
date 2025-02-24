import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
df_fake = pd.read_csv("Fake.csv")
df_real = pd.read_csv("True.csv")

# Add labels
df_fake["label"] = 0  # Fake news
df_real["label"] = 1  # Real news

# Combine datasets and shuffle
df = pd.concat([df_fake, df_real]).sample(frac=1).reset_index(drop=True)

# Text Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Apply cleaning
df['text'] = df['text'].apply(clean_text)

# Vectorization (TF-IDF)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = tfidf_vectorizer.fit_transform(df['text'])

# Model Training (Naive Bayes)
model = MultinomialNB()
model.fit(X_tfidf, df['label'])

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("vectorizer.pkl", "wb"))

# Streamlit App
st.title("📰 Fake News Detector")

# User Input
user_input = st.text_area("Enter news article text:")

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Prediction Function
def predict_news(news):
    cleaned_news = clean_text(news)
    vectorized_news = vectorizer.transform([cleaned_news])
    prediction = model.predict(vectorized_news)
    return "✅ Real News" if prediction[0] == 1 else "🚨 Fake News"

# Prediction Button
if st.button("Predict"):
    result = predict_news(user_input)
    st.subheader(f"Prediction: {result}")
