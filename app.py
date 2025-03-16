
import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# Load additional resources if needed
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')

# Load models and vectorizer
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
    "Multinomial Naïve Bayes": joblib.load("naive_bayes_model.joblib"),
    "Support Vector Machine (SVM)": joblib.load("svm_model.joblib")
}

vectorizer = joblib.load("vectorizer.joblib")
if not hasattr(vectorizer, 'idf_'):
    raise ValueError("The vectorizer is not fitted.")

st.set_page_config(
    page_title="Fake News Detection App",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar information
st.sidebar.header("🔍 About the App")
st.sidebar.write(
    """
    This AI-powered application classifies news articles as either **Real** or **Fake**.
    
    **Models Available:**
    - Logistic Regression 
    - Multinomial Naïve Bayes 
    - Support Vector Machine (SVM)

    **How to Use:**
    1. Select a model from the dropdown.
    2. Paste a news article in the text box.
    3. Click **Predict** to see the result.
    """
)

# Streamlit UI
st.title("Fake News Detection App 📰")
st.subheader("Enter a news article to check if it's real or fake")

# Model selection
target_model = st.selectbox("Choose a model:", list(models.keys()))

# User input
user_input = st.text_area("Enter news text here:")

def text_preprocessing(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Tokenization & lowercase
    stop_words = set(stopwords.words('english'))  
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    expanded_text = contractions.fix(" ".join(tokens))  # Expand contractions
    return " ".join(word_tokenize(expanded_text))  # Tokenize again & return

if st.button("Predict"):
    if user_input.strip():
        processed_text = text_preprocessing(user_input)
        text_vectorized = vectorizer.transform([processed_text])
        prediction = models[target_model].predict(text_vectorized)[0]
        
        if prediction == 1:
            st.success("✅ This news appears to be **REAL**!")
        else:
            st.error("❌ This news might be **FAKE**!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
