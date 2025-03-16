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

print("✅ All necessary NLTK resources are installed!")
# Load the saved model and vectorizer
# Load models and vectorizer
models = {
    "Logistic Regression": joblib.load("logistic_regression_model.joblib"),
    "Multinomial Naïve Bayes": joblib.load("naive_bayes_model.joblib"),
    "Support Vector Machine (SVM)": joblib.load("svm_model.joblib"),
    "Word2Vec + Logistic Regression": joblib.load("logistic_regression_model_word2vec.pkl")
}

vectorizer = joblib.load("vectorizer.joblib")
if not hasattr(vectorizer, 'idf_'):
    raise ValueError("The vectorizer is not fitted.")

# Load word2vec model (if needed)
# For example:
# word2vec_model = ...  # Load the Word2Vec model if it's not loaded already

# Sidebar information
st.sidebar.header("🔍 About the App")
st.sidebar.write(
    """
    This AI-powered application classifies news articles as either **Real** or **Fake**.
    
    **Models Available:**
    - Logistic Regression 
    - Multinomial Naïve Bayes 
    - Support Vector Machine (SVM)
    - Word2Vec + Logistic Regression

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

def sentence_to_vec(sentence, word2vec_model):
    words = sentence.split()
    word_vecs = [word2vec_model[word] for word in words if word in word2vec_model]
    if len(word_vecs) == 0:
        return np.zeros(word2vec_model.vector_size)
    return np.mean(word_vecs, axis=0)

if st.button("Predict"):
    if user_input.strip():
        if target_model == "Word2Vec + Logistic Regression":
            processed_text = text_preprocessing(user_input)
            text_vectorized = sentence_to_vec(processed_text, word2vec_model)
            text_vectorized = np.array(text_vectorized).reshape(1, -1)
            prediction = models[target_model].predict(text_vectorized)[0]
        else:
            processed_text = text_preprocessing(user_input)
            text_vectorized = vectorizer.transform([processed_text])
            prediction = models[target_model].predict(text_vectorized)[0]
        
        if prediction == 1:
            st.success("✅ This news appears to be **REAL**!")
        else:
            st.error("❌ This news might be **FAKE**!")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
