import streamlit as st
import joblib
import re
import nltk
import gensim.downloader as api
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions


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

# Streamlit Page Setup
st.set_page_config(
    page_title="News Classifier ",
    page_icon="📰",
    layout="centered",
    initial_sidebar_state="expanded",
)

# custom CSS for styling
st.markdown(
    """
    <style>
        /* Title and header styling */
        .title {
            font-size: 36px;
            font-weight: bold;
            text-align: center;
            color: #007BFF;
        }
        .subheader {
            font-size: 18px;
            text-align: center;
            color: #555;
        }
        
        /* Custom button styling */
        div.stButton > button {
            width: 100%;
            background-color: #007BFF;
            color: white;
            font-size: 18px;
            padding: 10px;
            border-radius: 10px;
            border: none;
        }

        /* Style for results */
        .real-news {
            color: green;
            font-size: 22px;
            font-weight: bold;
        }
        .fake-news {
            color: red;
            font-size: 22px;
            font-weight: bold;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load Word2Vec Model
#@st.cache_resource
def load_word2vec():
    return api.load("word2vec-google-news-300")
word2vec_model = load_word2vec()

# Load trained models
logistic_model = joblib.load("logistic_regression_model_word2vec.pkl")
conv1d_model = tf.keras.models.load_model("conv1d_model.h5")
svm_model = joblib.load("svm_model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

# Load Tokenizer
tokenizer = joblib.load("tokenizer.pkl")  

MAX_LEN = 7402  # Based on training setting

# Convert input text into a Word2Vec vector
def text_to_vector(text, word2vec_model):
    words = text.split()
    word_vecs = [word2vec_model[word] for word in words if word in word2vec_model]
    
    if len(word_vecs) == 0:
        return np.zeros(word2vec_model.vector_size)

    return np.mean(word_vecs, axis=0)

# Preprocess text for Conv1D model
def preprocess_text_conv1d(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post')
    return padded_sequence

# Preprocess text for SVM model
def text_preprocessing(text):
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    tokens = word_tokenize(text.lower())  # Tokenization & lowercase
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatization
    expanded_text = contractions.fix(" ".join(tokens))  # Expand contractions
    return " ".join(word_tokenize(expanded_text))  # Tokenize again & return

# Function to display confidence and plot the prediction
def display_confidence(prediction, model_choice):
    confidence = prediction
    if model_choice == "Conv1D Neural Network":
        confidence = prediction * 100

    # Show confidence score
    st.write(f"Confidence: **{confidence:.2f}%**")

    # Prepare values for plotting
    if prediction >= 0.5:
        real_confidence = confidence
        fake_confidence = 100 - confidence
    else:
        real_confidence = 100 - confidence
        fake_confidence = confidence

    # Plotting the confidence as a bar chart
    fig, ax = plt.subplots()
    ax.barh(["Real News", "Fake News"], [real_confidence, fake_confidence], color=["green", "red"])
    ax.set_xlabel("Confidence (%)")
    ax.set_title(f"Model Confidence for {model_choice}")
    st.pyplot(fig)

# Title
st.markdown("<p class='title'> News Classification App📰</p>", unsafe_allow_html=True)
st.markdown("<p class='subheader'>Detect Fake vs. Real News</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("🔍 About the App")
st.sidebar.write(
    """
    This AI-powered application classifies news articles as either **Real** or **Fake**.
    
    **Models Available:**
    - Logistic Regression (Word2Vec)
    - Deep Learning (Conv1D Neural Network)
    - Support Vector Machine (SVM)

    **How to Use:**
    1. Select a model from the dropdown.
    2. Paste a news article in the text box.
    3. Click **Predict** to see the result.
    """
)

# Model selection dropdown
model_choice = st.selectbox("Choose a model:", ["Logistic Regression", "Conv1D Neural Network", "SVM"])

# User input
user_input = st.text_area("Enter news text:", height=250)

# Predict button
if st.button("🔍 Predict"):
    if user_input:
        # Process input based on selected model
        if model_choice == "Logistic Regression":
            input_vector = text_to_vector(user_input, word2vec_model).reshape(1, -1)
            prediction = logistic_model.predict(input_vector)[0]
        elif model_choice == "Conv1D Neural Network":
            input_vector = preprocess_text_conv1d(user_input)
            prediction = conv1d_model.predict(input_vector)[0][0]  # Get probability
        else:  # SVM Model
            processed_text = text_preprocessing(user_input)
            text_vectorized = vectorizer.transform([processed_text])
            prediction = svm_model.predict(text_vectorized)[0]

        if prediction >= 0.5:
            st.markdown("<p class='real-news'>✅ Prediction: Real News</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='fake-news'>❌ Prediction: Fake News</p>", unsafe_allow_html=True)

        display_confidence(prediction, model_choice)

    else:
        st.warning("⚠️ Please enter some text for classification.")
