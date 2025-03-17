# Fake News Classifier 📰

## Project Overview

This project builds a machine learning-based classifier to predict whether a given news article is **real** or **fake**. It leverages several popular machine learning models to achieve this classification, including traditional models like **Logistic Regression** and **Naïve Bayes**, as well as advanced models like **Support Vector Machine (SVM)** and **Word2Vec-based models** combined with deep learning architectures like **Conv1D**.

## Models Used

The project evaluates the performance of the following models:

- **Multinomial Naïve Bayes**
  - **Accuracy**: 95%
  
- **Logistic Regression**
  - **Accuracy**: 98.8%

- **Support Vector Machine (SVM)**
  - **Accuracy**: 99.5%

- **Word2Vec + Conv1D**
  - **Accuracy**: 99.8%
  - **Loss**: 0.014

- **Word2Vec + Logistic Regression**
  - **Accuracy**: 95.11%

These models are compared to determine the best-performing approach for fake news detection.

## Data Description

The dataset used in this project is stored in the `data.csv` file and contains the following columns:

- **label**: 
  - `0` for Fake news
  - `1` for Real news
  
- **title**: The headline of the news article.
- **text**: The full content of the article.
- **subject**: The category or topic of the article (e.g., Politics, Technology, etc.).
- **date**: The publication date of the article.

The goal is to build a classifier that predicts whether a news article is real or fake based on the text data in this dataset.

### Example of a Data Entry:

| label | title                 | text                                                         | subject   | date       |
|-------|-----------------------|--------------------------------------------------------------|-----------|------------|
| 1     | "Real News Headline"   | "This is a real article with detailed news about current events." | Politics  | 2025-03-16 |

## Project Files

Here are the key files in the project:

- **`app.py`**: Main application script to run the fake news classifier as a web app using **Streamlit**.
- **`requirements.txt`**: List of Python dependencies required for the project.
- **`Final_main.ipynb`**: containing the code of training models.
- **`data.csv`**: Dataset containing labeled news articles.
- **`validation_predictions.csv`**: The file containing predictions made on the validation dataset.
- **Model Files**:
  - `conv1d_model.h5`: Trained **Conv1D** model for news classification.
  - `logistic_regression_model.joblib`: Trained **Logistic Regression** model.
  - `naive_bayes_model.joblib`: Trained **Naïve Bayes** model.
  - `svm_model.joblib`: Trained **Support Vector Machine (SVM)** model.
  - `logistic_regression_model_word2vec.pkl`: **Logistic Regression** model with **Word2Vec** features.
  - `tokenizer.pkl`: Pretrained tokenizer for text preprocessing (used for models like Word2Vec).
  - `vectorizer.joblib`: Fitted **TF-IDF Vectorizer** for text feature extraction.


## Download the Dataset and Model Files

Due to the large size of the dataset and Conv1D model, they are stored on **Google Drive**. You can download the necessary files as follows:

- **Dataset (`data.csv`)**:  
  [Download data.csv from Google Drive](https://drive.google.com/file/d/1MHw-rA-nilAfOaLW18MlpGCzLgSttwGe/view?usp=sharing)
  
- **Conv1D Model (`conv1d_model.h5`)**:  
  [Download conv1d_model.h5 from Google Drive](https://drive.google.com/file/d/1sNWADRWP27uFqFr5rNLvhsVKw0uaOxZe/view?usp=sharing)
  - **Presentation**:
(https://www.canva.com/design/DAGh7DlRIow/1KFfUG1Nqp0Q0ROaIT9zTg/view?utm_content=DAGh7DlRIow&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=hb7e71a3c82)


## Live Demo

You can also test the Fake News Classifier app online using the following [Streamlit link](https://fake-news-detection-neural-core.streamlit.app/).
