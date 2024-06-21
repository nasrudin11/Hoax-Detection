import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Download NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

# Function for text preprocessing
def preprocess_text(text):
    # Convert text to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r'\W', ' ', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Apply stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    # Join tokens back into text
    preprocessed_text = ' '.join(stemmed_tokens)
    return preprocessed_text

# Main function to run the Streamlit app
def main():
    st.title("Text Preprocessing and Prediction App")
    st.write("This app preprocesses text and makes predictions using a pre-trained SVM model.")

    # Load the pre-trained SVM model
    with open('svm_model_linear.pkl', 'rb') as f:
        svm_model_linear = pickle.load(f)

    # Load the pre-trained TfidfVectorizer
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)

    # Get user input
    text_input = st.text_input("Enter some news text:")
    if text_input:
        # Preprocess the text
        preprocessed_text = preprocess_text(text_input)
        st.write("Preprocessed Text:")
        st.write(preprocessed_text)

        # Vectorize the preprocessed text
        X_text = vectorizer.transform([preprocessed_text])

        # Make prediction
        prediction = svm_model_linear.predict(X_text)

        # Display prediction
        if prediction[0] == 1:
            st.write("Predicted News: Real News")
        else:
            st.write("Predicted News: Fake News")

# Run the app
if __name__ == '__main__':
    main()
