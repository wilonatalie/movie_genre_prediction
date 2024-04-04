import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
import string
import nltk
nltk.download('stopwords') # Stopwords
nltk.download('punkt') # Punctuation
nltk.download('wordnet') # Wordnet for lemmatization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow
from tensorflow.keras.saving import load_model

# st.set_page_config(page_title = "Prediction")

# Load file
model = load_model('model_nlp_final')

def run():
    
    # Judul
    st.title("Movie Genre Classification")

    # Subheader
    st.subheader("Prediction")     

    # Buat form input
    with st.form("Movie Synopsis"):
        text = st.text_area("Text to analyze", "Insert synopsis here")
        # Submit
        submitted = st.form_submit_button('Predict')
    
    df_inf = pd.DataFrame({'synopsis':text}, index = [0])

    # Define stopwords
    stopwords_nltk = list(set(stopwords.words('english')))
    stopwords_add = ['See full synopsis', 'KBBO-AINOS', 'one',  'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    stopwords_all = stopwords_nltk + stopwords_add

    # Define lemmatizer
    lemmatizer = WordNetLemmatizer()

    # Define pre-processing function
    def text_preprocessing(document):
        # Case folding
        document = document.lower()
        # Number, symbols, punctuations removal (keeping only letters & whitespace)
        document = re.sub('[^a-z\s]+', ' ', document)
        # Remove single character if any
        document = re.sub(r'\s+.\s', ' ', document)
        # Whitespace, tabs, new lines
        document = document.strip()
        # Tokenization
        tokens = word_tokenize(document)
        # Stopwords removal
        tokens = [word for word in tokens if word not in stopwords_all]
        # Lemmatization
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Combining Tokens
        document = ' '.join(tokens)
        return document

    # Pre-process synopsis
    df_inf['synopsis_processed'] = df_inf['synopsis'].apply(lambda x: text_preprocessing(x))

    if submitted:
        # Memprediksi data inferens
        y_pred_inf_proba = model.predict(df_inf['synopsis_processed'])
        y_pred_inf = np.argmax(y_pred_inf_proba, axis=-1)

        # Menampilkan hasil prediksi
        if y_pred_inf == 0:
            st.write('Movie genre: Fantasy')
        elif y_pred_inf == 1:
            st.write('Movie genre: Horror')
        elif y_pred_inf == 2:
            st.write('Movie genre: Family')
        elif y_pred_inf == 3:
            st.write('Movie genre: Sci-fi')
        elif y_pred_inf == 4:
            st.write('Movie genre: Action')
        elif y_pred_inf == 5:
            st.write('Movie genre: Crime')
        elif y_pred_inf == 6:
            st.write('Movie genre: Adventure')
        elif y_pred_inf == 7:
            st.write('Movie genre: Mystery')
        elif y_pred_inf == 8:
            st.write('Movie genre: Romance')
        else:
            st.write('Movie genre: Thriller')

if __name__ == "__main__":
  run()