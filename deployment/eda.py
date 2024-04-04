import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import nltk
nltk.download('stopwords') # Stopwords
nltk.download('punkt') # Punctuation
nltk.download('wordnet') # Wordnet for lemmatization
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# st.set_page_config(page_title = "EDA")

def run():
    
    # Judul
    st.title("Movie Genre Classification")
    
    # Subheader
    st.subheader("Exploratory Data Analysis")
    
    # Data load, rename, type
    df = pd.read_csv('data.csv')

    # EDA before Pre-processing
    st.write("##### **EDA before Pre-processing**")

    # EDA 1: Pie chart 
    st.write("**Genre Distribution**")
    st.write('''Data terdiri dari 10 genre film berbeda, dan ternyata semua genre memiliki banyak data yang sama.''')
    fig_1 = plt.figure(figsize = (8,5))
    df['genre'].value_counts().plot(kind = 'pie', colormap = 'Spectral', autopct = '%.1f%%')
    plt.title('Movie Genre')
    plt.ylabel('')
    plt.show()
    st.pyplot(fig_1)
    
    # EDA 2: Sample
    st.write("**Sample Synopsis**")
    st.write('''Di bawah ini adalah contoh 5 sinopsis dari masing-masing genre.''')
    pd.set_option('display.max_colwidth', None)
    for genre in df['genre'].unique().tolist():
        print('Sample: ', genre)
        sample = df[df['genre'] == genre].sample(n = 5, random_state = 17)
        print(sample['synopsis'])
        print('-'*150)
    st.write('''Insight:''')
    st.markdown("- Diksi dan jumlah kata sinopsis film sangat beragam")
    st.markdown("- Sinopsis ditulis dalam Bahasa Inggris")
    st.markdown("- Ada beberapa kata dari sampel di atas yang sepertinya bukan bagian dari sinopsis yang ditulis filmmakers: See full synopsis, ---KBBO-AINOS")
    st.markdown("- Banyak nama karakter yang ditemukan, sepertinya adalah karakter utama dari film")
    st.markdown("- Ada kata-kata tertentu yang jelas berkaitan dengan genre, seperti 'abandoned' di genre horror, 'disappears' di genre mystery, dan lain-lain")

    # EDA 3: Word count
    st.write('''Word count atau length dari synopsis untuk sinopsis dari beragam genre tidak berbeda secara signifikan.''')
    df['synopsis_length'] = df['synopsis'].apply(lambda x: len(nltk.word_tokenize(x)))
    for genre in df['genre'].unique():
        print('Genre: ', genre)
        print('Minimum word count: ', df[df['genre'] == genre]['synopsis_length'].min())
        print('Maximum word count: ', df[df['genre'] == genre]['synopsis_length'].max())
        print('Average word count: ', round(df[df['genre'] == genre]['synopsis_length'].mean(), 2))
        print('-'*100)
    fig_2 = plt.figure(figsize = (5,5))
    sns.barplot(data = df, x = 'genre', y = 'synopsis_length')
    plt.title('Average of Synopsis Words Length')
    plt.xticks(rotation = 90)
    plt.ylabel('Words')
    plt.xlabel('Genre')
    plt.show()
    st.pyplot(fig_2)

    # EDA 4: Word cloud
    st.write('''Di bawah ini adalah Word Cloud untuk keseluruhan genre, serta untuk per genre.''')
    st.write('''Insight:''')
    st.markdown("- Semua genre film mengandung kata 'find' dan 'life' yang sepertinya cukup sering muncul, melihat ukurannya pada masing-masing word cloud (kata 'life' tidak muncul sebanyak 'find')")
    st.markdown("- Selalu ada kata yang menggambarkan angka di word cloud, seperti 'one', 'two', dan lainnya. Ini akan menjadi tambahan untuk stopwords, karena tidak menambah makna sinopsis")
    st.markdown("- Terlihat kata yang kaitannya erat dengan genre, seperti 'father' pada genre family, 'alien' pada scifi, 'murder' pada crime dan mystery, dan lain-lain")

    text = df['synopsis'].values
    fig_3 = plt.figure(figsize = (10,5))
    plt.imshow(WordCloud(background_color="white").generate(" ".join(text)))
    plt.axis("off")
    plt.title('All Genre')
    plt.show()
    st.pyplot(fig_3)

    st.markdown("---")

    # Drop rows
    df = df[df['synopsis'].str.contains('Plot undisclosed.|Plot under wraps|Plot Unknown|Plot Unknown.|Under wraps|NA.|In development|Undisclosed plot|TBA|Plot internal.|TBD') == False]

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
    df['synopsis_processed'] = df['synopsis'].apply(lambda x: text_preprocessing(x))
    df[['synopsis', 'synopsis_processed']].head()


    # EDA after Pre-processing
    st.write("##### **EDA after Pre-processing**")

    # EDA 1: Pie chart 
    st.write("**Genre Distribution after Processing**")
    st.write('''Karena dari df sudah di-drop beberapa data, ingin dilihat bagaimana porsi tiap genre pada dataset.''')
    fig_4 = plt.figure(figsize = (8,5))
    df['genre'].value_counts().plot(kind = 'pie', colormap = 'Spectral', autopct = '%.1f%%')
    plt.title('Movie Genre (Processed)')
    plt.ylabel('')
    plt.show()
    st.pyplot(fig_4)
    
    # EDA 2: Sample
    st.write("**Sample Synopsis**")
    st.write('''Di bawah ini adalah contoh 5 sinopsis dari masing-masing genre yang sudah melalui text-processing.''')
    pd.set_option('display.max_colwidth', None)
    for genre in df['genre'].unique().tolist():
        print('Sample: ', genre)
        sample = df[df['genre'] == genre].sample(n = 5, random_state = 17)
        print(sample['synopsis_processed'])
        print('-'*150)
    st.write('''Insight:''')
    st.markdown("- Pre-processing bekerja dengan baik")
    st.markdown("- Sinopsis ditulis dalam Bahasa Inggris")
    st.markdown("- Sinopsis yang sudah diproses terlihat mengandung kata yang jauh lebih sedikit")
    st.markdown("- Masih ada nama karakter yang ditemukan, karena nama orang memang tidak termasuk dalam stopwords sehingga tidak ter-handle")

    # EDA 3: Word count
    st.write('''Ingin dilihat bagaimana perbedaan word count setelah sinopsis melalui pre-processing. Ternyata jumlah kata dalam sinopsis sekarang turun jauh angkanya, baik saat dilihat data per data maupun secara rata-rata. Penurunan jumlah kata hampir setengah dari sinopsis, dan ini hal yang baik karena model bisa mempertimbangkan hanya kata-kata yang lebih penting atau bermakna saja''')
    df['synopsis_processed_length'] = df['synopsis_processed'].apply(lambda x: len(nltk.word_tokenize(x)))
    pd.DataFrame({'Length of Synopsis (Before)':df['synopsis_length'],
                'Length of Synopsis (After)':df['synopsis_processed_length'],
                'Reduction (%)':round((df['synopsis_length'] - df['synopsis_processed_length'])/df['synopsis_length']*100, 2)
                })
    fig_5 = plt.figure(figsize = (5,5))
    sns.barplot(data = df, x = 'genre', y = 'synopsis_processed_length')
    plt.title('Average of Synopsis (Processed) Words Length')
    plt.xticks(rotation = 90)
    plt.ylabel('Words')
    plt.xlabel('Genre')
    plt.show()
    st.pyplot(fig_5)

    # EDA 4: Word cloud
    st.write('''Terlihat di bawah ini, word cloud tidak lagi mengandung kata yang mendeskripsikan angka dan single-letter word. Kata yang berukuran besar pada word cloud sebelum pre-processing, sekarang menjadi lebih besar lagi atau dapat dikatakan kemunculannya menjadi lebih dominan dari sebelumnya''')
    st.write('''Insight:''')
    st.markdown("- Semua genre film mengandung kata 'find' dan 'life' yang sepertinya cukup sering muncul, melihat ukurannya pada masing-masing word cloud (kata 'life' tidak muncul sebanyak 'find')")
    st.markdown("- Selalu ada kata yang menggambarkan angka di word cloud, seperti 'one', 'two', dan lainnya. Ini akan menjadi tambahan untuk stopwords, karena tidak menambah makna sinopsis")
    st.markdown("- Terlihat kata yang kaitannya erat dengan genre, seperti 'father' pada genre family, 'alien' pada scifi, 'murder' pada crime dan mystery, dan lain-lain")

    text_processed = df['synopsis_processed'].values
    fig_6 = plt.figure(figsize = (10,5))
    plt.imshow(WordCloud(background_color="black").generate(" ".join(text_processed)))
    plt.axis("off")
    plt.title('All Genre (Processed)')
    plt.show()
    st.pyplot(fig_6)

if __name__ == "__main__":
    run()