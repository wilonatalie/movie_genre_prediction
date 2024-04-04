import streamlit as st
import pandas as pd
from PIL import Image

# st.set_page_config(page_title = "Home")

def run():
    
    # Judul
    st.title("Movie Genre Classification")
    
    # Subheader
    st.subheader("Home")
    
    # Intro
    st.write("**Graded Challenge 7**")
    st.write('''Nama: Wilona Natalie Elvaretta  
    Batch: RMT-028''')
    st.markdown("---")

    # Problem statement
    st.write("**Problem**")
    image = Image.open('genre.jpg')
    st.image(image)
    st.write('''Genre classification berguna untuk banyak hal, beberapa di antaranya sebagai berikut:''')
    st.markdown("- Filmmaker bisa menguji apakah sinopsis yang mereka draft bisa dikategorikan sebagai genre yang dikehendaki. Jika prediksi tepat, penonton akan mudah untuk menangkap genre film yang ditawarkan hanya dengan membaca sinopsis")
    st.markdown("- Penonton dimungkinkan untuk melihat genre dari sinopsis, untuk memastikan mereka tidak menonton genre yang tidak diinginkan. Contoh, ada beberapa orang yang sangat menghindari film horor")
    st.markdown("- OTT platforms mampu mengkategorikan film dengan lebih akurat, dengan mengklarifikasi sinopsis dengan informasi genre yang diberikan")
    st.markdown('''Objektif dari program ini adalah untuk menjadi sarana mengkategorikan film ke genre yang sesuai berdasarkan sinopsisnya, dan diharapkan bisa dipakai oleh berbagai macam pengguna. Model dibuat berdasarkan algoritma Artificial Neural Network (ANN) yang terbaik performanya. Metric yang dipilih untuk menguji performa adalah Accuracy, karena ingin diminimalisir film yang salah diprediksi dan fokus pada model "correctness".
    Program diharapkan selesai per tanggal 19 Maret 2024.''')
    st.markdown("---")
    
    # Dataset
    st.write("**Dataset**")
    st.write("Dataset adalah tentang film, dan dapat diakses [di sini](https://www.kaggle.com/datasets/guru001/movie-genre-prediction/data?select=sample_submission.csv).")
    st.write("Ada 54000 film berbeda dengan 4 atribut, yaitu ID, judul, sinopsis, dan genre.")
    df = pd.read_csv('data.csv')
    st.dataframe(df)

if __name__ == "__main__":
  run()