import streamlit as st
import eda
import prediction
import home

page = st.sidebar.selectbox('Page: ', ('Home', 'Exploratory Data Analysis', 'Prediction'))

if page == 'Exploratory Data Analysis':
    eda.run()
elif page == 'Prediction':
    prediction.run()
else:
    home.run()
