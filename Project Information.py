import streamlit as st

with st.container():
    st.title("🏠 Home - Credit Card Fraud Detection System")

    st.markdown("""
    ### 💡 Project Overview
    Welcome to the **Credit Card Fraud Detection System**, a machine learning-powered application designed to detect potentially fraudulent credit card transactions in real time.

    This project uses a dataset containing **106 engineered features** derived from anonymized credit card transactions. These features are processed using a **logistic regression model** trained to distinguish between legitimate and fraudulent transactions with high accuracy.

    ### 📌 Purpose
    Fraudulent credit card transactions pose a serious threat to consumers and businesses alike. This system provides:
    - Early detection of suspicious activity
    - Insights based on transaction behavior
    - Easy-to-use interface for analysts, financial officers, and developers

    ### 🧠 Key Features
    - **EDA & Visualizations**: Explore the dataset and understand how different features relate to fraud.
    - **Manual & Smart Prediction**: Predict transaction legitimacy through smart questioning or full feature input.
    - **Chatbot Assistance**: Get fraud-related help and field explanations from an interactive assistant.
    - **Upload Predictions**: Upload your transaction files and receive instant fraud predictions.

    ### 📊 Dataset
    - **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mishra5001/credit-card)
    - **Features**: 106 anonymized variables including transaction amount, time, user behavior patterns, and more.
    - **Target**: Binary classification — `0` (Legitimate), `1` (Fraudulent)

    ### 🔧 Technologies Used
    - **Python** (Data Processing, Model Training)
    - **Scikit-Learn** (Logistic Regression Model)
    - **Pandas & NumPy** (Data Manipulation)
    - **Streamlit** (Frontend & UI)
    - **Matplotlib & Seaborn** (Visualization)

    ### 📚 Learning Goals
    - Explore end-to-end machine learning workflow
    - Practice feature engineering, model training, and deployment
    - Build user-friendly and interactive dashboards with Streamlit

    ---
    👇 Select a tab above to get started with **EDA**, **Model Predictions**, or explore the **Project Overview**!
    """)
