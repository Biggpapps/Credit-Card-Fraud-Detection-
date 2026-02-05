import streamlit as st

with st.container():
    st.title("🏠 Home - Credit Card Fraud Detection System")

    st.markdown(
        """
    ### 💡 Project Overview
    Welcome to the **Credit Card Fraud Detection System**, a machine learning-powered application designed to identify potentially fraudulent credit card transactions.

    The app uses **106 engineered features** derived from anonymized transaction records and serves both:
    - **Interpretation** (risk summaries, probability bands, and visual distributions)
    - **Prediction** (single-record and bulk CSV scoring)

    ### 📌 Why this matters
    Fraud detection is a high-impact, high-risk decisioning problem. This system is built to support analysts with:
    - Early alerts for suspicious behavior
    - Probability-based risk scoring (not only binary labels)
    - Threshold tuning to balance missed fraud vs false alarms

    ### 🧠 Core Capabilities
    - **Smart / Manual Input Prediction**: Quickly test one transaction with fraud probability and recommended action.
    - **Batch CSV Scoring**: Upload files and get record-level predictions, risk bands, and downloadable outputs.
    - **Risk Interpretation Aids**: Review top suspicious transactions and aggregate risk distribution.
    - **Help Assistant**: Built-in FAQ and field guidance for quick understanding.

    ### 📊 Dataset
    - **Source**: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mishra5001/credit-card)
    - **Features**: 106 anonymized variables from transaction behavior signals
    - **Target**: Binary classification — `0` (Legitimate), `1` (Fraudulent)

    ### 🔧 Technologies Used
    - **Python** (Data processing and model serving)
    - **Scikit-Learn** (Classification model)
    - **Pandas & NumPy** (Data handling)
    - **Streamlit** (Interactive app)

    ### ✅ Best-practice reminder
    Treat model output as a **decision-support signal**, not a final verdict. For high-risk cases, combine model alerts with manual review and business rules.

    ---
    👇 Use the tabs to explore predictions, interpretation views, and project details.
    """
    )
