import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Set page configuration
st.set_page_config(page_title="Credit Fraud Predictor", layout="wide")

# Set dark, blurred background
def set_blurred_dark_background(url):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(0,0,0,0.75), rgba(0,0,0,0.85)), url({url});
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_blurred_dark_background("https://aihubprojects.com/wp-content/uploads/2020/01/ccfd-scaled.jpeg")

# Load trained model
@st.cache_data
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# Title
st.title("💳 Credit Application Fraud Detection (106 Features)")
st.markdown("Choose your input method to detect potential fraud:")

# Input mode selection
input_mode = st.radio("Input Mode:", ["🧠 Smart Input", "📂 Full Manual Entry", "📁 Upload CSV File"])
user_input = []
submitted = False

# 🧠 Smart Input
if input_mode == "🧠 Smart Input":
    st.subheader("🧠 Answer a few simple questions")
    amount = st.number_input("Transaction Amount", min_value=0.0)
    hour = st.slider("Hour of Transaction", 0, 23)
    region_match = st.radio("Transaction in your usual region?", ["Yes", "No"])
    frequent_today = st.radio("High usage today?", ["Yes", "No"])
    known_merchant = st.radio("Merchant well known?", ["Yes", "No"])

    mapped = [0.0] * 106
    mapped[0] = amount
    mapped[1] = hour
    mapped[2] = 1 if region_match == "Yes" else 0
    mapped[3] = 1 if frequent_today == "Yes" else 0
    mapped[4] = 1 if known_merchant == "Yes" else 0
    user_input = mapped

    submitted = st.button("🔍 Predict")

# 📂 Full Manual Entry
elif input_mode == "📂 Full Manual Entry":
    st.subheader("📂 Enter all 106 Features Manually")
    with st.form("manual_106"):
        for i in range(1, 107):
            val = st.number_input(f"Feature F{i}", value=0.0, step=0.01)
            user_input.append(val)
        submitted = st.form_submit_button("🔍 Predict")

# 📁 Upload CSV File
elif input_mode == "📁 Upload CSV File":
    st.subheader("📁 Upload a CSV File with F1 to F106")
    file = st.file_uploader("Choose your CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
        if df.shape[1] != 106:
            st.error("⚠️ Your file must contain exactly 106 columns (F1 to F106).")
        else:
            st.dataframe(df.head())
            if st.button("🔍 Predict from File"):
                pred = model.predict(df)
                prob = model.predict_proba(df)[:, 1]
                df["Prediction"] = pred
                df["Fraud Probability"] = prob
                st.write(df)

# 🔮 Prediction (for Smart Input or Manual Entry)
if submitted and len(user_input) == 106:
    arr = np.array([user_input])
    pred = model.predict(arr)[0]
    prob = model.predict_proba(arr)[0][1]

    st.subheader("🔎 Prediction Result")
    if pred == 1:
        st.error(f"🚨 Fraud Detected (Probability: {prob:.2%})")
    else:
        st.success(f"✅ Legitimate (Fraud Probability: {prob:.2%})")

# 🤖 Sidebar Chatbot
st.sidebar.title("🤖 Need Help?")

tab = st.sidebar.radio("Choose Help Topic:", ["📘 Field Explanation", "💡 Fraud FAQs"])

# Field explanation bot
if tab == "📘 Field Explanation":
    feature = st.sidebar.selectbox("Select a feature", [f"F{i}" for i in range(1, 107)])
    st.sidebar.write(f"📘 {feature} is an anonymized feature derived from user or transaction behavior.")

# FAQ bot
else:
    faqs = {
        "What is credit fraud?": "It's when a transaction is made without the owner’s authorization.",
        "Can ML models detect all frauds?": "No. They help reduce risk but can't guarantee 100% accuracy.",
        "Why are features anonymized?": "To protect sensitive data and comply with privacy policies.",
        "Why 106 features?": "These features represent complex patterns in the application or transaction data.",
        "How accurate is the model?": "It was trained and validated on historical data and performs well, but results should be manually reviewed when needed."
    }
    q = st.sidebar.selectbox("Select a question", list(faqs.keys()))
    st.sidebar.write("💬", faqs[q])
