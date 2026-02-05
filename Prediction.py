import pickle

import numpy as np
import pandas as pd
import streamlit as st

# Set page configuration
st.set_page_config(page_title="Credit Fraud Predictor", layout="wide")


def set_blurred_dark_background(url: str) -> None:
    """Apply dark blurred background image to the app."""
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
        unsafe_allow_html=True,
    )


set_blurred_dark_background("https://aihubprojects.com/wp-content/uploads/2020/01/ccfd-scaled.jpeg")


@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)


model = load_model()
EXPECTED_FEATURES = [f"F{i}" for i in range(1, 107)]


def to_feature_frame(user_values: list[float]) -> pd.DataFrame:
    """Convert a list of 106 user values to a model-ready DataFrame."""
    return pd.DataFrame([user_values], columns=EXPECTED_FEATURES)


def normalize_uploaded_frame(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    """
    Normalize uploaded CSV to expected model schema.

    Accepts either:
    - Exactly 106 columns (any names): renamed to F1..F106 in order.
    - Data including F1..F106 columns: reordered to expected order.
    """
    if df.shape[1] == 106:
        out = df.copy()
        out.columns = EXPECTED_FEATURES
        return out, None

    missing = [c for c in EXPECTED_FEATURES if c not in df.columns]
    if missing:
        return None, f"Missing required columns: {', '.join(missing[:8])}{'...' if len(missing) > 8 else ''}"

    return df[EXPECTED_FEATURES].copy(), None


def add_prediction_columns(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Append probability, label, and action columns."""
    scores = model.predict_proba(df)[:, 1]
    labels = (scores >= threshold).astype(int)

    out = df.copy()
    out["Fraud Probability"] = scores
    out["Prediction"] = labels
    out["Risk Band"] = pd.cut(
        scores,
        bins=[-0.001, 0.20, 0.50, 0.80, 1.0],
        labels=["Low", "Moderate", "High", "Critical"],
    )
    out["Recommended Action"] = np.where(
        labels == 1,
        "Hold and manually review",
        "Approve (monitor normally)",
    )
    return out


# Title
st.title("💳 Credit Application Fraud Detection (106 Features)")
st.markdown("Choose your input method to detect potential fraud:")

# Decision threshold gives business teams control over sensitivity
threshold = st.slider(
    "Fraud decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.01,
    help="Higher threshold reduces false alarms; lower threshold catches more suspicious transactions.",
)

# Input mode selection
input_mode = st.radio("Input Mode:", ["🧠 Smart Input", "📂 Full Manual Entry", "📁 Upload CSV File"])
user_input = []
submitted = False

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

    st.caption("Only key fields are asked; remaining engineered features default to 0.")
    submitted = st.button("🔍 Predict")

elif input_mode == "📂 Full Manual Entry":
    st.subheader("📂 Enter all 106 Features Manually")
    with st.form("manual_106"):
        for i in range(1, 107):
            val = st.number_input(f"Feature F{i}", value=0.0, step=0.01)
            user_input.append(val)
        submitted = st.form_submit_button("🔍 Predict")

elif input_mode == "📁 Upload CSV File":
    st.subheader("📁 Upload a CSV File")
    st.caption("Accepted formats: exactly 106 columns, or explicitly named columns F1..F106.")
    file = st.file_uploader("Choose your CSV file", type=["csv"])

    if file:
        raw = pd.read_csv(file)
        st.write("Preview of uploaded data")
        st.dataframe(raw.head())

        normalized, err = normalize_uploaded_frame(raw)
        if err:
            st.error(f"⚠️ {err}")
        elif normalized.isnull().any().any():
            st.error("⚠️ Data contains missing values. Please clean your file and try again.")
        else:
            non_numeric = normalized.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                st.error(f"⚠️ Non-numeric columns found: {', '.join(non_numeric[:8])}")
            elif st.button("🔍 Predict from File"):
                result = add_prediction_columns(normalized, threshold)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(result))
                with col2:
                    st.metric("Flagged as Fraud", int(result["Prediction"].sum()))
                with col3:
                    st.metric("Average Fraud Probability", f"{result['Fraud Probability'].mean():.2%}")

                st.subheader("Risk Distribution")
                risk_counts = result["Risk Band"].value_counts().sort_index()
                st.bar_chart(risk_counts)

                st.subheader("Top 20 Most Suspicious Transactions")
                top_suspicious = result.sort_values("Fraud Probability", ascending=False).head(20)
                st.dataframe(top_suspicious)

                csv_bytes = result.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Download prediction results",
                    data=csv_bytes,
                    file_name="fraud_predictions.csv",
                    mime="text/csv",
                )

# Single prediction output
if submitted and len(user_input) == 106:
    frame = to_feature_frame(user_input)
    result = add_prediction_columns(frame, threshold).iloc[0]

    st.subheader("🔎 Prediction Result")
    st.progress(float(result["Fraud Probability"]))
    st.write(f"**Fraud Probability:** {result['Fraud Probability']:.2%}")
    st.write(f"**Risk Band:** {result['Risk Band']}")
    st.write(f"**Recommended Action:** {result['Recommended Action']}")

    if int(result["Prediction"]) == 1:
        st.error("🚨 Flagged as potentially fraudulent based on selected threshold.")
    else:
        st.success("✅ Classified as legitimate under current threshold.")

# Sidebar assistant
st.sidebar.title("🤖 Need Help?")
tab = st.sidebar.radio("Choose Help Topic:", ["📘 Field Explanation", "💡 Fraud FAQs"])

if tab == "📘 Field Explanation":
    feature = st.sidebar.selectbox("Select a feature", EXPECTED_FEATURES)
    st.sidebar.write(
        f"📘 {feature} is an anonymized feature derived from user or transaction behavior and transformed for model training."
    )
else:
    faqs = {
        "What is credit fraud?": "A transaction made without the owner’s authorization.",
        "Can ML models detect all frauds?": "No. They reduce risk, but no model guarantees 100% detection.",
        "Why are features anonymized?": "To protect sensitive data and preserve privacy.",
        "How should I choose threshold?": "Use lower values for stronger detection (more alerts), higher values for fewer false alarms.",
        "How accurate is the model?": "Model quality depends on data drift and class imbalance, so monitor performance over time.",
    }
    q = st.sidebar.selectbox("Select a question", list(faqs.keys()))
    st.sidebar.write("💬", faqs[q])
