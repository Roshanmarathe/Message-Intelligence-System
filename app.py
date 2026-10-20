import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Message Intelligence", page_icon="🛡️", layout="wide")

MODEL_PATH = Path(__file__).parent / "model" / "best_spam_model.joblib"

st.markdown("""
<style>
.stApp {background: radial-gradient(circle at 8% 0%, rgba(69,117,255,.13), transparent 28%),
radial-gradient(circle at 95% 8%, rgba(20,184,166,.10), transparent 24%);}
.block-container {max-width: 1350px; padding-top: 1.8rem;}
.hero {padding: 1.6rem 1.8rem; border-radius: 22px; color: white;
background: linear-gradient(120deg,#172554,#115e59); margin-bottom: 1.2rem;}
.hero h1 {margin:0; font-size:2.1rem;}
.hero p {color:#dbeafe; margin:.5rem 0 0;}
.result {padding:1.3rem; border-radius:16px; border:1px solid #64748b; margin-top:1rem;}
.safe {border-left:6px solid #10b981;} .spam {border-left:6px solid #ef4444;}
div[data-testid="stMetric"] {padding:.8rem; border-radius:12px; border:1px solid rgba(130,150,190,.2);}
</style>
<div class="hero">
  <div>SUPERVISED MACHINE LEARNING · CLASSIFICATION</div>
  <h1>🛡️ Message Intelligence System</h1>
  <p>Spam vs Legitimate prediction using your trained machine-learning pipeline.</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


def default_for(feature):
    defaults = {
        "message_length": 120, "word_count": 20, "num_urls": 0,
        "num_digits": 0, "num_special_chars": 2, "spam_keyword_score": 0.0,
        "legit_keyword_score": 0.0, "sender_activity_score": 50.0,
        "sender_account_age_days": 180, "messages_sent_last_24h": 5,
        "hour_of_day": 12, "day_of_week": 2,
    }
    return defaults.get(feature, 0.0)


def help_for(feature):
    descriptions = {
        "message_length": "Total number of characters in the message.",
        "word_count": "Number of words in the message.",
        "num_urls": "Number of URLs or links.",
        "num_digits": "Number of numeric digits.",
        "num_special_chars": "Count of special characters.",
        "spam_keyword_score": "Spam-keyword score as defined in the training dataset.",
        "legit_keyword_score": "Legitimate-keyword score as defined in the training dataset.",
        "sender_activity_score": "Sender activity score from the dataset.",
        "sender_account_age_days": "Sender account age in days.",
        "messages_sent_last_24h": "Messages sent by sender during the last 24 hours.",
        "hour_of_day": "Hour of day, 0–23.",
        "day_of_week": "Day-of-week encoding used by the dataset.",
    }
    return descriptions.get(feature, f"Numerical input feature: {feature}")


if not MODEL_PATH.exists():
    st.error(f"Model file not found: `{MODEL_PATH}`")
    st.info("Notebook se model export karke `model/best_spam_model.joblib` path par rakho.")
    st.stop()

try:
    bundle = load_bundle(str(MODEL_PATH))
    model = bundle["model"]
    model_name = bundle.get("model_name", type(model).__name__)
    feature_columns = bundle["feature_columns"]
    target_names = bundle.get("target_names", {0: "Legitimate Message", 1: "Spam Message"})
except Exception as exc:
    st.error("Model load nahi ho paaya. Compatible scikit-learn/joblib versions check karo.")
    st.exception(exc)
    st.stop()

with st.sidebar:
    st.header("About")
    st.write("Enter the numerical features used during model training.")
    st.markdown("🟢 **0 — Legitimate**")
    st.markdown("🔴 **1 — Spam**")
    st.caption(f"Loaded model: {model_name}")
    st.code("model/best_spam_model.joblib")

st.caption(f"Active model: **{model_name}**")
tab_predict, tab_info = st.tabs(["🔎 Predict message", "ℹ️ Model information"])

with tab_predict:
    st.subheader("Message and sender features")
    st.write("This app predicts from numerical dataset features; it does not analyze raw text directly.")
    values = {}
    with st.form("predict_form"):
        left, right = st.columns(2)
        for i, feature in enumerate(feature_columns):
            col = left if i % 2 == 0 else right
            default = default_for(feature)
            with col:
                if feature in {
                    "message_length", "word_count", "num_urls", "num_digits",
                    "num_special_chars", "sender_account_age_days",
                    "messages_sent_last_24h", "hour_of_day", "day_of_week"
                }:
                    max_value = 23 if feature == "hour_of_day" else 6 if feature == "day_of_week" else 100000
                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        min_value=0, max_value=max_value,
                        value=int(max(0, min(max_value, default))),
                        step=1, help=help_for(feature)
                    )
                else:
                    values[feature] = st.number_input(
                        feature.replace("_", " ").title(),
                        value=float(default), step=0.1, format="%.3f",
                        help=help_for(feature)
                    )
        submitted = st.form_submit_button("🔍 Analyze Message", type="primary", use_container_width=True)

    if submitted:
        try:
            input_df = pd.DataFrame([[values[c] for c in feature_columns]], columns=feature_columns)
            prediction = int(model.predict(input_df)[0])
            label = target_names.get(prediction, str(prediction))
            if prediction == 1:
                st.markdown(
                    f'<div class="result spam"><h2>🚨 {label}</h2><p>Model classifies these features as Spam.</p></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="result safe"><h2>✅ {label}</h2><p>Model classifies these features as Legitimate.</p></div>',
                    unsafe_allow_html=True
                )

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(input_df)[0]
                classifier = model.named_steps.get("classifier") if hasattr(model, "named_steps") else model
                classes = list(getattr(classifier, "classes_", []))
                if classes:
                    probs = {int(c): float(p) for c, p in zip(classes, probabilities)}
                    a, b = st.columns(2)
                    with a:
                        st.metric("Spam probability", f"{probs.get(1, 0) * 100:.1f}%")
                        st.progress(probs.get(1, 0))
                    with b:
                        st.metric("Legitimate probability", f"{probs.get(0, 0) * 100:.1f}%")
                        st.progress(probs.get(0, 0))
                    st.caption("Probabilities are model estimates, not guarantees.")
            with st.expander("View input features"):
                st.dataframe(input_df, use_container_width=True, hide_index=True)
        except Exception as exc:
            st.error("Prediction failed. Check the input schema and saved model.")
            st.exception(exc)

with tab_info:
    c1, c2 = st.columns(2)
    c1.metric("Required features", len(feature_columns))
    c2.metric("Estimator", type(model).__name__)
    st.subheader("Model input schema")
    st.code("\n".join(feature_columns))
    st.warning("Use the same feature definitions, units, and encodings as the training dataset. Validate before real-world use.")

st.divider()
st.caption("Message Intelligence System · Educational project")
