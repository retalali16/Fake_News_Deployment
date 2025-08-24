import streamlit as st
import joblib
import os
import urllib.request

# ------------------------
# Paths & Constants
# ------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))  # ensures correct path on deployment
MODELS = {
    "SVM": os.path.join(BASE_PATH, "svm_model.pkl"),
    "Logistic Regression": os.path.join(BASE_PATH, "fake_news_model.pkl")
}
VECTORIZER_PATH = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")
IMAGE_PATH = os.path.join(BASE_PATH, "fake news.jpg")

# ------------------------
# Streamlit Page Config
# ------------------------
st.set_page_config(page_title="Fake News Detector", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>📰 Fake News Detection App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Detect whether a news statement is Real or Fake using ML!</p>",
    unsafe_allow_html=True
)

if os.path.exists(IMAGE_PATH):
    st.image(IMAGE_PATH, use_container_width=True)

# ------------------------
# Load Vectorizer
# ------------------------
if not os.path.exists(VECTORIZER_PATH):
    st.error("Vectorizer file not found! Make sure 'tfidf_vectorizer.pkl' is in the same folder as this app.")
    st.stop()

vectorizer = joblib.load(VECTORIZER_PATH)

# ------------------------
# Model Selection
# ------------------------
st.subheader("Choose a Machine Learning Model:")
model_choice = st.selectbox("Select Model:", list(MODELS.keys()))

if not os.path.exists(MODELS[model_choice]):
    st.error(f"{model_choice} model file not found! Make sure it's in the same folder as this app.")
    st.stop()

model = joblib.load(MODELS[model_choice])

# ------------------------
# User Input
# ------------------------
st.subheader("Enter a news statement to analyze:")
user_input = st.text_area("Type your news headline or statement here:", height=150)

# ------------------------
# Analyze Button
# ------------------------
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news statement to analyze.")
    else:
        # Transform input
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(input_data).max() * 100

        # Display Result
        if prediction == 0:
            result = "✅ REAL news detected"
            color = "#00B050"
        else:
            result = "❌ FAKE news detected"
            color = "#FF0000"

        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{result}</h2>", unsafe_allow_html=True)
        if confidence is not None:
            st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)
