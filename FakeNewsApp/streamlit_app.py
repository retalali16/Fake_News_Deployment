
import streamlit as st
import joblib
import os

BASE_PATH = "."
MODELS = {
    "SVM": os.path.join(BASE_PATH, "svm_model.pkl"),
    "Logistic Regression": os.path.join(BASE_PATH, "fake_news_model.pkl")
}
VECTORIZER_PATH = os.path.join(BASE_PATH, "tfidf_vectorizer.pkl")
IMAGE_PATH = os.path.join(BASE_PATH, "fake news.jpg")

try:
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    st.error("Vectorizer file not found! Make sure 'tfidf_vectorizer.pkl' is in the same folder as this app.")
    st.stop()


st.set_page_config(page_title="Fake News Detector", layout="centered")


st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>📰 Fake News Detection App</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center; font-size:18px;'>Detect whether a news statement is Real or Fake using ML!</p>",
    unsafe_allow_html=True
)

st.image(IMAGE_PATH, use_container_width=True)


st.subheader("Choose a Machine Learning Model:")
model_choice = st.selectbox("Select Model:", list(MODELS.keys()))


try:
    model = joblib.load(MODELS[model_choice])
except FileNotFoundError:
    st.error(f"{model_choice} model file not found! Make sure it's in the same folder as this app.")
    st.stop()


st.subheader("Enter a news statement to analyze:")
user_input = st.text_area("Type your news headline or statement here:", height=150)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a news statement to analyze.")
    else:
        input_data = vectorizer.transform([user_input])
        prediction = model.predict(input_data)[0]
        confidence = model.predict_proba(input_data).max() * 100 if hasattr(model, "predict_proba") else None
        
        if prediction == 0:
            result = "✅ REAL news detected"
            color = "#00B050"
        else:
            result = "❌ FAKE news detected"
            color = "#FF0000"
        
        st.markdown(f"<h2 style='color:{color}; text-align:center;'>{result}</h2>", unsafe_allow_html=True)
        if confidence:
            st.markdown(f"<h4 style='text-align:center;'>Confidence: {confidence:.2f}%</h4>", unsafe_allow_html=True)








