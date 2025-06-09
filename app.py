

import streamlit as st
import joblib
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load saved model and vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Label map
emotion_labels = {
    0: "😢 Sadness",
    1: "😊 Joy",
    2: "❤️ Love",
    3: "😡 Anger",
    4: "😨 Fear",
    5: "😲 Surprise"
}

# Clean text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    return " ".join([word for word in text.split() if word not in stop_words])

# Prediction
def predict_emotion(text):
    cleaned = clean_text(text)
    if not cleaned.strip():
        return "⚠️ Please enter meaningful text."
    tfidf_vec = vectorizer.transform([cleaned]).toarray()
    pred = model.predict(tfidf_vec)[0]
    return emotion_labels.get(pred, "❓ Unknown")

# Streamlit UI
st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
st.title("🧠 Emotion Classifier")
st.markdown("Enter your text to detect emotion!")

user_input = st.text_area("✍️ Type something...")

if st.button("🔍 Predict Emotion"):
    result = predict_emotion(user_input)
    st.success(f"**Detected Emotion:** {result}")
