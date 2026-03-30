import streamlit as st
import pickle
import string
import nltk
from scipy.sparse import hstack

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

st.set_page_config(page_title="Spam Classifier", page_icon="📧")
st.title("📧 SMS & Email Spam Classifier")
st.write("Enter the message below to check whether it's Spam or Not Spam (Ham).")

# Load model and vectorizer safely
@st.cache_resource
def load_models():
    with open("spam_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please ensure train.py was run to generate the models.")
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

input_message = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if input_message.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner("Analyzing message..."):
            text_cleaned = clean_text(input_message)
            text_vec = vectorizer.transform([text_cleaned])
            
            prediction = model.predict(text_vec)[0]
            
            if prediction == 1:
                st.error("🚨 This message is classified as **SPAM**!")
                st.snow() # Optional fun effect
            else:
                st.success("✅ This message is **NOT SPAM (Ham)**.")
                st.balloons()
