import streamlit as st
import pickle
import string
import nltk
import os

st.set_page_config(page_title="Spam Classifier", page_icon="📧")

@st.cache_resource
def get_stop_words():
    import nltk
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    from nltk.corpus import stopwords
    return set(stopwords.words('english'))

stop_words = get_stop_words()

st.title("📧 SMS & Email Spam Classifier")
st.write("Enter the message below to check whether it's Spam or Not Spam (Ham).")

# Model Selection UI
st.subheader("⚙️ Model Settings")
model_choice = st.selectbox(
    "Choose a Machine Learning Model:",
    ["Logistic Regression", "Naive Bayes", "Random Forest"]
)

model_files = {
    "Logistic Regression": "model_lr.pkl",
    "Naive Bayes": "model_nb.pkl",
    "Random Forest": "model_rf.pkl"
}

# Load model and vectorizer safely
@st.cache_resource
def load_models():
    models = {}
    for name, path in model_files.items():
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[name] = pickle.load(f)
        else:
            return None, None # Missing files handled outside
            
    if os.path.exists("vectorizer.pkl"):
        with open("vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
    else:
        return None, None
        
    return models, vectorizer

models_dict, vectorizer = load_models()

if models_dict is None or vectorizer is None:
    st.error("⚠️ Model files not found. Please run `train.py` first to generate the models.")
    st.stop()

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.markdown("---")
input_message = st.text_area("Enter your message:", height=150)

if st.button("Predict"):
    if input_message.strip() == "":
        st.warning("Please enter a message to classify.")
    else:
        with st.spinner(f"Analyzing message using {model_choice}..."):
            text_cleaned = clean_text(input_message)
            text_vec = vectorizer.transform([text_cleaned])
            
            selected_model = models_dict[model_choice]
            prediction = selected_model.predict(text_vec)[0]
            
            # Calculate and display spam probability
            if hasattr(selected_model, "predict_proba"):
                probabilities = selected_model.predict_proba(text_vec)[0]
                spam_prob = probabilities[1] * 100
                st.info(f"📊 Spam Probability: **{spam_prob:.2f}%**")
            
            if prediction == 1:
                st.error("🚨 This message is classified as **SPAM**!")
            else:
                st.success("✅ This message is **NOT SPAM (Ham)**.")
                
