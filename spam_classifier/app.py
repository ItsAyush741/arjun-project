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
st.write("The system will analyze your text using all available models and suggest the best prediction based on the highest confidence level.")

# Add a strictness threshold
st.markdown("### 🎛️ Spam Strictness")
spam_threshold = st.slider(
    "Spam Probability Threshold (%)", 
    min_value=10, max_value=99, value=50, step=1,
    help="Increase this value if normal messages are being accidentally marked as Spam. Decrease if spam messages are slipping through."
) / 100.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_files = {
    "Logistic Regression": os.path.join(BASE_DIR, "model_lr.pkl"),
    "Naive Bayes": os.path.join(BASE_DIR, "model_nb.pkl"),
    "Random Forest": os.path.join(BASE_DIR, "model_rf.pkl")
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
            
    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    if os.path.exists(vectorizer_path):
        with open(vectorizer_path, "rb") as f:
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
        with st.spinner("Analyzing message across all models..."):
            text_cleaned = clean_text(input_message)
            text_vec = vectorizer.transform([text_cleaned])
            
            st.markdown("### 📊 Model Comparison")
            
            best_model = None
            highest_confidence = -1
            best_prediction = None
            
            results_data = []

            for name, model in models_dict.items():
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(text_vec)[0]
                    spam_prob = probs[1]
                    
                    # Apply custom threshold
                    is_spam = spam_prob >= spam_threshold
                    confidence = spam_prob if is_spam else (1.0 - spam_prob)
                    
                    results_data.append({
                        "Model": name,
                        "Prediction": "🚨 Spam" if is_spam else "✅ Not Spam",
                        "Spam Probability": f"{spam_prob * 100:.2f}%",
                        "Confidence": f"{confidence * 100:.2f}%"
                    })
                    
                    if confidence > highest_confidence:
                        highest_confidence = confidence
                        best_model = name
                        best_prediction = "SPAM" if is_spam else "NOT SPAM"
                else:
                    pred = model.predict(text_vec)[0]
                    results_data.append({
                        "Model": name,
                        "Prediction": "🚨 Spam" if pred == 1 else "✅ Not Spam",
                        "Spam Probability": "N/A",
                        "Confidence": "N/A"
                    })
            
            st.table(results_data)
            
            st.success(f"💡 **Preferred Model for this text: {best_model}** (Highest Confidence: {highest_confidence * 100:.2f}%)")
            
            # Show the final verdict based on the preferred model
            st.markdown("### 🎯 Final Decision")
            if best_prediction == "SPAM":
                st.error(f"🚨 Based on the preferred model ({best_model}), this message is classified as **SPAM**!")
            else:
                st.success(f"✅ Based on the preferred model ({best_model}), this message is **NOT SPAM (Ham)**.")
                
