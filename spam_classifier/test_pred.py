import pickle
import string
import nltk

try:
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

text = "Buy cheap medicines without prescription. Huge discounts available."
text_cleaned = clean_text(text)
text_vec = vectorizer.transform([text_cleaned])

try:
    probs = model.predict_proba(text_vec)[0]
    print("Probabilities:", probs)
except Exception as e:
    print("No proba mapping", e)
    
prediction = model.predict(text_vec)[0]
print(f"Prediction for '{text}': {'Spam' if prediction == 1 else 'Not Spam'}")
