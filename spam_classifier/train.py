# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import os
import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===============================
# DOWNLOAD STOPWORDS
# ===============================
nltk.download('stopwords', quiet=True)

# ===============================
# LOAD DATASET
# ===============================
dataset_files = [
    "dataset.csv",
    "../archive/emails.csv"
]

dfs = []
for file in dataset_files:
    if os.path.exists(file):
        try:
            temp_df = pd.read_csv(file, encoding="latin-1")
            
            if 'v1' in temp_df.columns and 'v2' in temp_df.columns:
                temp_df = temp_df[['v1', 'v2']]
                temp_df.columns = ['label', 'message']
                temp_df['label'] = temp_df['label'].map({'ham': 0, 'spam': 1})
                dfs.append(temp_df)
            elif 'text' in temp_df.columns and 'spam' in temp_df.columns:
                temp_df = temp_df[['spam', 'text']]
                temp_df.columns = ['label', 'message']
                dfs.append(temp_df)
            else:
                print(f"Warning: Unknown format in {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    else:
        print(f"Warning: File not found {file}")

if dfs:
    df = pd.concat(dfs, ignore_index=True)
    df.dropna(subset=['label', 'message'], inplace=True)
    df['label'] = df['label'].astype(int)
else:
    raise ValueError("No datasets loaded. Please check the dataset paths.")

print(f"\nTotal loaded samples: {len(df)}")
print("\nOriginal Class Distribution:\n")
print(df['label'].value_counts())

# ===============================
# TEXT PREPROCESSING
# ===============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

print("\nCleaning text data...")
df['message'] = df['message'].apply(clean_text)

# ===============================
# TRAIN TEST SPLIT
# ===============================
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ===============================
# FEATURE ENGINEERING & TF-IDF
# ===============================
print("\nVectorizing text data...")
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw)

# ===============================
# MULTIPLE MODELS TRAINING
# ===============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
}

trained_models = {}

print("\nTraining and Evaluating Models...\n")
for name, model in models.items():
    print(f"--- {name} ---")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}\n")
    
    trained_models[name] = model

# ===============================
# SAVE MODELS
# ===============================
pickle.dump(trained_models["Logistic Regression"], open("model_lr.pkl", "wb"))
pickle.dump(trained_models["Naive Bayes"], open("model_nb.pkl", "wb"))
pickle.dump(trained_models["Random Forest"], open("model_rf.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Models and vectorizer saved successfully!\n")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_spam(text):
    text_cleaned = clean_text(text)
    text_vec = vectorizer.transform([text_cleaned])
    
    print(f"\n--- Analyzing Text: '{text}' ---")
    
    best_model = None
    highest_confidence = -1
    
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(text_vec)[0]
            spam_prob = probs[1]
            pred = model.predict(text_vec)[0]
            label = "Spam" if pred == 1 else "Not Spam"
            confidence = max(probs)
            
            print(f"{name+':':<20} {label:<10} (Spam Probability: {spam_prob:>7.2%})")
            
            if confidence > highest_confidence:
                highest_confidence = confidence
                best_model = name
        else:
            pred = model.predict(text_vec)[0]
            label = "Spam" if pred == 1 else "Not Spam"
            print(f"{name+':':<20} {label:<10}")
            
    if best_model:
        print(f"💡 Preferred Model for this text: {best_model} (Highest Confidence: {highest_confidence:.2%})")

# ===============================
# TEST
# ===============================
predict_spam("URGENT! You won $5000 prize")
predict_spam("Hey bro are we meeting tomorrow")
predict_spam("Invest $100 and earn $5000 in 24 hours")