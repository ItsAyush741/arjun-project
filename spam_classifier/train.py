# ===============================
# IMPORT LIBRARIES
# ===============================
import pandas as pd
import re
import string
import nltk
import pickle

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import resample

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# ===============================
# DOWNLOAD STOPWORDS
# ===============================
nltk.download('stopwords')

# ===============================
# LOAD DATASET
# ===============================
import os

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

df['message'] = df['message'].apply(clean_text)

# ===============================
# TRAIN TEST SPLIT
# ===============================
# Split BEFORE upsampling to prevent Data Leakage!
X_train_raw, X_test_raw, y_train_raw, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ===============================
# HANDLE CLASS IMBALANCE (TRAINING DATA ONLY)
# ===============================
train_df = pd.DataFrame({'message': X_train_raw, 'label': y_train_raw})
df_majority = train_df[train_df.label == 0]
df_minority = train_df[train_df.label == 1]

df_minority_upsampled = resample(
    df_minority,
    replace=True,
    n_samples=len(df_majority),
    random_state=42
)

train_df_up = pd.concat([df_majority, df_minority_upsampled])
X_train_raw = train_df_up['message']
y_train = train_df_up['label']

print("\nBalanced Class Distribution (Train Set Only):\n")
print(y_train.value_counts())

# ===============================
# FEATURE ENGINEERING & TF-IDF
# ===============================
# Fit vectorizer on training data ONLY
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train = vectorizer.fit_transform(X_train_raw)
X_test = vectorizer.transform(X_test_raw) # Only transform test data

# ===============================
# HYPERPARAMETER TUNING & MULTIPLE MODELS
# ===============================
print("\nHyperparameter Tuning (Logistic Regression with 10-Fold CV)\n")

# Using Logistic Regression which works best with TF-IDF and is robust
param_grid = {
    'C': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(
    LogisticRegression(max_iter=1000, class_weight='balanced'),
    param_grid,
    cv=10,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:", grid.best_params_)

# ===============================
# FINAL MODEL (USE BEST)
# ===============================
final_model = grid.best_estimator_

y_pred = final_model.predict(X_test)

print("\nFINAL MODEL PERFORMANCE\n")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ===============================
# SAVE MODEL
# ===============================
pickle.dump(final_model, open("spam_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully!")

# ===============================
# PREDICTION FUNCTION
# ===============================
def predict_spam(text):

    text_cleaned = clean_text(text)
    text_vec = vectorizer.transform([text_cleaned])
    pred = final_model.predict(text_vec)

    return "Spam" if pred[0] == 1 else "Not Spam"


# ===============================
# TEST
# ===============================
print("\nTESTING\n")

print(predict_spam("URGENT! You won $5000 prize"))
print(predict_spam("Hey bro are we meeting tomorrow"))
print(predict_spam("Invest $100 and earn $5000 in 24 hours"))