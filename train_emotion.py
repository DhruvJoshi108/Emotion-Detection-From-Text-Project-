import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import pickle

# ----------------------------
# Load dataset
# ----------------------------
df = pd.read_csv("D:\\Data Set for Project\\emotion_dataset.csv")
print("RUNNING THIS TRAIN_EMOTION.PY FILE")



# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

df["text"] = df["text"].apply(clean_text)

# ----------------------------
# Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"],
    df["emotion"],
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Vectorization
# ----------------------------
keep_words = {"not", "no", "never"}
custom_stop_words = list(ENGLISH_STOP_WORDS.difference(keep_words))
vectorizer = TfidfVectorizer(stop_words=custom_stop_words, max_features=30000)
vectorizer.fit(X_train)


X_train_vec = vectorizer.fit_transform(X_train)

# ----------------------------
# Model training
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# ----------------------------
# Save model and vectorizer
# ----------------------------
with open("model_emotion.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer_emotion.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Training complete. model.pkl and vectorizer.pkl saved.")

