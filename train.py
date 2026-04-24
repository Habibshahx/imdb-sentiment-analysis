import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Import cleaning function from utils
from utils import clean_text

df = pd.read_csv("data/IMDB Dataset.csv")

df["clean_review"] = df["review"].apply(clean_text)

X = df["clean_review"]
y = df["sentiment"].map({"positive": 1, "negative": 0})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Pipeline (TF-IDF + Model)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )),
    ("model", LogisticRegression(
        max_iter=1000,
        C=2,
        solver="liblinear"
    ))
])

pipeline.fit(X_train, y_train)

# Evaluate Model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.4f}")

# Save Pipeline
with open("model/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model pipeline saved successfully!")