# IMDB Sentiment Analysis using Machine Learning

A complete end-to-end Machine Learning project that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative** using Natural Language Processing (NLP).

---

## Project Overview

This project builds a sentiment classification system using the IMDB 50K movie reviews dataset. It includes:

- Data preprocessing and cleaning
- Feature extraction using TF-IDF
- Model training using Logistic Regression
- Model evaluation and comparison
- Deployment using Streamlit

The system is designed with a **pipeline-based architecture** to ensure consistency between training and inference.

---

## Project Structure
Sentiment-Analysis/
│
├── data/
│ └── IMDB Dataset.csv
│
├── model/
│ └── pipeline.pkl
│
├── notebook/
│ └── analysis.ipynb
│
├── utils.py
├── train.py
├── app.py
│
└── README.md

---

## Features

- Text preprocessing (lowercasing, HTML removal, regex cleaning)
- TF-IDF vectorization (with unigrams and bigrams)
- Logistic Regression model
- Pipeline integration (prevents preprocessing mismatch)
- Streamlit web application
- Real-time sentiment prediction with confidence score

---

## Machine Learning Approach

### 1. Data Preprocessing
- Removed HTML tags
- Removed non-alphabetic characters
- Normalized whitespace
- Converted text to lowercase

### 2. Feature Extraction
- TF-IDF Vectorizer
- `max_features = 5000`
- `ngram_range = (1,2)`

### 3. Model
- Logistic Regression
- Optimized with:
  - `max_iter = 1000`
  - `C = 2`
  - `solver = liblinear`

### 4. Pipeline
A Scikit-learn Pipeline is used to combine:
- TF-IDF Vectorizer
- Logistic Regression Model

This ensures:
- Consistent preprocessing
- No training/inference mismatch
- Cleaner deployment

---

## Model Performance

| Model               | Accuracy |
|--------------------|---------|
| Logistic Regression | ~89%    |
| Naive Bayes         | ~85%    |

Logistic Regression performs better due to its ability to handle high-dimensional sparse data effectively.

---

## Streamlit Application

The web app allows users to:

- Enter a movie review
- Get sentiment prediction (Positive/Negative)
- View confidence score
- See probability distribution

---

## How to Run the Project

### 1. Clone Repository
git clone https://github.com/Habibshahx/sentiment-analysis.git
cd sentiment-analysis

### 2. Install Dependencies

### 3. Train Model

### 4. Run Streamlit App

---

## Example Inputs

- "This movie was absolutely fantastic, I loved it!"
- "Worst film ever, complete waste of time."
- "It was okay, not great but not terrible either."

---

## Limitations

- Struggles with sarcasm and irony
- Limited understanding of context (TF-IDF based)
- Short inputs may produce less reliable predictions

---

## Future Improvements

- Implement Transformer-based models (e.g., BERT)
- Add hyperparameter tuning (GridSearchCV)
- Improve handling of negation and sarcasm
- Enhance UI with interpretability (feature importance)
- Deploy on cloud (Streamlit Cloud / Render)

---

## Tech Stack

- Python
- Scikit-learn
- Pandas
- NLTK (optional)
- Streamlit

---

## Key Learning Outcomes

- End-to-end ML pipeline development
- NLP preprocessing techniques
- Model evaluation and comparison
- Debugging real-world ML issues (pipeline mismatch)
- Deployment of ML models as web apps

---

## Author

**Habib Shah**
