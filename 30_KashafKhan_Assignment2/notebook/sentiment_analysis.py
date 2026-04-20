# ==============================
# SENTIMENT ANALYSIS - F1 MOVIE
# Assignment 2
# ==============================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import seaborn as sns

# ==============================
# 1. LOAD DATASET
# ==============================

df = pd.read_csv("data/f1_reviews.csv")

print("Dataset Loaded")
print(df.head())

print("\nClass Distribution:")
print(df['sentiment'].value_counts())

# ==============================
# 2. TRAIN-TEST SPLIT (80/20)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    df['review'],
    df['sentiment'],
    test_size=0.2,
    random_state=42
)

# ==============================
# 3. TEXT VECTORIZATION
# ==============================

vectorizer = CountVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ==============================
# 4. NAIVE BAYES MODEL
# ==============================

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

nb_pred = nb_model.predict(X_test_vec)

print("\n==============================")
print("NAIVE BAYES RESULTS")
print("==============================")
print(classification_report(y_test, nb_pred))

print("Accuracy:", accuracy_score(y_test, nb_pred))

# ==============================
# 5. LOGISTIC REGRESSION MODEL
# ==============================

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)

lr_pred = lr_model.predict(X_test_vec)

print("\n==============================")
print("LOGISTIC REGRESSION RESULTS")
print("==============================")
print(classification_report(y_test, lr_pred))

print("Accuracy:", accuracy_score(y_test, lr_pred))

# ==============================
# 6. CONFUSION MATRIX (BEST MODEL)
# ==============================

cm = confusion_matrix(y_test, lr_pred, labels=["positive", "negative", "neutral"])

plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["positive", "negative", "neutral"],
            yticklabels=["positive", "negative", "neutral"])

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("outputs/confusion_matrix.png")
plt.show()

print("\nConfusion matrix saved in outputs folder")

# ==============================
# 7. MODEL COMPARISON
# ==============================

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")

print("Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_pred))
