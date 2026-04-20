Assignment Title
F1 Movie Sentiment Analysis

1) Objective
The objective of this project is to analyze audience sentiment for the movie “F1 the Movie” using machine learning techniques.

The system classifies movie reviews into:
Positive 
Neutral 
Negative 

This helps understand public opinion and overall reception of the movie.

2) Problem Statement

Movie reviews are unstructured text data. This project aims to:
Convert text reviews into numerical features
Train machine learning models
Predict sentiment categories
Compare model performance

3) Models Used

The following machine learning models are implemented:

Naive Bayes
Fast and efficient probabilistic model
Works well with text classification
Logistic Regression
Strong linear classifier
Performs better with balanced datasets

4) Dataset
Total Reviews: 100
Source: Manually created dataset (simulated movie reviews)
Format:
review → text content of movie review
sentiment → label (positive / neutral / negative)

Dataset is split into:

80% Training data
20% Testing data

5) Methodology
1. Data Preprocessing
Converted text to lowercase
Removed punctuation and special characters
Cleaned raw review text
2. Feature Extraction
Used CountVectorizer / Bag of Words
Converted text into numerical vectors
3. Model Training
Trained Naive Bayes model
Trained Logistic Regression model
4. Evaluation

Models are evaluated using:

Accuracy
Precision
Recall
F1-score

6) Results

Logistic Regression performed better overall in terms of accuracy and balanced predictions.
Naive Bayes performed well but struggled with complex sentence structures.
Due to small dataset size (100 samples), performance is moderate.

7) Visualization
Confusion Matrix generated for model evaluation
Shows correct vs incorrect predictions
Helps compare model performance visually

8) Conclusion
Logistic Regression is the better-performing model for this dataset.

However, model performance can be further improved by:
Increasing dataset size
Using TF-IDF instead of simple vectorization
Hyperparameter tuning

9) How to Run
pip install -r requirements.txt
python sentiment_analysis.py

10) Requirements
pandas
numpy
scikit-learn
matplotlib
seaborn

11) Project Structure

30_KashafKhan_Assignment2/
│
├── data/
│   └── f1_reviews.csv
│
├── sentiment_analysis.py
│
├── outputs/
│   └── confusion_matrix.png
│
├── reports/
│   └── report.pdf
│
├── README.md
└── requirements.txt

12) Student Details

Kashaf Khan
Roll No: 30
TE (AI&DS)