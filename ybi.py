# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Step 2: Load Dataset
df = pd.read_csv("Fake.csv")  # or merge with "True.csv" if combining datasets
df['label'] = 0  # Fake News -> 0
# if merging with real news:
# real = pd.read_csv("True.csv")
# real['label'] = 1
# df = pd.concat([df, real])

# Step 3: Preprocessing
df = df[['text', 'label']]  # using only text and label
df.dropna(inplace=True)

# Step 4: Train-Test Split
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Feature Extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Step 6: Train Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 7: Predictions
y_pred = model.predict(X_test_tfidf)

# Step 8: Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
