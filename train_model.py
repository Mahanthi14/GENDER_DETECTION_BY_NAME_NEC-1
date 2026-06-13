import pandas as pd
import joblib

from sklearn.feature_extraction import DictVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

from utils.preprocessing import extract_features

# Load dataset

df = pd.read_csv("datasets/gender_dataset.csv")

# Features

X = [extract_features(name) for name in df["Name"]]
y = df["Gender"]

# Vectorization

vectorizer = DictVectorizer(sparse=False)
X = vectorizer.fit_transform(X)

# Encode labels

encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Train model

model = GaussianNB()
model.fit(X, y)

# Save files

joblib.dump(model, "gender_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(encoder, "label_encoder.pkl")

print("Model Trained Successfully")