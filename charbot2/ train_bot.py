import json
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

with open("intents.json") as f:
    data = json.load(f)

corpus = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        corpus.append(pattern.lower())
        tags.append(intent["tag"])


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

model = LogisticRegression()
model.fit(X, tags)

with open("chatbot_model.pkl", "wb") as f:
    pickle.dump((model, vectorizer, data), f)

print("✅ Model trained and saved.")
