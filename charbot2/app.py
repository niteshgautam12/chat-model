from flask import Flask, render_template, request, jsonify
import pickle
import random

app = Flask(__name__)

with open("chatbot_model.pkl", "rb") as f:
    model, vectorizer, intents = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/chat', methods=['POST'])
def chat():
    user_msg = request.json['message']
    vec = vectorizer.transform([user_msg.lower()])
    tag = model.predict(vec)[0]

    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return jsonify({"reply": random.choice(intent["responses"])})
    
    return jsonify({"reply": "Sorry, I didn’t get that. Can you rephrase?"})

if __name__ == '__main__':
    app.run(debug=True)
