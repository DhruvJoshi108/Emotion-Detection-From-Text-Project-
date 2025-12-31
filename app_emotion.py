from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

with open("model_emotion.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer_emotion.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    user_text = ""

    if request.method == "POST":
        user_text = request.form["text"]
        cleaned = clean_text(user_text)
        vec = vectorizer.transform([cleaned])
        emotion = model.predict(vec)[0]

    return render_template(
        "index.html",
        emotion=emotion,
        user_text=user_text
    )

if __name__ == "__main__":
    app.run(debug=True)
