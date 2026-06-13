from flask import Flask, render_template, request
import joblib

from utils.preprocessing import extract_features

app = Flask(__name__)

# Load model

model = joblib.load("gender_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
encoder = joblib.load("label_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""

    if request.method == "POST":

        name = request.form["name"]

        features = extract_features(name)

        vector = vectorizer.transform([features])

        result = model.predict(vector)

        prediction = encoder.inverse_transform(result)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)