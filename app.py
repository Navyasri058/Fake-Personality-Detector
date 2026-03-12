from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient
import joblib
import pandas as pd
from collections import defaultdict
import random

# ---------------- Load dataset to get reasons ----------------
data = pd.read_csv("C:\\Users\\navya\\OneDrive\\Documents\\Personality_Detector_dataset.csv")

# Create dictionary: Label -> list of reasons
reason_dict = defaultdict(list)
for i in range(len(data)):
    label = data["Label"][i]
    reason = data["Reason"][i]
    reason_dict[label].append(reason)
    
# ---------------- Initialize Flask ----------------
app = Flask(__name__)

# ---------------- Load ML Components ----------------
vectorizer = joblib.load("vectorizer.joblib")
model = joblib.load("text_model.joblib")

# ---------------- MongoDB Connection ----------------
client = MongoClient("mongodb://localhost:27017/")
db = client["Personality_login_db"]
users_collection = db["Users"]

# ---------------- Routes ----------------

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        user = users_collection.find_one({"username": username})

        if user:
            if user["password"] == password:
                return redirect(url_for("dashboard"))
            else:
                return "Password incorrect"
        else:
            return "User does not exist"

    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]

        existing_user = users_collection.find_one({"username": username})

        if existing_user:
            return "Username already exists"

        users_collection.insert_one({
            "username": username,
            "email": email,
            "password": password
        })

        return redirect(url_for("login"))

    return render_template("signup.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/detect", methods=["GET", "POST"])
def detect():

    prediction = None
    reasons = None  # <-- initialize correctly
    percentage = None

    if request.method == "POST":
        text = request.form.get("statement")
        text_vector = vectorizer.transform([text])

        # Predict using ML model
        result = model.predict(text_vector)[0]
        prediction = result  # keep full label

        # Predict probability
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(text_vector)[0]
            class_index = list(model.classes_).index(result)
            percentage = round(proba[class_index] * 100, 2)

        # Get multiple reasons for that label
        if result in reason_dict:
            reasons = random.sample(reason_dict[result], min(3, len(reason_dict[result])))
        else:
            reasons = ["No reasons available"]

    # Pass reasons list to template
    return render_template(
        "detect.html",
        prediction=prediction,
        reasons=reasons,   # <-- fix here
        percentage=percentage
    )

# ---------------- Run Server ----------------
if __name__ == "__main__":
    app.run(debug=True)