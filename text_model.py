import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from collections import defaultdict
import random


# -------------------------------
# 1 Load Dataset
# -------------------------------

data = pd.read_csv("C:\\Users\\navya\\OneDrive\\Documents\\Personality_Detector_dataset.csv")
# Use only required columns
texts = data["Text"]
labels = data["Label"]
# Create a dictionary for Label → Reason
reason_dict = defaultdict(list)

for i in range(len(data)):
    label = data["Label"][i]
    reason = data["Reason"][i]
    reason_dict[label].append(reason)


# -------------------------------
# 2 Split Dataset
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# -------------------------------
# 3 Convert Text → Numbers
# -------------------------------

vectorizer = TfidfVectorizer(stop_words="english")

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 4 Train Model
# -------------------------------

model = MultinomialNB()

model.fit(X_train_vec, y_train)

# -------------------------------
# 5 Test Model
# -------------------------------

predictions = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:", accuracy * 100, "%\n")

print("Classification Report:\n")
print(classification_report(y_test, predictions))

# -------------------------------
# 6 Save Model
# -------------------------------

joblib.dump(model, "text_model.joblib")
joblib.dump(vectorizer, "vectorizer.joblib")

print("\nModel saved successfully!")

# -------------------------------
# 7 Test Prediction
# -------------------------------

while True:

    user_input = input("\nEnter a sentence (or type exit): ")

    if user_input.lower() == "exit":
        break

    text_vec = vectorizer.transform([user_input])

    prediction = model.predict(text_vec)[0]

    print("\nPrediction:", prediction)

    # Show reason
    if prediction in reason_dict:
        reason = random.choice(reason_dict[prediction])
        print("Reason:", reason)
    else:
         print("Reason not available")