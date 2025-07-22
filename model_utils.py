import joblib
import json
import numpy as np

clf = joblib.load("clf.pkl")
label_encoder = joblib.load("label_encoder.pkl")

with open("label_to_therapy.json", "r") as f:
    label_to_therapy = json.load(f)

def predict_user_query(query: str) -> dict:
    prediction = clf.predict([query])[0]
    category = label_encoder.inverse_transform([prediction])[0]
    therapies = label_to_therapy.get(category, ["No suggestion found"])
    
    return {
        "Predicted Category": category,
        "Suggested Therapies": therapies
    }
