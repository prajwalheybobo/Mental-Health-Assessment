from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import joblib
import json

app = FastAPI()

clf = joblib.load("clf.pkl")
encoder = joblib.load("label_encoder.pkl")

with open("label_to_therapy.json", "r") as f:
    label_to_therapy = json.load(f)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

class UserQuery(BaseModel):
    text: str

@app.get("/")
def read_root():
    return {"message": "Mental Health Classifier API is running."}

@app.post("/predict")
def predict_condition(query: UserQuery):
    try:
        vec = embedder.encode([query.text])
        pred = clf.predict(vec)[0]
        label = encoder.inverse_transform([pred])[0]
        therapy = label_to_therapy.get(label, [])

        return {
            "Predicted Condition": label,
            "Suggested Therapies": therapy
        }
    except Exception as e:
        return {"error": str(e)}
#uvicorn main:app --reload