# 🧠 Mental Health Assessment API

This project is an AI-powered mental health assessment tool built with **FastAPI** and **Sentence Transformers**. It predicts the most probable mental health condition from a user's text input and suggests relevant therapies based on a trained machine learning model.

---

## 🚀 Features

- Predicts mental health conditions from open-ended user input
- Suggests suitable therapy methods
- Built with:
  - [`sentence-transformers`](https://www.sbert.net/) for semantic embeddings
  - `scikit-learn` for model training
  - `FastAPI` for serving the prediction model via an API

---

## 🧩 Model Details

- **Embedding Model**: `all-MiniLM-L6-v2` (384-dimensional sentence vectors)
- **Classifier**: Logistic Regression
- **Training Input**: Pre-cleaned user questions related to therapy and mental health
- **Labels**: Conditions like PTSD, Depression, Anxiety, Grief, etc.

---

## 📦 Project Structure

.
├── main.py # FastAPI app
├── clf.pkl # Trained Logistic Regression model
├── label_encoder.pkl # LabelEncoder for inverse label transformation
├── label_to_therapy.json # Mapping from condition to therapies
├── requirements.txt # All required dependencies
└── README.md # You're here!


---

## 🔧 Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/mental-health-assessment-api.git
cd mental-health-assessment-api
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the FastAPI server
```bash
uvicorn main:app --reload
```

### 🧪 Example API Request
POST /predict
Request JSON:
```json
{
  "text": "I constantly feel overwhelmed and can't sleep at night. I have trouble concentrating."
}
```
Response JSON:

```json
{
  "Predicted Condition": "Anxiety",
  "Suggested Therapies": [
    "Cognitive Behavioral Therapy (CBT)",
    "Mindfulness-Based Stress Reduction",
    "Exposure Therapy"
  ]
}
```

### ✅ Requirements
- Python 3.8+

- fastapi

- uvicorn

- scikit-learn

- sentence-transformers

- joblib

