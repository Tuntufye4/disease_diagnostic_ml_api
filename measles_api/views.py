from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once when server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "measles_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# List of symptoms (order matters – same as training dataset)
SYMPTOMS = [
    "Age", "Sex", "Fever", "Cough", "Runny_Nose", "Red_Eyes", "Koplik_Spots",
    "Rash", "Fatigue", "Muscle_Aches", "Sore_Throat", "Vomiting",
    "Comorbidity", "Vaccinated"
]

# Normalization helper
def normalize_value(col, val):
    if pd.isna(val):
        return 0
    if isinstance(val, str):
        v = val.strip().lower()
        if v in ["yes", "y", "true", "1"]:
            return 1
        if v in ["no", "n", "false", "0"]:
            return 0
        if col == "Sex":
            if v in ["male", "m"]:
                return 1
            if v in ["female", "f"]:
                return 0
    try:
        return int(val)
    except ValueError:
        return 0

class MeaslesPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # If single dict, wrap in a list
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

            # Add missing columns with default 0
            for col in SYMPTOMS:
                if col not in df.columns:
                    df[col] = 0

            # Reorder + normalize values
            df = df[SYMPTOMS].apply(lambda col: col.map(lambda x: normalize_value(col.name, x)))

            # Predict
            prediction = model.predict(df)
            proba = model.predict_proba(df)[:, 1]

            results = [
                {"prediction": int(p), "probability": round(float(pr), 2)}
                for p, pr in zip(prediction, proba)
            ]

            return Response(results)

        except Exception as e:  
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
