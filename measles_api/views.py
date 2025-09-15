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

class MeaslesPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # If single dict, wrap in a list
            if isinstance(data, dict):
                data = [data]

            # Build DataFrame in correct column order
            df = pd.DataFrame(data)
            for col in SYMPTOMS:
                if col not in df.columns:
                    df[col] = 0  # default missing fields

            df = df[SYMPTOMS]

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
   

       