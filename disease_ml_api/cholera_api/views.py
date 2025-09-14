from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once when server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cholera_diagnostics_model.joblib")
model = joblib.load(MODEL_PATH)

# Feature order exactly as specified
FEATURES = [
    "watery_diarrhea",
    "vomiting",
    "leg_cramps",
    "dehydration",
    "rapid_heartbeat",
    "low_blood_pressure",
    "dry_mouth",
    "thirst",
    "restlessness",
    "sunken_eyes",
    "muscle_pain",
    "fatigue"
]

class CholeraPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # Support single patient dict
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

            # Ensure all features exist and convert to numeric
            for col in FEATURES:
                if col not in df.columns:
                    df[col] = 0  # default missing features to 0
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Reorder columns to match FEATURES
            df = df[FEATURES]

            # Predict
            prediction = model.predict(df)
            proba = model.predict_proba(df)[:, 1]

            results = [{"prediction": int(p), "probability": round(float(pr), 2)} for p, pr in zip(prediction, proba)]

            return Response(results)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
