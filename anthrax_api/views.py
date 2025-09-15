import os
import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Load model + feature list
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anthrax_pipeline.joblib")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
TRAINED_FEATURES = model_bundle["features"]

# Define which features are categorical
CATEGORICAL_FEATURES = ["Sex", "Comorbidity"]  # update according to your dataset

class AnthraxPredictView(APIView):
    def post(self, request):
        try:
            data = request.data
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

            # Add missing columns
            for col in TRAINED_FEATURES:
                if col not in df.columns:
                    df[col] = 0

            # Ensure correct order
            df = df[TRAINED_FEATURES]

            # Force categorical columns to string and numeric to float/int
            for col in df.columns:
                if col in CATEGORICAL_FEATURES:
                    df[col] = df[col].astype(str)
                else:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

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
