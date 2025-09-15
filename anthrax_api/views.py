import os
import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

# Load model + feature list
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anthrax_pipeline.joblib")
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]         # <-- This is the actual trained pipeline
TRAINED_FEATURES = model_bundle["features"]  # <-- Feature order

class AnthraxPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # Wrap single dict in a list to handle batch prediction
            if isinstance(data, dict):
                data = [data]

            # Build DataFrame in correct order
            df = pd.DataFrame(data)

            # Add missing columns with 0
            for col in TRAINED_FEATURES:
                if col not in df.columns:
                    df[col] = 0

            df = df[TRAINED_FEATURES]  # ensure correct column order

            # Predict using the actual model
            prediction = model.predict(df)
            proba = model.predict_proba(df)[:, 1]

            results = [
                {"prediction": int(p), "probability": round(float(pr), 2)}
                for p, pr in zip(prediction, proba)
            ]

            return Response(results)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)  
   