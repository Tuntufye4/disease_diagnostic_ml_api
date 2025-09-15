from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tb_diagnostics_pipeline.joblib")
model = joblib.load(MODEL_PATH)

NUMERIC_COLS = ["Age","Cough_Duration_weeks","Fever","Night_Sweats",
                "Weight_Loss","Fatigue","Chest_Pain","Hemoptysis"]

FEATURES = ["Age","Sex","Cough_Duration_weeks","Fever","Night_Sweats",
            "Weight_Loss","Fatigue","Chest_Pain","Hemoptysis","HIV_Status","Smoker"]

class TuberculosisPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # Wrap single dict into list
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

            # Convert numeric columns only
            for col in NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Keep categorical columns as-is (strings)
            prediction = model.predict(df)
            proba = model.predict_proba(df)[:, 1]

            results = []
            for p, pr in zip(prediction, proba):
                results.append({
                    "prediction": int(p),
                    "probability": round(float(pr), 2)
                })

            return Response(results)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
