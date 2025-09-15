from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "anthrax_pipeline.joblib")
model = joblib.load(MODEL_PATH)

NUMERIC_COLS = ["Age","Fever","Chills","Fatigue","Muscle_Aches","Sore_Throat",
                "Skin_Lesion","Shortness_of_Breath","Nausea_Vomiting","Confusion"]

FEATURES = ["Age","Sex","Fever","Chills","Fatigue","Muscle_Aches","Sore_Throat",
            "Skin_Lesion","Shortness_of_Breath","Nausea_Vomiting","Confusion","Comorbidity"]

class AnthraxPredictView(APIView):
    def post(self, request):
        try:
            data = request.data

            # Support single patient dict
            if isinstance(data, dict):
                data = [data]

            df = pd.DataFrame(data)

            # Convert numeric columns only
            for col in NUMERIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

            # Leave categorical columns as-is (Sex, Comorbidity)
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
