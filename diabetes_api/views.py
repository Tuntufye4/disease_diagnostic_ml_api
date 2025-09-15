from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "diabetes_pipeline.joblib")
model = joblib.load(MODEL_PATH)

SYMPTOMS = [
    "Age","Sex","Frequent_Urination","Excessive_Thirst","Unexplained_Weight_Loss",
    "Extreme_Hunger","Fatigue","Blurred_Vision","Slow_Healing_Sores",
    "Tingling_Numbness","Family_History","Obesity"
]

class DiabetesPredictView(APIView):
    def post(self, request):
        try:
            data = request.data
            features_df = pd.DataFrame([{sym: data.get(sym, 0) for sym in SYMPTOMS}])
            prediction = model.predict(features_df)[0]
            proba = model.predict_proba(features_df)[0][1]
            return Response({
                "prediction": int(prediction),    
                "probability": round(float(proba), 2)
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
