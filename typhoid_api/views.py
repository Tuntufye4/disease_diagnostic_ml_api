from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import pandas as pd

# Load model once when server starts   
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "typhoid_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# List of symptoms (order matters – same as training dataset)  
SYMPTOMS = [
    "Age","Sex","Fever","Headache","Fatigue","Abdominal_Pain","Diarrhea","Constipation",
    "Loss_of_Appetite","Rash","Nausea_Vomiting","Muscle_Aches","Comorbidity","Vaccinated"
]

class TyphoidPredictView(APIView):
    def post(self, request):
        try:
            # Get symptom data from JSON request
            data = request.data

            # Create DataFrame (pipeline expects tabular format)
            features_df = pd.DataFrame([{symptom: data.get(symptom, 0) for symptom in SYMPTOMS}])

            # Predict
            prediction = model.predict(features_df)[0]
            proba = model.predict_proba(features_df)[0][1]  

            return Response({
                "prediction": int(prediction),
                "probability": round(float(proba), 2)
            })
        except Exception as e:   
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
