from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import joblib
import os
import numpy as np

# Load model once when server starts
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "covid19_pipeline.joblib")
model = joblib.load(MODEL_PATH)

# List of symptoms (order matters – same as training dataset)  
SYMPTOMS = [   
    "Age","Sex","Fever","Cough","Shortness_of_Breath","Loss_of_Taste_Smell",
    "Fatigue","Sore_Throat","Headache","Diarrhea","Comorbidity","Vaccinated"
]

class Covid19PredictView(APIView):
    def post(self, request):
        try:
            # Get symptom data from JSON request
            data = request.data
            features = [int(data.get(symptom, 0)) for symptom in SYMPTOMS]

            # Reshape for model
            features = np.array(features).reshape(1, -1)

            # Predict
            prediction = model.predict(features)[0]
            proba = model.predict_proba(features)[0][1]  

            return Response({
                "prediction": int(prediction),
                "probability": round(float(proba), 2)
            })
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
