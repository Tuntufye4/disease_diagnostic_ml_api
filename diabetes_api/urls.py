from django.urls import path
from .views import DiabetesPredictView
   
urlpatterns = [
    path("diabetes/", DiabetesPredictView.as_view(), name="diabetes_predict"),
]
             