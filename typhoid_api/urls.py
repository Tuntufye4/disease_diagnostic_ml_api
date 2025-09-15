from django.urls import path
from .views import TyphoidPredictView
   
urlpatterns = [
    path("typhoid/", TyphoidPredictView.as_view(), name="typhoid_predict"),
]
             