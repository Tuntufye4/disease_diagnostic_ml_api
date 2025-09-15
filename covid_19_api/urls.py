from django.urls import path
from .views import Covid19PredictView
   
urlpatterns = [
    path("covid19/", Covid19PredictView.as_view(), name="covid19_predict"),
]
             