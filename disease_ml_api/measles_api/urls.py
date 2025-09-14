from django.urls import path
from .views import MeaslesPredictView
   
urlpatterns = [
    path("measles/", MeaslesPredictView.as_view(), name="measles_predict"),
]
             