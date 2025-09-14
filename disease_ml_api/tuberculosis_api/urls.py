from django.urls import path
from .views import TuberculosisPredictView
   
urlpatterns = [
    path("tuberculosis/", TuberculosisPredictView.as_view(), name="tuberculosis_predict"),
]
             