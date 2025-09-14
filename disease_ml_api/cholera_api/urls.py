from django.urls import path
from .views import CholeraPredictView

urlpatterns = [
    path("cholera/", CholeraPredictView.as_view(), name="cholera_predict"),
]
           