from django.urls import path
from .views import MalariaPredictView

urlpatterns = [
    path("malaria/", MalariaPredictView.as_view(), name="malaria_predict"),
]
