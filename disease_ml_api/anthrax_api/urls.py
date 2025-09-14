from django.urls import path
from .views import AnthraxPredictView

urlpatterns = [
    path("anthrax/", AnthraxPredictView.as_view(), name="anthrax_predict"),
]
        