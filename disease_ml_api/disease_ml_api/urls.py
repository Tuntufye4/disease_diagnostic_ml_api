from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/", include("malaria_api.urls")),
    path("api/", include("cholera_api.urls")),
    path("api/", include("tuberculosis_api.urls")),
    path("api/", include("covid_19_api.urls")),
    path("api/", include("measles_api.urls")), 
    path("api/", include("anthrax_api.urls")),

]
