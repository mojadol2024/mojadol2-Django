from django.contrib import admin
from django.urls import path, include


urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/speech/', include('speech.urls')),
    path('api/questions/', include('generate-questions.urls')),
    path('api/tracking/', include('tracking.urls')),
]