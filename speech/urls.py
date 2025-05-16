from django.urls import path
from .views import STTView

urlpatterns = [
    path('stt', STTView.as_view(), name='stt'),
]
