from django.urls import path
from .views import STTView
from .views import TTSView

urlpatterns = [
    path('stt', STTView.as_view(), name='stt'),
    path('tts', TTSView.as_view(), name='tts'),
]
