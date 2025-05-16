from django.urls import path
from .views import GenerateQuestionsView

urlpatterns = [
    path("generate", GenerateQuestionsView.as_view(), name="generate-questions"),
]
