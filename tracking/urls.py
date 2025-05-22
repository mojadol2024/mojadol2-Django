from django.urls import path
from .views import TrackingView

urlpatterns = [
    path('tracking', TrackingView.as_view(), name='tracking'),
]
