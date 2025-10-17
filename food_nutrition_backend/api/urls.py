from django.urls import path
from . import views

urlpatterns = [
    path('analyze/', views.analyze_food, name='analyze_food'),
    path('health/', views.health_check, name='health_check'),
]