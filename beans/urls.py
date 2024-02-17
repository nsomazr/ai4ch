from django.urls import path, include
from . import views

urlpatterns = [
    path('bean-desease-classifier',view=views.classifier, name='classifier'),
]