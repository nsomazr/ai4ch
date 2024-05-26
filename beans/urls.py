from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "beans"  

urlpatterns = [
    path('predict/', view=views.PredictImageView.as_view(), name='predict'),
    path('beans-desease-classifier',view=views.classifier, name='beans-classifier'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
   urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
