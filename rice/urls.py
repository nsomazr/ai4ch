from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "rice"  

urlpatterns = [
    path('rice-desease-classifier',view=views.classifier, name='rice-classifier'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
   urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)
