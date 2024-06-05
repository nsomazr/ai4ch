from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "rice"  

urlpatterns = [
    path('predict/', view=views.PredictImageView.as_view(), name='predict'),
    path('rice-desease-classifier',view=views.classifier, name='rice-classifier'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
