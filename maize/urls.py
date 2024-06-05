from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static


app_name = "maize"  

urlpatterns = [
    path('predict/', view=views.PredictImageView.as_view(), name='predict'),
    path('detect/', view=views.MaizeDetectAPI.as_view(), name='detect'),
    path('maize-desease-classifier',view=views.maize_classifier, name='maize-classifier'),
    path('maize-desease-detector',view=views.maize_detect, name='maize-detector'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

