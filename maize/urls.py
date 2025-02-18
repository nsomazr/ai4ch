from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static


app_name = "maize"  

urlpatterns = [
    path('predict/', view=views.MaizePredictImageView.as_view(), name='predict'),
    # path('detect/', view=views.MaizeDetectAPI.as_view(), name='detect'),
    path('image-maize-desease-classifier',view=views.image_maize_classifier, name='image-maize-classifier'),
    path('image-maize-desease-detector',view=views.image_maize_detect, name='image-maize-detector'),
    path('video-maize-desease-detector',view=views.video_maize_detect, name='video-maize-detector'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

