from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "cassava"  

urlpatterns = [
   path('predict/', view=views.CassavaPredictImageView.as_view(), name='predict'),
#     path('detect/', view=views.CassavaDetectImageAPI.as_view(), name='detect'),
    path('image-cassava-desease-classifier',view=views.image_cassava_classifier, name='image-cassava-classifier'),
    path('image-cassava-desease-detector',view=views.image_cassava_detect, name='image-cassava-detector'),
    path('video-cassava-desease-detector',view=views.video_cassava_detect, name='video-cassava-detector'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
