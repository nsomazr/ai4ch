from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "rice"  

urlpatterns = [
    path('predict/', view=views.RicePredictImageView.as_view(), name='predict'),
    # path('detect/', view=views.RiceDetectImageAPI.as_view(), name='detect'),
    path('image-rice-desease-classifier',view=views.image_rice_classifier, name='image-rice-classifier'),
    path('image-rice-desease-detector',view=views.image_rice_detect, name='image-rice-detector'),
    path('video-rice-desease-detector',view=views.video_rice_detect, name='video-rice-detector'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
