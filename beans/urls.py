from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "beans"  

urlpatterns = [
    path('predict/', view=views.PredictImageView.as_view(), name='predict'),
    # path('detect/', view=views.BeansDetectAPI.as_view(), name='detect'),
    path('image-beans-desease-classifier',view=views.image_beans_classifier, name='image-beans-classifier'),
    path('image-beans-desease-detector',view=views.image_beans_detect, name='image-beans-detector'),
    path('video-beans-desease-detector',view=views.video_beans_detect, name='video-beans-detector'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

