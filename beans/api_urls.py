from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "beans-api"  

urlpatterns = [
    # path('beans/predict/', view=views.PredictImageView.as_view(), name='beans-api-predict'),
    path('', view=views.BeansDetectAPI.as_view(), name='beans-api-detect'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)