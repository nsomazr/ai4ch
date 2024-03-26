from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "ai4chapp"  

urlpatterns = [
    path("",view=views.home, name='home' ),
    path("about/",view=views.about, name='about' ),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)