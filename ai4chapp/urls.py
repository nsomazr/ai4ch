from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path("",view=views.index, name='index' ),
    path("about/",view=views.about, name='about' ),
    path("news/",view=views.news, name='news' ),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)