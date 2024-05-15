from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('detection/',view=views.classifier, name='detect-emotion'),
    path('',view=views.login, name='login'),
    path('logout',view=views.logout, name='logout'),
    path('register',view=views.register, name='register'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)