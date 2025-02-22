"""
ai4ch URL Configuration

The `urlpatterns` list routes URLs to views. For more information, please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.auth.decorators import login_required
from django.views.decorators.cache import never_cache
from django.contrib.auth import views as auth_views
from ckeditor_uploader import views as ckeditor_views
from django.views.static import serve  
from django.conf.urls.i18n import i18n_patterns

urlpatterns = [
    path('beans/detect/', include('beans.api_urls', namespace='beans-api-detect')), 
    path('cassava/detect/', include('cassava.api_urls', namespace='cassava-api-detect')), 
    path('maize/detect/', include('maize.api_urls', namespace='maize-api-detect')), 
    path('rice/detect/', include('rice.api_urls', namespace='rice-api-detect')), 
    path('i18n/', include('django.conf.urls.i18n')),
]

urlpatterns += i18n_patterns(
    path('admin/', admin.site.urls),
    # Include Django browsable API
    path('auth-api/', include('rest_framework.urls', namespace='rest_framework')),
    # re_path(r'^media/(?P<path>.*)$', serve,{'document_root': settings.MEDIA_ROOT}),
    # re_path(r'^static/(?P<path>.*)$', serve,{'document_root': settings.STATIC_ROOT}),
    # Application URLs
    path('', include('ai4chapp.urls')),
    path('users/', include('users.urls')),
    path('news/', include('news.urls', namespace='news')),
    path('beans/', include('beans.urls', namespace='beans')),   
    path('cassava/', include('cassava.urls', namespace='cassava')),
    path('maize/', include('maize.urls', namespace='maize')),
    path('rice/', include('rice.urls', namespace='rice')),
    path('reports/', include('reports.urls')),
    # Password reset views
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(
        template_name='users/password/password_reset_done.html'), name='password_reset_done'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(
        template_name='users/password/password_reset_complete.html'), name='password_reset_complete'),
    
    # Social authentication
    path('accounts/', include('allauth.urls')),
    
    # WYSIWYG editor
    path('tinymce/', include('tinymce.urls')),
    re_path(r'^ckeditor/upload/', login_required(ckeditor_views.upload), name='ckeditor_upload'),
    re_path(r'^ckeditor/browse/', never_cache(login_required(ckeditor_views.browse)), name='ckeditor_browse'),
) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)

# + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


# if settings.DEBUG:
#    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)