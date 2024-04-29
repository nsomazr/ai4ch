"""ai4ch URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
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
from django.urls import path, include
from ai4chapp import urls as a4chapp_urls
from users import urls as users_urls
from news import urls as news_urls
from django.contrib.auth import views as auth_views 
from beans import urls as beans_urls
from cassava import urls as cassava_urls
from maize import urls as maize_urls
from rice import urls as rice_urls
from django.conf import settings
from django.conf.urls.static import static
# from users.forms import ConfirmResetForms
urlpatterns = [
    path("admin/", admin.site.urls),
    # include django browserable 
    path("auth-api/", include("rest_framework.urls", namespace="rest_framework")),
    path('',include(a4chapp_urls)),
    path('users/', include(users_urls)),
    path('news/', include(news_urls)),
    # we can mention them this way
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(template_name='users/password/password_reset_done.html'), name='password_reset_done'),
    # path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(form_class=ConfirmResetForm, template_name="users/password/password_reset_confirm.html"), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(template_name='users/password/password_reset_complete.html'), name='password_reset_complete'),   
    # for social media authentication
    path('accounts/', include('allauth.urls')),   
    path('tinymce/', include('tinymce.urls')),
    path('ckeditor/', include('ckeditor_uploader.urls')),
    path('beans/', include(beans_urls)),
    path('cassava/', include(cassava_urls)),
    path('maize/', include(maize_urls)),
    path('rice/', include(rice_urls))
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
