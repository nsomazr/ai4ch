from django.contrib import admin
from django.contrib.auth.models import User
from .models import PlatformUser

admin.site.register(PlatformUser)
# Register your models here.
