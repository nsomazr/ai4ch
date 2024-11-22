from django.db import models
# Create your models here.
import os
from users.models import PlatformUser
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))


class MaizeData(models.Model):
    file_id = models.CharField(max_length=100)
    file_path = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,'files'))
    file_name = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey(PlatformUser, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(default=timezone.now)


