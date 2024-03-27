from django.db import models
# Create your models here.
import os

from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))


class CassavaData(models.Model):
    image_id = models.CharField(max_length=100)
    image_path = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,'images'))
    image_name = models.CharField(max_length=100)
    upload_date = models.DateTimeField(default=timezone.now)
