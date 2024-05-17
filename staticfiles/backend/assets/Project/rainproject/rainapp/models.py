from django.db import models
# Create your models here.
import os

from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))


class RainData(models.Model):
    image_id = models.CharField(max_length=100)
    filepaths = models.FileField(max_length=500,upload_to=os.path.join(BASE_DIR,'images'))
    filename = models.CharField(max_length=100)
    upload_date = models.DateTimeField(default=timezone.now)

from django.db import models
from django.utils import timezone

# Create your models here.


class Users(models.Model):
    user_id = models.AutoField(primary_key=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    email = models.EmailField(max_length=100)
    username = models.EmailField(max_length=100)
    password = models.CharField(max_length=100)
    status = models.IntegerField()
    created_at = models.DateTimeField(default=timezone.now)