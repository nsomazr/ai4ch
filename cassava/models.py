from django.db import models
# Create your models here.
import os
from users.models import User
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))

class CassavaData(models.Model):
    file_id = models.CharField(max_length=100)
    file_path = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,'files'))
    file_name = models.CharField(max_length=100)
    latitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    longitude = models.DecimalField(max_digits=9, decimal_places=6, null=True, blank=True)
    region = models.CharField(max_length=255, null=True, blank=True)
    district = models.CharField(max_length=255, null=True, blank=True)
    country = models.CharField(max_length=255, null=True, blank=True)
    full_address = models.TextField(null=True, blank=True)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(default=timezone.now)

class CassavaPredictionResult(models.Model):
    """
    Model to store cassava classification prediction results
    """
    DISEASE_CHOICES = [
        ('Bacterial Blight', 'Bacterial Blight'),
        ('Brown Spot', 'Brown Spot'),
        ('Browm Steak', 'Browm Steak'),
        ('Green Mottle', 'Green Mottle'),
        ('Healthy', 'Healthy'),
        ('Mosaic Desease', 'Mosaic Desease')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cassava_predictions')
    result_id = models.CharField(max_length=100,unique=True)
    file_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to=os.path.join(BASE_DIR,'cassava_predictions'))
    predicted_disease = models.CharField(max_length=50, choices=DISEASE_CHOICES)
    confidence_score = models.FloatField()
    probabilities = models.JSONField(help_text='Detailed probabilities for each disease class')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.file_name} - {self.predicted_disease}"

class CassavaDetectionResult(models.Model):
    """
    Model to store maize object detection results
    """
    TYPE_CHOICES = [
        ('image', 'Image'),
        ('video', 'Video')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='cassava_detections')
    result_id = models.CharField(max_length=100,unique=True)
    file_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to=os.path.join(BASE_DIR,'cassava_detections'))
    output_path = models.FileField(upload_to=os.path.join(BASE_DIR,'cassava_detection_outputs'))
    file_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    detection_results = models.JSONField(help_text='Detailed detection results including classes and counts')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.file_name} - Detection Results"

