from django.db import models
# Create your models here.
import os
from users.models import User
from django.utils import timezone

BASE_DIR = os.path.dirname(os.path.dirname(os.path.relpath(__file__)))


class MaizeData(models.Model):
    file_id = models.CharField(max_length=100)
    file_path = models.FileField(max_length=200,upload_to=os.path.join(BASE_DIR,'files'))
    file_name = models.CharField(max_length=100)
    uploaded_by = models.ForeignKey(User, on_delete=models.CASCADE)
    upload_date = models.DateTimeField(default=timezone.now)

class MaizePredictionResult(models.Model):
    """
    Model to store maize classification prediction results
    """
    DISEASE_CHOICES = [
        ('Common Rust', 'Common Rust'),
        ('Fally Army Worm', 'Fally Army Worm'),
        ('Gray Leaf Spot', 'Gray Leaf Spot'),
        ('Healthy', 'Healthy'),
        ('Leaf Blight', 'Leaf Blight'),
        ('Lethal Necrosis', 'Lethal Necrosis'),
        ('Pysoderma Leaf Spot', 'Pysoderma Leaf Spot'),
        ('Streak Virus', 'Streak Virus'),
        ('Undetermined', 'Undetermined')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='maize_predictions')
    result_id = models.CharField(max_length=100,unique=True)
    file_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to=os.path.join(BASE_DIR,'maize_predictions'))
    predicted_disease = models.CharField(max_length=50, choices=DISEASE_CHOICES)
    confidence_score = models.FloatField()
    probabilities = models.JSONField(help_text='Detailed probabilities for each disease class')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.file_name} - {self.predicted_disease}"

class MaizeDetectionResult(models.Model):
    """
    Model to store maize object detection results
    """
    TYPE_CHOICES = [
        ('image', 'Image'),
        ('video', 'Video')
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='maize_detections')
    result_id = models.CharField(max_length=100,unique=True)
    file_name = models.CharField(max_length=255)
    file_path = models.FileField(upload_to=os.path.join(BASE_DIR,'maize_detections'))
    output_path = models.FileField(upload_to=os.path.join(BASE_DIR,'maize_detection_outputs'))
    file_type = models.CharField(max_length=10, choices=TYPE_CHOICES)
    detection_results = models.JSONField(help_text='Detailed detection results including classes and counts')
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"{self.file_name} - Detection Results"
