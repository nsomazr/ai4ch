# serializers.py
from rest_framework import serializers

class ImageSerializer(serializers.Serializer):
      image = serializers.ImageField()
      
class FileSerializer(serializers.Serializer):
      file = serializers.FileField()
      user_id= serializers.IntegerField()