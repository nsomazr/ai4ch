from rest_framework import serializers
from .models import News

class NewsModelSerializer(serializers.ModelSerializer):

    class Meta:

        model = News

        fields = ('heading', 'banner', 'highlight', 'content')