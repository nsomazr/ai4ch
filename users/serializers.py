from rest_framework import serializers
from django.contrib.auth.models import User
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer

class UserModelSerializer(serializers.ModelSerializer):

    # write_only argument is set to True, which means that the password
    # field will only be used when writing data to the serializer 
    # but will not be included in the serialized output 
    # since we do not want the password to be compromised.

    password = serializers.CharField(min_length = 8, write_only = True) #password wont be included in the serialized output

    class Meta:

        model = User

        fields = ('first_name','last_name','username','email', 'password')


class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['username'] = user.username
        token['email'] = user.email
        return token
