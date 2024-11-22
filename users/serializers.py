# serializers.py
from rest_framework import serializers
from django.contrib.auth.password_validation import validate_password
from rest_framework_simplejwt.serializers import TokenObtainPairSerializer
from .models import User
from django.utils.translation import gettext_lazy as _

class UserRegistrationSerializer(serializers.ModelSerializer):
    password1 = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    password2 = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ('email', 'phone_number', 'region', 'district', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        # Accept a custom context parameter to identify the use case
        self.registration = kwargs.pop('registration', False)
        super().__init__(*args, **kwargs)

        # Make region and district required only during registration
        if self.registration:
            self.fields['region'].required = True
            self.fields['district'].required = True
        else:
            self.fields['region'].required = False
            self.fields['district'].required = False
            # Add the 'role' field dynamically for non-registration cases
            self.fields['role'] = serializers.ChoiceField(
                choices=User.ROLE_CHOICES, required=True
            )

    def validate(self, attrs):
        if attrs['password1'] != attrs['password2']:
            raise serializers.ValidationError({"password": "Password fields didn't match."})
        return attrs

    def create(self, validated_data):
        # Ensure role is handled correctly for non-registration cases
        role = validated_data.pop('role', 'normal') if not self.registration else 'normal'

        user = User.objects.create(
            email=validated_data['email'],
            phone_number=validated_data['phone_number'],
            region=validated_data.get('region'),  # Optional
            district=validated_data.get('district'),  # Optional
            role=role,  # Assign role
            username=str(validated_data['email']).split('@')[0]  # Using email as username
        )
        user.set_password(validated_data['password1'])
        user.save()
        return user



class MyTokenObtainPairSerializer(TokenObtainPairSerializer):
    """Custom token serializer including additional user data"""
    
    @classmethod
    def get_token(cls, user):
        token = super().get_token(user)
        token['email'] = user.email
        token['role'] = user.role
        return token

class ChangePasswordSerializer(serializers.Serializer):
    """Serializer for password change"""
    
    old_password = serializers.CharField(required=True)
    new_password = serializers.CharField(
        required=True,
        validators=[validate_password]
    )
    confirm_password = serializers.CharField(required=True)

    def validate(self, data):
        if data['new_password'] != data['confirm_password']:
            raise serializers.ValidationError({
                "new_password": _("Password fields didn't match.")
            })
        return data

    def validate_old_password(self, value):
        user = self.context['request'].user
        if not user.check_password(value):
            raise serializers.ValidationError(_("Old password is not correct"))
        return value
    
    
class LoginSerializer(serializers.Serializer):
    username = serializers.CharField(required=True)
    password = serializers.CharField(required=True, write_only=True)

    def validate(self, attrs):
        return attrs
