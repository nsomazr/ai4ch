from django.contrib.auth.models import AbstractUser
from django.db import models
from django.core.validators import RegexValidator

class User(AbstractUser):
    # Role choices
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('manager', 'Manager'),
        ('agrovet', 'Agrovet'),
        ('normal', 'Normal'),
    ]

    email = models.EmailField(unique=True)
    phone_regex = RegexValidator(
        regex=r'^\+255[0-9]{9}$',
        message="Phone number must be entered in the format: '+255XXXXXXXXX'"
    )
    phone_number = models.CharField(
        validators=[phone_regex],
        max_length=13,
        unique=True
    )
    region = models.CharField(max_length=100, null=True, blank=True)
    district = models.CharField(max_length=100, null=True, blank=True)
    ward = models.CharField(max_length=100, null=True, blank=True)
    street = models.CharField(max_length=100, null=True, blank=True)
    role = models.CharField(
        max_length=20,
        choices=ROLE_CHOICES,
        default='normal'  # Default role for registration
    )
    is_verified = models.BooleanField(default=False)
    verification_code = models.IntegerField(null=True)
    status = models.BooleanField(default=True)
    # Make email the primary identifier instead of username
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username', 'phone_number', 'region', 'district','ward','street']

    groups = models.ManyToManyField(
        'auth.Group',
        related_name='custom_user_groups',
        blank=True,
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='custom_user_permissions',
        blank=True,
    )


    class Meta:
        db_table = 'users'
        verbose_name = 'User'
        verbose_name_plural = 'Users'

    def __str__(self):
        return self.email
