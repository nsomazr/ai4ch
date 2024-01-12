from django.db import models
from django.contrib.auth.models import User
from django.utils import timezone
from django.contrib.auth.models import AbstractUser


class UserProfile(AbstractUser):
    # custom fields for userprofile
    first_name = models.CharField(max_length=500)
    last_name = models.CharField(max_length=500)
    email = models.EmailField(max_length=500, null=False, blank=False)
    status = models.IntegerField(default=0)
    role_choices = [
        ('pi', 'Principal Investigator'),
        ('co_pi', 'Co-Principal Investigator'),
        ('collaborator', 'Collaborator'),
    ]
    role = models.CharField(max_length=20, choices=role_choices)
    password = models.CharField(max_length=500) 

    # Set unique related names for groups and user_permissions
UserProfile._meta.get_field("groups").remote_field.related_name = "userprofile_groups"
UserProfile._meta.get_field("user_permissions").remote_field.related_name = "userprofile_user_permissions"