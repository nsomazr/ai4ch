from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm, PasswordResetForm, SetPasswordForm
from django.contrib.auth.models import User
from .models import UserProfile
from django_countries.data import COUNTRIES
from django.contrib.auth.forms import AuthenticationForm
from .models import UserProfile
from django.core.exceptions import ValidationError


class UserLoginForm(forms.Form):

    username = forms.CharField(max_length=200, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type Username', 'id':'username'})))
    password = forms.CharField(max_length=200, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Type Password', 'id':'password'})))

    def clean(self):
        cleaned_data = super().clean()
        username = cleaned_data.get('username')
        password = cleaned_data.get('password')

        # Attempt authentication against the custom user model
        try:
            custom_user = UserProfile.objects.get(username=username)
        except UserProfile.DoesNotExist:
            custom_user = None
        except Exception as e:
            raise ValidationError('An unexpected error occurred while authenticating against the custom user model.: ', e)

        try:
            if custom_user and custom_user.check_password(password):
                return cleaned_data
        except Exception as e:
            raise ValidationError('An unexpected error occurred while authenticating against the custom user model.: ', e)
        
        # If custom authentication fails, try the default user model
        default_auth_form = AuthenticationForm(data={'username': username, 'password': password})
        try:
            if default_auth_form.is_valid():
                return cleaned_data
        except Exception as e:
            raise ValidationError('An unexpected error occurred while authenticating against the default user model.: ', e)

        # Both custom and default authentication failed, add errors
        # raise ValidationError('Invalid username or password for both default and custom user models.')



# class UserLoginForm(AuthenticationForm):

#     def __init__(self, *args, **kwargs):
#         super(UserLoginForm, self).__init__(*args, **kwargs)

#     username = forms.CharField(max_length=200, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type Username', 'id':'username'})))
#     password = forms.CharField(max_length=200, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Type Password', 'id':'password'})))


class NewUserForm(UserCreationForm):

    def __init__(self, *args, **kwargs):
        super(UserCreationForm, self).__init__(*args, **kwargs)
    
    first_name = forms.CharField(required=True, max_length=30, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type firstname', 'id':'first-name'})))
    last_name = forms.CharField(required=True, max_length=30, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type last name', 'id':'last-name'})))
    username = forms.CharField(required=True, max_length=30, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Username', 'id':'username'})))
    company = forms.CharField(required=True, max_length=30, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Company/Institution', 'id':'company'})))
    email = forms.EmailField(required=True, max_length=100, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type Email', 'id':'email'})))
    country = forms.ChoiceField(choices=sorted(COUNTRIES.items()), widget=(forms.Select(attrs={'class':'form-control','placeholder':'Select Country', 'id':'country'})))
    password1 = forms.CharField(max_length=500, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Type Password', 'id':'password'})))
    password2 = forms.CharField(max_length=500, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Re-Type Password', 'id':'cpassword'})))

    class Meta:

        model = User

        fields = ('first_name', 'last_name', 'username', 'company', 'country', 'email', 'password1', 'password2')

    def save(self, commit=True):
        user = super(NewUserForm, self).save(commit=False)
        user.email = self.cleaned_data['email']
        user.company = self.cleaned_data['company']
        user.country = self.cleaned_data['country']
        if commit:
            user.save()
        return user

class ResetPasswordForm(PasswordResetForm):

    def __init__(self, *args, **kwargs):
        super(PasswordResetForm, self).__init__(*args, **kwargs)

    email = forms.EmailField(required=True, max_length=100, widget=(forms.TextInput(attrs={'class':'form-control','placeholder':'Type Email', 'id':'email'})))

class ConfirmResetForm(SetPasswordForm):

    new_password1 = forms.CharField(max_length=200, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Type New Password', 'id':'password1'})))
    new_password2 = forms.CharField(max_length=200, widget=(forms.PasswordInput(attrs={'class':'form-control','placeholder':'Confirm New Password', 'id':'password2'})))


class StaffForm(UserCreationForm):
    first_name = forms.CharField(required=True, max_length=30, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First Name', 'id': 'first_name'}))
    last_name = forms.CharField(required=True, max_length=30, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last Name', 'id': 'username'}))
    username = forms.CharField(required=True, max_length=30, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Username', 'id': 'username'}))
    email = forms.EmailField(required=True, max_length=100, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Type Email', 'id': 'email'}))
    role = forms.ChoiceField(
        choices=[
            ('pi', 'PI'), ('co_pi', 'CO-PI'), ('ltc', 'Lab Training Coordinator'),
            ('lo', 'Laison Officer'), ('coordinator', 'Coordinator'), ('asst_coordinator', 'Asst. Coordinator'),
            ('researcher', 'Researcher'), ('asst_researcher', 'Asst. Researcher'), ('innovator', 'Innovator'),
            ('student', 'Student')
        ],
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    password1 = forms.CharField(max_length=500, widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Type Password', 'id': 'password'}))
    password2 = forms.CharField(max_length=500, widget=forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': 'Re-Type Password', 'id': 'cpassword'}))

    class Meta:
        model = UserProfile
        fields = ['first_name', 'last_name', 'username', 'email', 'role', 'password1', 'password2']

    def save(self, commit=True):
        user = super(StaffForm, self).save(commit=False)
        user.first_name = self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.username = self.cleaned_data['username']
        user.role = self.cleaned_data['role']
        user.email = self.cleaned_data['email']

        # Set the password using set_password
        password = self.cleaned_data['password1']
        user.set_password(password)

        user.is_staff = True  # Use True instead of 1 for clarity
        if commit:
            user.save()
        return user
