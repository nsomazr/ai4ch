from django import forms

class ImageHorizontal(forms.Form):
    image_file = forms.FileField(max_length=200, widget=(forms.FileInput(attrs={'class': 'form-control input-file img-responsive img-rounded',
                                                                                'id': 'img_section_one',
                                                                                'name': 'image_file',
                                                                                'placeholder': 'Choose eye image',
                                                                                'onchange': 'form.submit()'})))


class RegisterForm(forms.Form):
    first_name=forms.CharField(max_length=20,widget=(forms.TextInput(attrs={'class':'form-control','placeholder':''})))
    last_name = forms.CharField(max_length=20,widget=(forms.TextInput(attrs={'class': 'form-control', 'placeholder': ''})))
    email = forms.EmailField(max_length=20, widget=(forms.EmailInput(attrs={'class': 'form-control', 'placeholder': ''})))
    username = forms.CharField(max_length=20,widget=(forms.TextInput(attrs={'class': 'form-control', 'placeholder': ''})))
    password = forms.CharField(max_length=20,required=True, widget=(forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': ''})))
    cpassword = forms.CharField(max_length=20,required=True, widget=(forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': ''})))
    
    

class LoginForm(forms.Form):
    email_username = forms.CharField(max_length=20, widget=(forms.TextInput(attrs={'class': 'form-control', 'placeholder': '','id':'fi'})))
    password = forms.CharField(max_length=20, widget=(forms.PasswordInput(attrs={'class': 'form-control', 'placeholder': '','id':'si'})))