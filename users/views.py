from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
# from rest_framework import viewsets
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import UserModelSerializer, MyTokenObtainPairSerializer
from django.contrib.auth.models import User
from django.shortcuts import  render, redirect
from .forms import NewUserForm
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from rest_framework import viewsets
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm 
from django.core.mail import send_mail, BadHeaderError
from django.http import HttpResponse
from django.contrib.auth.forms import PasswordResetForm
from django.template.loader import render_to_string
from django.db.models.query_utils import Q
from django.utils.http import urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.core.mail import EmailMultiAlternatives
from django import template
from .forms import UserLoginForm, ResetPasswordForm
from .forms import StaffForm
from .models import UserProfile
from django.contrib.auth.hashers import check_password

class UsersAPIView(APIView):

    def get(self, request):
        users = User.objects.all()
        serializer = UserModelSerializer(users, many = True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = UserModelSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = 201)
        return Response(serializer.errors, status = 400)


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


def register_request(request):
    if request.method == 'POST':
       register_form = NewUserForm(request.POST)
       if register_form.is_valid():
          user = register_form.save()
          username = register_form.cleaned_data.get('username')
          messages.success(request, "Registration successful." )
        #   login(request, user)
          backend = 'django.contrib.auth.backends.ModelBackend'
          login(request, user, backend=backend)
          return redirect("users:dashboard")
       else:
          messages.error(request,"Account creation failed")
        #   print(register_form.errors.as_data()) # here you print errors to terminal
          return redirect("users:register")

    register_form = NewUserForm()
    return render (request=request, template_name="system/users/register.html", context={"register_form":register_form})

def update_user(request,id):
		user= User.objects.get(id=id)
		update_form = NewUserForm(instance=user)# prepopulate the form with an existing band
		return render(request, 'system/users/update_user.html',{'update_form': update_form})


# def login_request(request):
#     print("Am in 1")
#     if request.method == "POST":
#         print("Am in 2")
#         try:
#             print("Am in 3")
#             form = UserLoginForm(request, data=request.POST)
#             print("Am in 4")
#             username = request.POST['username']
#             password = request.POST['password']
#             if form.is_valid():
#                 print("Am in 5")
#                 username = form.cleaned_data.get('username')
#                 password = form.cleaned_data.get('password')
#                 print("Form in", password)
#                 user = authenticate(username=username, password=password)
#                 print("Auth good")
#                 if user is not None:
#                     login(request, user)
#                     messages.info(request, f"You are now logged in as {username}.")
#                     request.session['user_id'] = user.id
#                     return redirect("users:dashboard")
#                 else:
#                     print("User not found")
#                     messages.error(request, "Invalid username or password.")
#             elif UserProfile.objects.filter(username=username).exists():
#                 print("From UserProfile")
#                 user = UserProfile.objects.get(username=username, password=password)
#                 login(request, user)
#                 messages.info(request, f"You are now logged in as {username}.")
#                 request.session['user_id'] = user.id
#                 return redirect("users:dashboard")
#             else:
#                 print(form.errors.as_data())
#                 messages.error(request, "Invalid username or password.")
#         except Exception as e:
#             print("Error: ", e)
#             messages.error(request, f"An error occurred: {str(e)}")

#     login_form = UserLoginForm()
#     return render(request=request, template_name="users/login.html", context={"login_form": login_form})

def login_request(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data['username']
            password = form.cleaned_data['password']

            try:
                user = authenticate(request, username=username, password=password)
                custom_user_data = UserProfile.objects.filter(username=username)
                if user and user.is_active:
                    login(request, user)
                    request.session['user_id'] = user.id
                    return redirect("users:dashboard")
                elif custom_user_data:
                    try:
                        custom_user = UserProfile.objects.get(username=username)
                        if check_password(password, custom_user.password):
                            print("Custom user password check successful")
                            backend = 'users.backends.CustomUserBackend'
                            login(request, custom_user, backend=backend)
                            request.session['user_id'] = custom_user.id
                            request.session['username'] = custom_user.username
                            request.session['first_name'] = custom_user.first_name
                            request.session['last_name'] = custom_user.last_name
                            request.session['email'] = custom_user.email
                            request.session['role'] = custom_user.role
                            return redirect("users:dashboard")
                        else:
                            print("Password check failed")
                    except UserProfile.DoesNotExist:
                        print("User not found")
                    except Exception as e:
                        print(f"An unexpected error occurred: {e}")

                    else:
                        form.add_error(None, 'Invalid username or password.')
                else:
                    # Authentication failed, display an error message
                    form.add_error(None, 'Invalid username or password.')
                    

            except UserProfile.DoesNotExist:
                # Handle the case where the custom user does not exist
                form.add_error('username', 'User does not exist.')
            except Exception as e:
                # Handle other exceptions, log them, or take appropriate action
                print(f"An unexpected error occurred: {e}")
                form.add_error(None, 'An unexpected error occurred during authentication.')
        else:
            print(form.errors.as_data())
            messages.error(request, "Invalid username or password.")
    else:
        login_form = UserLoginForm()
        return render(request=request, template_name="system/dashboard/login.html", context={"login_form": login_form})

def dashboard(request):
    return render(request, template_name = 'system/dashboard/admin.html', context={})

def logout_request(request):
	request.session.clear()  # Clears all session data for the current sessio
    # request.session.flush()  # Same as clear(), but also deletes the session cookie
	logout(request)
	messages.info(request, "You have successfully logged out.") 
	return redirect("ai4chapp:home")

#email sms single alternative

# def password_reset_request(request):
# 	if request.method == "POST":
# 		password_reset_form = PasswordResetForm(request.POST)
# 		if password_reset_form.is_valid():
# 			data = password_reset_form.cleaned_data['email']
# 			associated_users = User.objects.filter(Q(email=data))
# 			if associated_users.exists():
# 				for user in associated_users:
# 					subject = "Password Reset Requested"
# 					email_template_name = "main/password/password_reset_email.txt"
# 					c = {
# 					"email":user.email,
# 					'domain':'127.0.0.1:8000',
# 					'site_name': 'Website',
# 					"uid": urlsafe_base64_encode(force_bytes(user.pk)),
# 					'token': default_token_generator.make_token(user),
# 					'protocol': 'http',
# 					}
# 					email = render_to_string(email_template_name, c)
# 					try:
# 						send_mail(subject, email, 'admin@example.com' , [user.email], fail_silently=False)
# 					except BadHeaderError:

# 						return HttpResponse('Invalid header found.')
						
# 					messages.success(request, 'A message with reset password instructions has been sent to your inbox.')
# 					return redirect ("main:homepage")
# 			messages.error(request, 'An invalid email has been entered.')
# 	password_reset_form = PasswordResetForm()
# 	return render(request=request, template_name="main/password/password_reset.html", context={"password_reset_form":password_reset_form})

# send email multi alternative
def password_reset_request(request):
	if request.method == "POST":
		password_reset_form = ResetPasswordForm(request.POST)
		if password_reset_form.is_valid():
			data = password_reset_form.cleaned_data['email']
			associated_users = User.objects.filter(Q(email=data)|Q(username=data))
			if associated_users.exists():
				for user in associated_users:
					subject = "Password Reset Requested"
					plaintext = template.loader.get_template('system/users/password/password_reset_email.txt')
					htmltemp = template.loader.get_template('system/users/password/password_reset_email.html')
					c = { 
					"email":user.email,
					'domain':'127.0.0.1:8000',
					'site_name': 'Website',
					"uid": urlsafe_base64_encode(force_bytes(user.pk)),
					"user": user,
					'token': default_token_generator.make_token(user),
					'protocol': 'http',
					}
					text_content = plaintext.render(c)
					html_content = htmltemp.render(c)
					try:
						msg = EmailMultiAlternatives(subject, text_content, 'Website nsoma.me>', [user.email], headers = {'Reply-To': 'ai@nsoma.me'})
						msg.attach_alternative(html_content, "text/html")
						msg.send()
					except BadHeaderError:
						return HttpResponse('Invalid header found.')
					messages.info(request, "Password reset instructions have been sent to the email address entered.")
					return redirect ("password_reset_done")
	password_reset_form = ResetPasswordForm()
	return render(request=request, template_name="system/users/password/password_reset.html", context={"password_reset_form":password_reset_form})


def add_staff(request):
    if request.method == 'POST':
        staff_form = StaffForm(request.POST)
        try:
            if staff_form.is_valid():
                user = staff_form.save()
                username = staff_form.cleaned_data.get('username')
                messages.success(request, "Registration successful.")
                # Uncomment the following lines if you want to log in the user after registration
                # login(request, user, backend='django.contrib.auth.backends.ModelBackend')
                # messages.success(request, f"Welcome, {username}!")
                return redirect("users:staffs")
            else:
                messages.error(request, "Account creation failed")
                # print(staff_form.errors.as_data())
                return redirect("users:add-staff")
        except Exception as e:
            # Print the error message to identify the issue
            # print(f"An error occurred: {e}")
            messages.error(request, "An error occurred during account creation.")
            return redirect("users:add-staff")

    staff_form = StaffForm()
    return render(request=request, template_name="system/dashboard/add_staff.html", context={"staff_form": staff_form})


def staffs(request):
	staffs = UserProfile.objects.all()
	context = {'staffs':staffs}
	return render(request, template_name='system/dashboard/staffs.html', context=context)
