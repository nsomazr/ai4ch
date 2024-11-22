from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
# from rest_framework import viewsets
from rest_framework_simplejwt.views import TokenObtainPairView
from .serializers import UserRegistrationSerializer, MyTokenObtainPairSerializer, LoginSerializer
from django.contrib.auth.models import User
from django.shortcuts import  render, redirect
from django.urls import reverse
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from rest_framework import viewsets
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib.auth.forms import AuthenticationForm 
from django.core.mail import send_mail, BadHeaderError
from django.http import HttpResponse
from django.contrib.auth.forms import PasswordResetForm
from django.template.loader import render_to_string
from django.db.models.query_utils import Q
from rest_framework.decorators import api_view
from django.utils.http import urlsafe_base64_encode
from django.contrib.auth.tokens import default_token_generator
from django.utils.encoding import force_bytes
from django.core.mail import EmailMultiAlternatives
from django import template
from .models import PlatformUser
from news.models import News
from beans.models import BeansData
from maize.models import MaizeData
from rice.models import RiceData
from cassava.models import CassavaData
from django.contrib.auth.hashers import check_password,make_password
from mtaa import tanzania
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.utils.translation import gettext_lazy as _
from rest_framework.response import Response
from rest_framework import status
from django.core.exceptions import ValidationError
import logging


logger = logging.getLogger(__name__)

class UsersAPIView(APIView):

    def get(self, request):
        users = User.objects.all()
        serializer = UserRegistrationSerializer(users, many = True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = UserRegistrationSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = 201)
        return Response(serializer.errors, status = 400)


class MyTokenObtainPairView(TokenObtainPairView):
    serializer_class = MyTokenObtainPairSerializer


class LocationAPIView(APIView):
    
    def get_districts(self, request):
        region = request.GET.get('region')
        if not region:
            return JsonResponse({'error': 'Region parameter is required'}, status=400)
        
        try:
            # Access region directly as attribute
            region_obj = getattr(tanzania, str(region))
            # Convert districts to list if it's not already
            districts = list(region_obj.districts)
            return JsonResponse({'districts': districts[1:]})
        except (AttributeError, TypeError) as e:
            return JsonResponse({'error': f'Error getting districts for region {region}: {str(e)}'}, status=404)

    def get_wards(self, request):
        region = request.GET.get('region')
        district = request.GET.get('district')
        if not (region and district):
            return JsonResponse({'error': 'Both region and district parameters are required'}, status=400)
        
        try:
            # Access region and district attributes
            region_obj = getattr(tanzania, str(region))
            # Get wards for the district
            district_data = getattr(region_obj.districts, district.replace(' ', '_'))
            wards = list(district_data.wards)
            return JsonResponse({'wards': wards[1:]})
        except (AttributeError, TypeError) as e:
            return JsonResponse({'error': f'Error getting wards: {str(e)}'}, status=404)

    def get_streets(self, request):
        region = request.GET.get('region')
        district = request.GET.get('district')
        ward = request.GET.get('ward')
        if not (region and district and ward):
            return JsonResponse({'error': 'Region, district, and ward parameters are required'}, status=400)
        
        try:
            # Access region, district and ward attributes
            region_obj = getattr(tanzania, region)
            district_data = getattr(region_obj.districts, district.replace(' ', '_'))
            ward_data = getattr(district_data.wards, ward.replace(' ', '_'))
            streets = list(ward_data.streets)
            return JsonResponse({'streets': streets[1:]})
        except (AttributeError, TypeError) as e:
            return JsonResponse({'error': f'Error getting streets: {str(e)}'}, status=404)

@require_http_methods(["GET", "POST"])
def register_request(request):
    """
    Hybrid view handling both form rendering and API-style registration
    GET: Renders registration form
    POST: Processes registration data using serializer
    """
    if request.method == "GET":
        # Handle GET request - render form
        regions = [region for region in dir(tanzania) 
                  if not region.startswith('_') and region != 'post_code']
        return render(request, 'backend/pages/register.html', {
            'regions': regions
        })

    # Handle POST request - process data using serializer
    serializer = UserRegistrationSerializer(data=request.POST, registration=True)
    
    try:
        if serializer.is_valid():
            # Save the user
            user = serializer.save()
            
            # Log the user in
            # login(request, user)
            
            return JsonResponse({
                'success': True,
                'message': _('Registration successful!'),
                'redirect_url': reverse('users:login'),
                'user': {
                    'email': user.email,
                    'phone_number': user.phone_number,
                    'region': user.region,
                    'district': user.district
                }
            })
        
        # Handle validation errors
        return JsonResponse({
            'success': False,
            'message': _('Please correct the errors below.'),
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        # Handle unexpected errors
        return JsonResponse({
            'success': False,
            'message': _('Registration failed.'),
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

def update_user(request,id):
        if request.session.get('user_id'):
            user= User.objects.get(id=id)
            # update_form = PlatformUserLoginForm(instance=user)# prepopulate the form with an existing band
            return render(request, 'backend/users/update_user.html')
        else:
            return redirect("ai4chapp:login")

def login_request(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')

        logger.info(f"Login attempt for username: {email}")

        if not email or not password:
            logger.warning("Login failed: email or password missing")
            return JsonResponse({
                "success": False,
                "message": "Both email and password are required.",
                "redirect_url": None
            }, status=400)

        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            request.session['user_id'] = user.id

            logger.info(f"Login successful for email: {email}")
            return JsonResponse({
                "success": True,
                "message": "Login successful. Redirecting to your dashboard.",
                "redirect_url": reverse('users:dashboard')
            }, status=200)
        else:
            logger.warning(f"Login failed for email: {email}")
            return JsonResponse({
                "success": False,
                "message": "Invalid username or password.",
                "redirect_url": reverse('users:login')
            }, status=400)
    else:
        return render(request, 'backend/pages/login.html')
    
def dashboard(request):
    if request.session.get('user_id'):
        news = News.objects.filter(status=1, publisher=request.session['user_id'])
        pulished_news = News.objects.filter(publish=1, status=1,publisher=request.session['user_id'])
        beans = BeansData.objects.filter(uploaded_by=request.session['user_id'])
        cassava = CassavaData.objects.filter(uploaded_by=request.session['user_id'])
        maize = MaizeData.objects.filter(uploaded_by=request.session['user_id'])
        rice = RiceData.objects.filter(uploaded_by=request.session['user_id'])
        context={'news':len(news),'published':len(pulished_news),'beans':len(beans),'cassava':len(cassava),'maize':len(maize),'rice':len(rice)}
        return render(request, template_name = 'backend/pages/admin.html', context=context)
    else:
        return redirect("ai4chapp:login")

def logout_request(request):
    request.session.clear()  # Clears all session data for the current session
    # request.session.flush()  # Same as clear(), but also deletes the session cookie
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect("ai4chapp:login")

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
		password_reset_form = PasswordResetForm(request.POST)
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
	password_reset_form = PasswordResetForm()
	return render(request=request, template_name="system/users/password/password_reset.html", context={"password_reset_form":password_reset_form})


@login_required
@require_http_methods(["GET", "POST"])
def add_staff(request):
    """
    View for adding staff members using HTML form
    """
    if request.method == "GET":
        # Get regions for the form
        regions = [region for region in dir(tanzania) 
                  if not region.startswith('_') and region != 'post_code']
        
        return render(request, 'backend/pages/add_staff.html', {
            'regions': regions
        })

# Check if request accepts JSON
# Handle POST request
    serializer = UserRegistrationSerializer(data=request.POST, registration=False)
    
    try:
        if serializer.is_valid():
            # Save the user with staff status
            user = serializer.save(is_staff=True)
            
            return JsonResponse({
                'success': True,
                'message': _('Staff member added successfully!'),
                'redirect_url': reverse('users:staff-list'),
                'user': {
                    'email': user.email,
                    'phone_number': user.phone_number,
                    'region': user.region,
                    'district': user.district
                }
            })
        
        # Return validation errors
        return JsonResponse({
            'success': False,
            'message': _('Please correct the errors below.'),
            'errors': serializer.errors
        }, status=status.HTTP_400_BAD_REQUEST)

    except Exception as e:
        return JsonResponse({
            'success': False,
            'message': _('An error occurred while adding staff.'),
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    

def staff_list(request):
    if request.session.get('user_id'):
        # Filter users by roles or permissions
        staffs = PlatformUser.objects.filter(
            Q(role__in=['admin', 'manager',]) | Q(is_staff=True) | Q(is_superuser=True)
        )
        context = {'staffs': staffs}
        return render(request, template_name='backend/pages/staff_list.html', context=context)
    else:
        # Redirect to login or handle unauthorized access
        return redirect('login')


def delete_staff(request,id):
    if request.user.id:
        staff = PlatformUser.objects.filter(id=id)
        if staff:
            staff.delete()
            messages.success(request, "Staff deleted." )
            return redirect('users:staffs')
        messages.success(request, "Staff doesn't exist." )
        return redirect('users:staff-list')
    else:
        return redirect("ai4chapp:login")

@require_http_methods(["GET", "POST"])
def update_info(request):
    if not request.user.is_authenticated:
        return JsonResponse({
            'success': False,
            'message': 'Please login to continue'
        }, status=401)

    if request.method == 'POST':
        try:
            user_id = request.user.id
            user = PlatformUser.objects.get(id=user_id)

            # Get form data
            email = request.POST.get('email', '').strip()
            first_name = request.POST.get('first_name', '').strip()
            last_name = request.POST.get('last_name', '').strip()
            username = request.POST.get('username', '').strip()

            # Validate email format
            # try:
            #     validate_email(email)
            # except ValidationError:
            #     return JsonResponse({
            #         'success': False,
            #         'message': 'Please enter a valid email address'
            #     }, status=400)

            # Check if email is already taken by another user
            if PlatformUser.objects.exclude(id=user_id).filter(email=email).exists():
                return JsonResponse({
                    'success': False,
                    'message': 'This email is already in use'
                }, status=400)

            # Check if username is already taken by another user
            if PlatformUser.objects.exclude(id=user_id).filter(username=username).exists():
                return JsonResponse({
                    'success': False,
                    'message': 'This username is already taken'
                }, status=400)

            # Update user info
            user.email = email
            user.first_name = first_name
            user.last_name = last_name
            user.username = username

            # Validate and save
            try:
                user.full_clean()
                user.save()

                # Update session data
                request.session['username'] = username
                request.session['first_name'] = first_name
                request.session['last_name'] = last_name
                # request.session['role'] = user.role

                return JsonResponse({
                    'success': True,
                    'message': 'Profile updated successfully'
                })

            except ValidationError as e:
                return JsonResponse({
                    'success': False,
                    'message': 'Validation error: ' + ', '.join(e.messages)
                }, status=400)

        except PlatformUser.DoesNotExist:
            return JsonResponse({
                'success': False,
                'message': 'User not found'
            }, status=404)
        
        except Exception as e:
            print("Error", e)
            return JsonResponse({
                'success': False,
                'message': 'An error occurred while updating your profile'
            }, status=500)

    # GET request - render the form
    return render(request, 'backend/pages/update_info.html')
    
    
def change_password(request):
    if request.session.get('user_id'):
        if request.method == 'POST':
            password = request.POST['new_password1']
            id = request.session.get('user_id')
            user= PlatformUser.objects.get(id=id)
            if  user.password == make_password(password):  
                return redirect('users:change-password')
            else:
                user.password = make_password(password)
                user.save()
                return redirect('users:dashboard')
        return render(request, template_name = 'backend/pages/change_password.html', context={})
    else:
        return redirect("ai4chapp:login")

def deactivate_staff(request,id):
    if request.session.get('user_id'):
        user= PlatformUser.objects.get(id=id)
        user.status = 0
        user.save()
        return redirect('users:staffs') 
    else:
        return redirect("ai4chapp:login")