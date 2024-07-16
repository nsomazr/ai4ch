from django.shortcuts import render
from users.forms import UserLoginForm

# Create your views here.

def login(request):
    login_form = UserLoginForm()
    return render(request=request, template_name="backend/pages/login.html", context={"login_form": login_form})
