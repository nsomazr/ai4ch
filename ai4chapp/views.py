from django.shortcuts import render

# Create your views here.

def login(request):
    return render(request=request, template_name="backend/pages/login.html")
