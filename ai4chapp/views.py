from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, template_name='pages/index.html', context={})

def about(request):
    return render(request, template_name='pages/about.html', context={})

def news(request):
    return render(request, template_name='pages/news.html', context={})