from django.shortcuts import render
from news.models import News

# Create your views here.

def home(request):
    news = News.objects.filter(publish=1, status=1)
    context = {"news": news}
    return render(request, template_name='system/pages/index.html', context=context)

def about(request):
    return render(request, template_name='system/pages/about.html', context={})
