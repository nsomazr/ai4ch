from django.shortcuts import render
from news.models import News

# Create your views here.

def index(request):
    news = News.objects.filter(publish=1, status=1)
    context = {"news": news}
    return render(request, template_name='pages/index.html', context=context)

def about(request):
    return render(request, template_name='pages/about.html', context={})
