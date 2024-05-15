from django.shortcuts import render,redirect
from rest_framework.response import Response
from rest_framework.views import APIView
from .models import News
from .serializers import NewsModelSerializer
from .forms import NewsForm
from django.contrib import messages
# Create your views here.

class NewsAPIView(APIView):

    def get(self, request):
        news = News.objects.all()
        serializer = NewsModelSerializer(news, many = True)
        return Response(serializer.data)
    
    def post(self, request):
        serializer = NewsModelSerializer(data = request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status = 201)
        return Response(serializer.errors, status = 400)

    def news_list(request):
        news = News.objects.filter(publisher=request.session['user_id'])
        context = {'blogs':news}
        return render(request, template_name='backend/pages/news_list.html', context=context)

    def blogs(request):
        news = News.objects.filter(publish=1, status=1,publisher=request.session['user_id'])
        context = {"blogs": news}
        return render(request, template_name='backend/pages/blogs.html', context=context)

    def add_new(request):

        if request.method == 'POST' and request.FILES['thumbnail']:

            new_form = NewsForm(request.POST,request.FILES)
            # print(f"Body content: {request.POST['body']}")
            if blog_form.is_valid():
                title  = request.POST['title']
                body = new_form.cleaned_data['body']
                thumbnail = request.FILES['thumbnail']
                # header_image = request.FILES['header_image']
                # file= request.FILES['file']
                # description = request.POST['description']
                thematic_area= request.POST['thematic_area']
                status = 1
                # print(f"Body content: {body}")
                slug = title.replace(' ','-').lower()
                new_new = News(title=title, body=body, thumbnail=thumbnail, publisher_id=request.session['user_id'], status=status, slug=slug, thematic_area=thematic_area)
                get_objects = News.objects.filter(title=title, status=1)
                if get_objects:
                    messages.success(request, "News already exist." )
                    new_form = NewsForm()
                    return render(request, template_name='backend/pages/add_new.html', context={'new_form':new_form})
                else:
                    new_new.save()
                    messages.success(request, "News successful added." )
                    print("Here")
                    return redirect('news:news-list')
            else:
                print(new_form.errors.as_data())
                
        new_form = NewsForm()
        return render(request, template_name='backend/pages/add_new.html', context={'new_form':new_form})

    def review_new(request,id):
        new = News.objects.get(id=id)
        context = {'blog':new}
        return render(request, template_name='backend/pages/review_new.html', context=context)
    
    def read_blog(request,slug):
        new = News.objects.get(slug=slug)
        news = News.objects.filter(publish=1, status=1).exclude(slug=slug)
        context = {'new':new, 'news':news}
        return render(request, template_name='frontend/pages/read_new.html', context=context)
    
    def view_new(request,id):
        new = News.objects.get(id=id)
        context = {'new':new}
        return render(request, template_name='backend/pages/view_new.html', context=context)
    
    def publish_new(request,id):
            new = News.objects.get(id=id)
            new.publish = 1
            new.save()
            return redirect('news:news-list')
            
    
    def delete_new(request,id):
        new = News.objects.filter(id=id)
        if new:
            new.delete()
            messages.success(request, "News deleted." )
            return redirect('news:news-list')
        messages.success(request, "News doesn't exist." )
        return redirect('news:news-list')