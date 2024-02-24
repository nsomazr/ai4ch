from django.urls import path,re_path
from .views import NewsAPIView
from django.conf import settings
from django.conf.urls.static import static
app_name = "news"  

urlpatterns = [
               path("news/",view=NewsAPIView.news, name='news' ),
               path('add-news/', NewsAPIView.add_news, name="add-news"),
               # path('<str:slug>/', NewsAPIView.read_news, name="index-read-news"),
               path('news/<str:slug>/', NewsAPIView.read_news, name="read-news"),
               path('news-list/review-news/publish-news/<int:id>', NewsAPIView.publish_news, name="publish-news"),
               path('news-list/review-news/<int:id>', NewsAPIView.review_news, name="review-news"),
               path('news-list/view-news/<int:id>', NewsAPIView.view_news, name="view-news"),
               path('news-list/delete-news/<int:id>', NewsAPIView.delete_news, name="delete-news"),
               path('news-list/', NewsAPIView.news_list, name="news-list"),
               path('api/news', NewsAPIView.as_view(), name="news-api")] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)


if settings.DEBUG:
   urlpatterns += static(settings.MEDIA_URL,document_root=settings.MEDIA_ROOT)