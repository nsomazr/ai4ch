from django.urls import path
from .views import NewsAPIView
from django.conf import settings
from django.conf.urls.static import static

app_name = "news"

urlpatterns = [
    path("", NewsAPIView.news, name='news'),
    path('add-new/', NewsAPIView.add_new, name="add-new"),
    path('news-list/review-new/publish-new/<int:id>', NewsAPIView.publish_new, name="publish-new"),
    path('news-list/review-new/<int:id>', NewsAPIView.review_new, name="review-new"),
    path('news-list/view-new/<int:id>/', NewsAPIView.view_new, name="view-new"),
    path('news-list/delete-new/<int:id>/', NewsAPIView.delete_new, name="delete-new"),
    path('news-list/', NewsAPIView.news_list, name="news-list"),
    path('news/<str:slug>/', NewsAPIView.read_new, name="read-new"),
    path('<str:slug>/', NewsAPIView.read_new, name="read-new"),
    path('api/news', NewsAPIView.as_view(), name="news-api"),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT) + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)