from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "reports"  

urlpatterns = [
    path('reports/', view=views.report_view, name='traffic'),
    path('report/download/', views.download_csv_report, name='download_csv_report'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
