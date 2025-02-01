from django.urls import path, include
from . import views
from django.conf import settings
from django.conf.urls.static import static
app_name = "reports"  

urlpatterns = [
    path('reports/', view=views.report_view, name='traffic'),
    path('download/', views.download_csv_report, name='download_csv_report'),
    path('analytics/', views.analytics, name='analytics'),
    path('view/<str:result_id>/<str:crop_type>/<str:result_type>', views.result_detail, name='view_predictions'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)


urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
