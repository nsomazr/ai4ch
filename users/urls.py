from django.urls import path, include
from .views import UsersAPIView,MyTokenObtainPairView
from rest_framework_simplejwt.views import TokenRefreshView
from . import views
from django.conf import settings
from django.conf.urls.static import static

app_name = "users"   

urlpatterns = [
    path('users/', UsersAPIView.as_view()),
    path('login_token/', MyTokenObtainPairView.as_view()),
    path('token/refresh/', TokenRefreshView.as_view()),
    path("dashboard/", views.dashboard, name="dashboard"),
    path("logout/", views.logout_request, name= "logout"),
    path("login/", views.login_request, name="login"),
    path("add_staff/", views.add_staff, name="add-staff"),
        path("staffs/", views.staffs, name="staffs"),
    path("register/", views.register_request, name="register"),
    path("password_reset/", views.password_reset_request, name="password_reset"),
    
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
