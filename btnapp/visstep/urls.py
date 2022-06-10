from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('month/', views.month, name='month'),
    path('year/', views.year, name='year'),
]