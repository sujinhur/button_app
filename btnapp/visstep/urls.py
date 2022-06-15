from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('month/', views.month, name='month'),
    path('month/<int:num>/', views.month, name='month'),
    path('year/', views.year, name='year'),
    path('year/<int:num>/', views.year, name='year'),
    path('specify/', views.specify, name='specify'),
    path('compare/', views.compare, name='compare'),
]