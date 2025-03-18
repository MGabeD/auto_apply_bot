from django.urls import path
from . import views

urlpatterns = [
    path('hello/', views.hello_world, name='hello_world'),
    path('query-rag/', views.query_rag, name='query_rag')
]