from django.urls import path
from . import views

urlpatterns = [
    # TODO these will likely be removed in future commits
    path('hello/', views.hello_world, name='hello_world'),
    path('query-rag/', views.query_rag, name='query_rag'),
    path('rag-up-docs/', views.upload_documents_rag, name="upload_documents_rag"),

    # MARK: direct job submission through the controller queue manager
    #  post and poll style endpoints
    path('job/submit/', views.submit_job_view, name='submit_job'),
    path('job/status/<str:job_id>/', views.job_status_view, name='job_status'),
    path('job/result/<str:job_id>/', views.job_result_view, name='job_result'),

    # submit and forget style endpoints
    path('job/run/', views.run_job_view, name='run_job'),
]