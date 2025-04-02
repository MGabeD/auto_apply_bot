from django.urls import path
from . import views

urlpatterns = [
    # MARK: Non-controller logic endpoints
    path('upload-rag-docs/', views.upload_rag_docs, name='upload_rag_docs'),

    # MARK: direct job submission through the controller queue manager
    #  post and poll style endpoints
    path('job/submit/', views.submit_job_view, name='submit_job'),
    path('job/status/<str:job_id>/', views.job_status_view, name='job_status'),
    path('job/result/<str:job_id>/', views.job_result_view, name='job_result'),

    # submit and forget style endpoints
    path('job/run/', views.run_job_view, name='run_job'),
]