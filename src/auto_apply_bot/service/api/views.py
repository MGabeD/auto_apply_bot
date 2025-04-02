import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from auto_apply_bot.service.controller_service.queue_manager import controller_queue, JobStatus
from auto_apply_bot.utils.logger import get_logger
from auto_apply_bot.utils.path_sourcing import ensure_path_is_dir_or_create


logger = get_logger(__name__)


ALLOWED_FILE_EXTENSIONS = ['.pdf', '.docx', '.doc', '.txt', '.json']
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TOTAL_FILE_SIZE = 50 * 1024 * 1024  # 50MB

@ensure_path_is_dir_or_create
def get_upload_dir() -> Path:
    return Path(settings.SERVICE_DIR) / 'uploads'


def validate_allowed_file(file) -> bool:
    return Path(file.name).suffix.lower() in ALLOWED_FILE_EXTENSIONS


@csrf_exempt  
@require_POST
def upload_files_view(request):
    files = request.FILES.getlist("files")
    total_size = sum(f.size for f in files)
    total_limit_bytes = MAX_TOTAL_FILE_SIZE

    if total_size > total_limit_bytes:
        return JsonResponse({
            "success": False,
            "error": f"Total upload exceeds {MAX_TOTAL_FILE_SIZE}MB limit.",
            "files": []
        }, status=400)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_results = []

    for file in files:
        result = {
            "original_name": file.name,
            "size": file.size,
        }

        if not validate_allowed_file(file):
            result["status"] = "rejected"
            result["reason"] = f"Unsupported file type: {Path(file.name).suffix.lower()}"
        elif file.size > MAX_FILE_SIZE:
            result["status"] = "rejected"
            result["reason"] = f"File exceeds {MAX_FILE_SIZE}MB limit."
        else:
            try:
                unique_name = f"{uuid.uuid4().hex}{Path(file.name).suffix.lower()}"
                save_path = UPLOAD_DIR / unique_name

                with open(save_path, "wb+") as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)

                result["status"] = "uploaded"
                result["stored_as"] = unique_name
                result["timestamp"] = now().isoformat()

            except Exception as e:
                result["status"] = "error"
                result["reason"] = str(e)

        file_results.append(result)

    return JsonResponse({
        "success": True,
        "results": file_results
    })


@csrf_exempt
def submit_job_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)
    
    try:
        body = json.loads(request.body)
        fn_name = body['fn_name']
        args = body.get('args', [])
        kwargs = body.get('kwargs', {})
        timeout = body.get('timeout_sec', 120)
        job_id = controller_queue.submit_job(fn_name, args, kwargs, timeout)
        return JsonResponse({'job_id': job_id}, status=202)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)


def job_status_view(request, job_id):
    status = controller_queue.get_job_status(job_id)
    return JsonResponse({'job_id': job_id, 'status': status.value})


def job_result_view(request, job_id):
    result = controller_queue.get_job_result(job_id)
    error = controller_queue.get_job_error(job_id)
    status = controller_queue.get_job_status(job_id)
    return JsonResponse({'job_id': job_id, 'status': status.value, 'result': result, 'error': error})


@csrf_exempt
def run_job_view(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed'}, status=405)

    try:
        body = json.loads(request.body)
        fn_name = body['fn_name']
        args = body.get('args', [])
        kwargs = body.get('kwargs', {})
        timeout = body.get('timeout_sec', 120)
        result = controller_queue.run_job(fn_name, args, kwargs, timeout)
        return JsonResponse(result.to_dict())
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=400)
    

