import json
import os
import uuid
from pathlib import Path

from django.conf import settings
from django.http import JsonResponse
from django.utils.timezone import now
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.core.files.uploadedfile import TemporaryUploadedFile

from auto_apply_bot.service.controller_service.queue_manager import controller_queue, JobStatus
from auto_apply_bot.utils.logger import get_logger
from auto_apply_bot.utils.path_sourcing import ensure_path_is_dir_or_create
from auto_apply_bot.utils.file_validation import validate_file
from auto_apply_bot import ALLOWED_EXTENSIONS 


logger = get_logger(__name__)


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TOTAL_FILE_SIZE = 50 * 1024 * 1024  # 50MB


@ensure_path_is_dir_or_create
def get_upload_dir() -> Path:
    return Path(settings.SERVICE_DIR) / 'uploads'


@csrf_exempt  
@require_POST
def upload_files_view(request):
    files = request.FILES.getlist("files")
    total_size = sum(f.size for f in files)
    if total_size > MAX_TOTAL_FILE_SIZE:
        return JsonResponse({
            "success": False,
            "error": f"Total upload exceeds {MAX_TOTAL_FILE_SIZE // (1024 * 1024)}MB limit.",
            "files": []
        }, status=400)

    upload_dir = get_upload_dir()
    file_results = []
    for file in files:
        result = {
            "original_name": file.name,
            "size": file.size,
        }
        try:
            is_valid, reason = validate_file(file, ALLOWED_EXTENSIONS, MAX_FILE_SIZE)
            if not is_valid:
                result["status"] = "rejected"
                result["reason"] = reason
                if isinstance(file, TemporaryUploadedFile):
                    os.remove(file.temporary_file_path())
                else:
                    file.close()
            else:
                unique_name = f"{uuid.uuid4().hex}{Path(file.name).suffix.lower()}"
                save_path = upload_dir / unique_name
                with open(save_path, "wb+") as dest:
                    for chunk in file.chunks():
                        dest.write(chunk)

                result.update({
                    "status": "uploaded",
                    "stored_as": unique_name,
                    "timestamp": now().isoformat(),
                })
        except Exception as e:
            logger.error(f"Error processing file {file.name}: {e}")
            result["status"] = "error"
            result["reason"] = str(e)
            try:
                if isinstance(file, TemporaryUploadedFile):
                    os.remove(file.temporary_file_path())
                else:
                    file.close()
            except Exception as e:
                logger.error(f"Error closing file {file.name}: {e}")
                pass

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
    

