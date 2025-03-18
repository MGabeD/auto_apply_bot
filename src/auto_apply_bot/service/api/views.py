from auto_apply_bot.logger import get_logger
from django.http import JsonResponse, HttpRequest
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
import os
from auto_apply_bot.retrieval_interface.retrieval import LocalRagIndexer


logger = get_logger(__name__)
rag = LocalRagIndexer()


def hello_world(request):
    return JsonResponse({"message": "Hello from Django"})

def query_rag(request):
    return JsonResponse({"message": "Hello from RAG"})

# TODO remove this csrf once I actually build everything out
@csrf_exempt
def upload_documents(request: HttpRequest):
    if request.method != "POST":
        return JsonResponse({"error": "Only POST method allowed"}, status=405)
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file(s) provided'}, status=400)

    files = request.FILES.getlist('file')
    statuses = []

    total_size = sum(f.size for f in files)
    if total_size > 50 * 1024 * 1024: # 50MB data max because I don't want to add more than that if I make a mistake
        return JsonResponse({'error': 'Total upload size exceeds 50MB limit'}, status=400)

    for file in files:
        ext = os.path.splitext(file.name)[1].lower()
        if not rag.is_allowed_file_type(file.name):
            logger.warning(f"Rejected file '{file.name}' due to invalid extensions")
            statuses.append({"file": file.name, "status": "error", "details": "Invalid file type"})
            continue

        if file.size > 10 * 1024 * 1024:
            logger.warning(f"Rejected file '{file.name}' due to exceeding 10MB limit")
            statuses.append({"file": file.name, "status": "error", "details": "File exceeds 10MB limit"})
            continue

        # local django project path not on the machine BASE_DIR but django's
        file_path = os.path.join(settings.BASE_DIR, "upload_files", file.name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with default_storage.open(file_path, 'wb+') as destination:
            for chunk in file.chunks():
                destination.write(chunk)

        try:
            rag.add_documents([str(file_path)])
            logger.info(f"File '{file.name}' successfully added to the RAG")
            statuses.append({"file": file.name, "status": "added"})

        except Exception as e:
            logger.error(f"Failed to add '{file.name}' to RAG: {e}", exc_info=True)
            statuses.append({"file": file.name, "status": "error", "details": str(e)})

        finally:
            try:
                os.remove(file_path)
                logger.info(f"Temporary file '{file_path}' removed after adding file to RAG")
            except Exception as cleanup_err:
                logger.warning(f"Failed cleanup file {file_path}: {cleanup_err}")

    return JsonResponse({'results': statuses})

