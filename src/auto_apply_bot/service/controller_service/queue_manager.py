import threading
import queue
import uuid
import traceback
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from auto_apply_bot.service.controller_service.controller_service import get_controller
from auto_apply_bot.logger import get_logger


logger = get_logger(__name__)


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class JobResult:
    status: JobStatus
    result: Optional[Any] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "result": self.result,
            "error": self.error
        }


class ControllerQueueManager:
    def __init__(self):
        self.job_queue: queue.Queue = queue.Queue()
        self.results: dict[str, JobResult] = {}
        self._lock = threading.Lock()
        logger.info("Starting ControllerQueueManager worker thread")
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_job(
        self,
        fn_name: str,
        args: list = None,
        kwargs: dict = None,
        timeout_sec: Optional[int] = 120,
    ) -> str:
        job_id = str(uuid.uuid4())
        with self._lock:
            self.results[job_id] = JobResult(status=JobStatus.PENDING)
        self.job_queue.put((job_id, fn_name, args or [], kwargs or {}, timeout_sec))
        logger.info(f"Submitted job {job_id} to function {fn_name} (timeout: {timeout_sec} seconds)")
        return job_id

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        with self._lock:
            return self.results.get(job_id, JobResult(status=JobStatus.FAILED)).status

    def get_job_result(self, job_id: str) -> Optional[Any]:
        with self._lock:
            return self.results.get(job_id, JobResult(status=JobStatus.FAILED)).result

    def get_job_error(self, job_id: str) -> Optional[str]:
        with self._lock:
            return self.results.get(job_id, JobResult(status=JobStatus.FAILED)).error

    def _worker_loop(self):
        while True:
            job_id, fn_name, args, kwargs, timeout_sec = self.job_queue.get()

            def job_wrapper():
                try:
                    controller = get_controller()
                    target = controller
                    for attr in fn_name.split("."):
                        target = getattr(target, attr)
                    result = target(*args, **kwargs)
                    with self._lock:
                        if self.results[job_id].status == JobStatus.RUNNING:
                            self.results[job_id] = JobResult(JobStatus.COMPLETED, result=result)
                except Exception as e:
                    error_msg = traceback.format_exc()
                    logger.error(f"Job {job_id} failed:\n{error_msg}")
                    with self._lock:
                        self.results[job_id] = JobResult(JobStatus.FAILED, error=error_msg)
                finally:
                    self.job_queue.task_done()

            with self._lock:
                self.results[job_id].status = JobStatus.RUNNING

            thread = threading.Thread(target=job_wrapper, daemon=True)
            thread.start()

            if timeout_sec:
                timer = threading.Timer(timeout_sec, self._handle_timeout, args=[job_id])
                timer.start()
                thread.join(timeout_sec)
                timer.cancel()
            else:
                thread.join()

    def _handle_timeout(self, job_id: str):
        with self._lock:
            if self.results.get(job_id) and self.results[job_id].status == JobStatus.RUNNING:
                self.results[job_id] = JobResult(
                    status=JobStatus.TIMEOUT,
                    error="Job timed out."
                )
                logger.warning(f"Job {job_id} timed out and was marked as TIMEOUT.")
        try:
            controller = get_controller()
            controller.cleanup()
            logger.info(f"Controller cleanup called after timeout on job {job_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup controller after timeout for job {job_id}: {e}")


# Singleton instance
controller_queue = ControllerQueueManager()
