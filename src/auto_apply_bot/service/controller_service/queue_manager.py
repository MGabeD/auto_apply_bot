import multiprocessing
import queue
import uuid
import traceback
import time
from enum import Enum
from dataclasses import dataclass
from typing import Any, Optional
from auto_apply_bot.service.controller_service.controller_service import get_controller
from auto_apply_bot.utils.logger import get_logger


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


def resolve_controller_callable(fn_path: str):
    """
    Resolves a dot-delimited path to a callable from the Controller instance.
    Raises ValueError if path is invalid or not callable.
    """
    controller = get_controller()
    target = controller
    for attr in fn_path.split("."):
        if not hasattr(target, attr):
            raise ValueError(f"Attribute '{attr}' not found in path '{fn_path}'")
        target = getattr(target, attr)
    if not callable(target):
        raise ValueError(f"Resolved path '{fn_path}' is not callable.")
    return target


def process_target(fn_name, args, kwargs, result_pipe):
    try:
        controller = get_controller()  
        target = controller
        for attr in fn_name.split("."):
            target = getattr(target, attr)
        result = target(*args, **kwargs)
        result_pipe.send(("completed", result, None))
    except Exception:
        error_msg = traceback.format_exc()
        result_pipe.send(("failed", None, error_msg))
    finally:
        try:
            controller.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup in subprocess failed: {e}")
        result_pipe.close()


class ControllerQueueManager:
    def __init__(self):
        self.job_queue: queue.Queue = queue.Queue()
        self.results: dict[str, JobResult] = {}
        self._lock = multiprocessing.Lock()
        logger.info("Starting ControllerQueueManager worker thread")
        self.thread = multiprocessing.Process(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_job(
        self,
        fn_name: str,
        args: list = None,
        kwargs: dict = None,
        timeout_sec: Optional[int] = 120,
    ) -> str:
        job_id = str(uuid.uuid4())
        try:
            resolve_controller_callable(fn_name)
        except ValueError as e:
            logger.error(f"Job {job_id} submission failed: Invalid function path: {fn_name}")
            raise
        with self._lock:
            self.results[job_id] = JobResult(status=JobStatus.PENDING)
        self.job_queue.put((job_id, fn_name, args or [], kwargs or {}, timeout_sec))
        logger.info(f"Submitted job {job_id} to function {fn_name} (timeout: {timeout_sec} seconds)")
        return job_id
    
    def run_job(self, fn_name: str, args: list = None, kwargs: dict = None, timeout_sec: int = 120) -> JobResult:
        job_id = self.submit_job(fn_name, args=args, kwargs=kwargs, timeout_sec=timeout_sec)
        
        start = time.time()
        while True:
            with self._lock:
                status = self.results[job_id].status
                if status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.TIMEOUT):
                    return self.results[job_id]

            if time.time() - start > timeout_sec + 1:
                return JobResult(status=JobStatus.TIMEOUT, error="Timed out waiting for job.")
            time.sleep(0.1)

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

            parent_conn, child_conn = multiprocessing.Pipe()
            process = multiprocessing.Process(
                target=process_target,
                args=(fn_name, args, kwargs, child_conn),
                daemon=True
            )

            with self._lock:
                self.results[job_id].status = JobStatus.RUNNING

            process.start()
            process.join(timeout=timeout_sec)

            if process.is_alive():
                process.terminate()
                process.join()
                with self._lock:
                    self.results[job_id] = JobResult(
                        status=JobStatus.TIMEOUT,
                        error="Job timed out and was terminated."
                    )
                logger.warning(f"Job {job_id} timed out.")
            else:
                try:
                    status_str, result, error = parent_conn.recv()
                    status = JobStatus.COMPLETED if status_str == "completed" else JobStatus.FAILED
                    with self._lock:
                        self.results[job_id] = JobResult(status=status, result=result, error=error)
                except EOFError:
                    with self._lock:
                        self.results[job_id] = JobResult(status=JobStatus.FAILED, error="No result returned.")
                    logger.error(f"Job {job_id} process ended without sending a result.")

            parent_conn.close()
            self.job_queue.task_done()


# Singleton instance
controller_queue = ControllerQueueManager()
