"""Instance-owned serial audio label writes, independent of Qt."""

import logging
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread

from base.audio_label_repository import AudioLabelPersistenceError


@dataclass(frozen=True)
class AudioLabelJob:
    task_id: str
    wav_path: str
    label: str


class AudioLabelSubmissionError(RuntimeError):
    """A label job could not be accepted because its worker failed to start."""


class AudioLabelSyncService:
    def __init__(self, repository, completed_callback):
        self._repository = repository
        # The owner supplies a queued signal emitter, never a UI consumer.
        self._completed = completed_callback
        self._jobs = Queue()
        self._thread = None
        self._lock = Lock()
        self._closing = False

    def submit(self, job):
        with self._lock:
            if self._closing:
                raise RuntimeError("Audio label service is closing")
            if self._thread is None:
                try:
                    thread = Thread(target=self._run, name="audio-label-sync", daemon=False)
                    thread.start()
                except (RuntimeError, OSError) as error:
                    # Nothing was enqueued, so a failed start leaves no work to drain.
                    raise AudioLabelSubmissionError(
                        f"Cannot start audio label worker for task {job.task_id}: {error}"
                    ) from error
                self._thread = thread
            self._jobs.put(job)

    def begin_shutdown(self):
        with self._lock:
            if not self._closing:
                self._closing = True
                if self._thread is not None:
                    self._jobs.put(None)

    @property
    def closed(self):
        with self._lock:
            return self._closing and (self._thread is None or not self._thread.is_alive())

    def _run(self):
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                update = self._repository.update(job.wav_path, job.label)
            except AudioLabelPersistenceError as error:
                self._completed(job, None, error)
            except Exception as error:
                # Injected repository implementations may fail unexpectedly. Finish
                # this job with a diagnostic and keep later accepted writes running.
                logging.getLogger("core").exception(
                    "Unexpected audio label job failure: %s", job.task_id
                )
                self._completed(job, None, error)
            else:
                self._completed(job, update, None)
