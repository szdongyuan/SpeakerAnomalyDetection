"""Spawn entry point: initialize once, execute tasks serially, then wait again."""

import logging
import os

from .runtime_protocol import (
    RunTask, StopWorker, WorkerEvent, failed_task, runtime_error,
)


def analysis_worker_main(connection, station_id, generation):
    logger = logging.getLogger(__name__)

    def emit(kind, task_id=None, payload=None):
        connection.send(WorkerEvent(
            kind, station_id, generation, task_id, payload or {},
        ))

    try:
        # Loading the management module in the parent must not load algorithms.
        from .executor import execute_task

        emit("ready", payload={"pid": os.getpid()})
        while True:
            command = connection.recv()
            if isinstance(command, StopWorker):
                emit("stopped")
                return
            if not isinstance(command, RunTask):
                raise TypeError("Unknown analysis worker command")
            task = command.task
            if task.station_id != station_id:
                raise ValueError("Task station does not match worker station")
            emit("task_started", task.task_id)
            try:
                result = execute_task(task)
            except Exception as error:
                # Last boundary around the executor, not a silent retry.
                logger.exception("Unhandled analysis task failure: %s", task.task_id)
                result = failed_task(task, runtime_error(
                    "execution", type(error).__name__, error,
                ))
            emit("task_result", task.task_id, result)
            # A resident idle worker must not retain the last audio result.
            del result, task, command
    except EOFError:
        # The owning parent closed the command channel.
        return
    except Exception as error:
        logger.exception("Analysis worker failed for station %s", station_id)
        try:
            emit("worker_error", payload=runtime_error(
                "worker", type(error).__name__, error,
            ))
        except (OSError, EOFError):
            logger.exception("Cannot send worker failure to parent")
    finally:
        connection.close()
