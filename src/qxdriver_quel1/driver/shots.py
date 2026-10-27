"""Waiting for the shots started by one action as a bundle."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import CancelledError, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from logging import getLogger
from typing import Any, Final

from quel_ic_config import TaskAbandonedError

logger = getLogger(__name__)

_POLLING_PERIOD: Final[float] = 0.01


def cancel_windows(windows: Sequence[Any]) -> None:
    """
    Cancel the shots of the windows at once and wait for their ends.

    A window that has ended already is left as it is. A failure of the
    cancellation, e.g., `TimedStartCancelUncertainError`, is logged, since it
    is called before another failure is raised.
    """
    if not windows:
        return
    with ThreadPoolExecutor(max_workers=len(windows)) as executor:
        for _ in executor.map(_cancel_window, windows):
            pass


def _cancel_window(window: Any) -> None:
    try:
        window.cancel()
    except Exception:
        logger.warning("failed to cancel the shot of %s", window, exc_info=True)


def wait_windows(windows: Sequence[Any]) -> None:
    """
    Wait for the ends of the shots of the windows, and raise one failure if any.

    When a window has failed, the shots of the others not ended yet are
    cancelled at once, so that no shot started by the action is left running.
    The failure is raised after all of them have ended. It is the first one in
    the order of the windows that is neither `TaskAbandonedError` nor a
    cancellation, or else the first abandonment or cancellation; put the
    capture window of a shot before its AWG window, so that a failure of the
    capture, e.g., `E7awgCaptureDataError`, is raised as it is. When the wait
    is interrupted, the shots not ended yet are cancelled too.

    The windows are watched with `done()`, which does not access the devices,
    every 10 ms.
    """
    pending = list(windows)
    failed = False
    try:
        while pending:
            ended = [window for window in pending if window.done()]
            if not ended:
                _wait_a_while(pending[0])
                continue
            ended_ids = {id(window) for window in ended}
            pending = [window for window in pending if id(window) not in ended_ids]
            if not failed and any(_error_of(window) for window in ended):
                failed = True
                cancel_windows(pending)
    except BaseException:
        cancel_windows(pending)
        raise
    failure = _failure_to_raise([_error_of(window) for window in windows])
    if failure is not None:
        raise failure


def _wait_a_while(window: Any) -> None:
    try:
        window.exception(timeout=_POLLING_PERIOD)
    except (FutureTimeoutError, CancelledError):
        pass


def _error_of(window: Any) -> BaseException | None:
    try:
        return window.exception()
    except CancelledError as e:
        return e


def _failure_to_raise(errors: Sequence[BaseException | None]) -> BaseException | None:
    def _rank(error: BaseException) -> int:
        if isinstance(error, CancelledError):
            return 2
        if isinstance(error, TaskAbandonedError):
            return 1
        return 0

    failures = [error for error in errors if error is not None]
    return min(failures, key=_rank, default=None)
