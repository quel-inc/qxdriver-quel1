from __future__ import annotations

from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError

import pytest
from e7awghal import E7awgCaptureDataError
from quel_ic_config import (
    AwgAbandonedError,
    CaptureAbandonedError,
    TaskAbortedError,
    TimedStartCancelUncertainError,
    TimedStartTooLateError,
)

from qxdriver_quel1.driver import shots


class _Window:
    """A shot that has ended with `error`, or runs until it is cancelled."""

    def __init__(
        self,
        error: BaseException | None = None,
        *,
        running: bool = False,
        cancel_error: Exception | None = None,
    ) -> None:
        self._error = error
        self._running = running
        self._cancel_error = cancel_error
        self.cancelled = False

    def done(self) -> bool:
        return not self._running

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if self._running:
            raise FutureTimeoutError()
        return self._error

    def cancel(self, timeout: float | None = None) -> bool:
        if not self._running:
            return False
        self._running = False
        self._error = CancelledError()
        self.cancelled = True
        if self._cancel_error is not None:
            raise self._cancel_error
        return True


def _abandoned_by(abandoned: type[Exception], cause: Exception) -> Exception:
    error = abandoned("abandoned")
    error.__cause__ = cause
    return error


_awg_late = TimedStartTooLateError("late")
_capture_broken = E7awgCaptureDataError("broken")
_capture_aborted = TaskAbortedError("capture aborted")
_awg_aborted = TaskAbortedError("awg aborted")


@pytest.mark.parametrize(
    ("capture_error", "awg_error", "expected"),
    [
        (_abandoned_by(CaptureAbandonedError, _awg_late), _awg_late, _awg_late),
        (
            _capture_broken,
            _abandoned_by(AwgAbandonedError, _capture_broken),
            _capture_broken,
        ),
        (_capture_aborted, _awg_aborted, _capture_aborted),
    ],
    ids=["awg_failed", "capture_failed", "both_aborted"],
)
def test_wait_windows_raises_failure_not_abandonment(
    capture_error: Exception, awg_error: Exception, expected: Exception
) -> None:
    """Given a failed triggered shot, the failure is raised rather than its abandonment, the capture first."""
    with pytest.raises(type(expected)) as exc_info:
        shots.wait_windows([_Window(capture_error), _Window(awg_error)])

    assert exc_info.value is expected


@pytest.mark.parametrize(
    "cancel_error",
    [None, TimedStartCancelUncertainError("uncertain")],
    ids=["cancelled", "cancel_uncertain"],
)
def test_wait_windows_cancels_running_shots_on_failure(
    cancel_error: Exception | None,
) -> None:
    """Given a failed shot among running ones, the others are cancelled and the failure is raised."""
    failure = TimedStartTooLateError("late")
    running = [
        _Window(running=True, cancel_error=cancel_error),
        _Window(running=True),
    ]

    with pytest.raises(TimedStartTooLateError) as exc_info:
        shots.wait_windows([running[0], _Window(failure), running[1]])

    assert exc_info.value is failure
    assert all(window.cancelled for window in running)
