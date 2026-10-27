from __future__ import annotations

from types import MappingProxyType
from typing import cast

import pytest
from e7awghal import E7awgCaptureDataError
from quel_ic_config import (
    AwgAbandonedError,
    CaptureAbandonedError,
    DeviceStalledError,
    Quel1Box,
    TimedStartTooLateError,
)

from qxdriver_quel1.driver import single


class _FailedTask:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def result(self) -> None:
        raise self._error


def _abandoned_by(abandoned: type[Exception], cause: Exception) -> Exception:
    error = abandoned("abandoned")
    error.__cause__ = cause
    return error


_awg_late = TimedStartTooLateError("late")
_capture_broken = E7awgCaptureDataError("broken")
_awg_stalled = DeviceStalledError("awg stalled")
_capture_stalled = DeviceStalledError("capture stalled")


@pytest.mark.parametrize(
    ("awg_error", "capture_error", "expected"),
    [
        (_awg_late, _abandoned_by(CaptureAbandonedError, _awg_late), _awg_late),
        (
            _abandoned_by(AwgAbandonedError, _capture_broken),
            _capture_broken,
            _capture_broken,
        ),
        (_awg_stalled, _capture_stalled, _capture_stalled),
    ],
    ids=["awg_failed", "capture_failed", "both_failed"],
)
def test_triggered_capture_stop_raises_failure_not_abandonment(
    awg_error: Exception, capture_error: Exception, expected: Exception
) -> None:
    """Given a failed triggered shot, the failure is raised rather than the abandonment it causes."""
    action = single.Action(
        cast(Quel1Box, None),
        MappingProxyType({}),
        MappingProxyType({}),
        MappingProxyType({}),
    )
    futures: dict[single.CaptureFutureKey, object] = {
        "__triggered__": (_FailedTask(capture_error), _FailedTask(awg_error))
    }

    with pytest.raises(type(expected)) as exc_info:
        action.capture_stop(futures)

    assert exc_info.value is expected
