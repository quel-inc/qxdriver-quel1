from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import CancelledError
from concurrent.futures import TimeoutError as FutureTimeoutError
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from qxdriver_quel1.driver import multi, single


class _Window:
    """A shot that has ended with `error`, or runs until it is cancelled."""

    def __init__(self, error: Exception | None = None, *, running: bool) -> None:
        self._error: BaseException | None = error
        self._running = running

    def done(self) -> bool:
        return not self._running

    def exception(self, timeout: float | None = None) -> BaseException | None:
        if self._running:
            raise FutureTimeoutError()
        return self._error

    def result(self, timeout: float | None = None) -> dict[Any, Any]:
        if self._running:
            raise FutureTimeoutError()
        if self._error is not None:
            raise self._error
        return {}

    def cancel(self, timeout: float | None = None) -> bool:
        if not self._running:
            return False
        self._running = False
        self._error = CancelledError()
        return True


class _Box:
    """Starts running shots; fails the start named, or ends the shot named with a failure."""

    def __init__(self, fail_start: str | None = None, fail_shot: str | None = None):
        self._fail_start = fail_start
        self._fail_shot = fail_shot
        self.failure = RuntimeError(f"{fail_start or fail_shot} failed")
        self.windows: list[_Window] = []

    def _start(self, kind: str) -> _Window:
        if kind == self._fail_start:
            raise self.failure
        if kind == self._fail_shot:
            window = _Window(self.failure, running=False)
        else:
            window = _Window(running=True)
        self.windows.append(window)
        return window

    def start_capture_now(self, runits: Any) -> _Window:
        return self._start(f"capture_now:{min(runits)[0]}")

    def start_capture_by_awg_trigger(
        self, runits: Any, channels: Any, timecounter: int | None = None
    ) -> tuple[_Window, _Window]:
        return self._start("trigger_capture"), self._start("trigger_awg")

    def start_wavegen(self, channels: Any, timecounter: int | None = None) -> _Window:
        return self._start("wavegen")

    def get_current_timecounter(self) -> int:
        return 0

    def get_latest_sysref_timecounter(self) -> int:
        return 0


_AWG = single.AwgId(port=1, channel=0)


def _capture_action(box: _Box, ports: tuple[int, ...]) -> single.Action:
    return single.Action(
        cast(Any, box),
        MappingProxyType({}),
        MappingProxyType(
            {single.RunitId(port=p, runit=0): cast(Any, None) for p in ports}
        ),
        MappingProxyType({}),
    )


def _triggered_action(box: _Box) -> single.Action:
    return single.Action(
        cast(Any, box),
        MappingProxyType({_AWG: cast(Any, None)}),
        MappingProxyType({single.RunitId(port=0, runit=0): cast(Any, None)}),
        MappingProxyType({0: _AWG}),
    )


def _emission_action(box: _Box) -> single.Action:
    return single.Action(
        cast(Any, box),
        MappingProxyType({_AWG: cast(Any, None)}),
        MappingProxyType({}),
        MappingProxyType({}),
    )


def _multi_action(actions: dict[str, single.Action]) -> multi.Action:
    system = SimpleNamespace(
        displacement=0,
        timing_shift=dict.fromkeys(actions, 0),
        box={name: action.box for name, action in actions.items()},
    )
    return multi.Action(
        cast(Any, system),
        MappingProxyType(actions),
        MappingProxyType(dict.fromkeys(actions, 0)),
        next(iter(actions)),
        0,
    )


def _single_start(boxes: list[_Box]) -> Callable[[], object]:
    return _capture_action(boxes[0], (0, 1)).capture_start


def _single_stop(boxes: list[_Box]) -> Callable[[], object]:
    action = _capture_action(boxes[0], (0, 1))
    return lambda: action.capture_stop(action.capture_start())


def _multi(boxes: list[_Box]) -> Callable[[], object]:
    return _multi_action(
        {
            "a": _triggered_action(boxes[0]),
            "b": _capture_action(boxes[1], (0,)),
            "c": _emission_action(boxes[2]),
            "d": _triggered_action(boxes[3]),
            "e": _emission_action(boxes[4]),
        }
    ).action


@pytest.mark.parametrize(
    ("run", "failing", "failure"),
    [
        (_single_start, 0, {"fail_start": "capture_now:1"}),
        (_single_stop, 0, {"fail_shot": "capture_now:1"}),
        (_multi, 1, {"fail_start": "capture_now:0"}),
        (_multi, 4, {"fail_start": "wavegen"}),
        (_multi, 1, {"fail_shot": "capture_now:0"}),
        (_multi, 2, {"fail_shot": "wavegen"}),
        (_multi, 3, {"fail_shot": "trigger_capture"}),
    ],
    ids=[
        "single_start",
        "single_wait",
        "multi_capture_start",
        "multi_emission_start",
        "multi_capture_wait",
        "multi_emission_wait",
        "multi_triggered_wait",
    ],
)
def test_failure_leaves_no_shot_running(
    run: Callable[[list[_Box]], Callable[[], object]],
    failing: int,
    failure: dict[str, str],
) -> None:
    """Given a failure of one start or one shot, the shots still running are cancelled and the failure is raised."""
    boxes = [_Box(**failure) if i == failing else _Box() for i in range(5)]

    with pytest.raises(RuntimeError) as exc_info:
        run(boxes)()

    assert exc_info.value is boxes[failing].failure
    windows = [window for box in boxes for window in box.windows]
    assert windows
    assert all(window.done() for window in windows)
