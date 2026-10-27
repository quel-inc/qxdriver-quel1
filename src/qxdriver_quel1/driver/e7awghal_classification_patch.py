"""Scoped e7awghal patch for raw DSP classification line registers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from threading import RLock
from typing import Any

import numpy as np
import numpy.typing as npt

from qxdriver_quel1.classification import LineParam, normalize_line_param

RAW_CLASSIFICATION_LINES_ATTR = "__qxdriver_raw_classification_lines__"

_PATCH_LOCK = RLock()


def _line_to_reg_half(
    line: LineParam,
    total_exponent_offset: int,
) -> npt.NDArray[np.float32]:
    """Convert one raw line equation into the register half used by e7awghal."""
    a, b, c = normalize_line_param(line)
    coeff_max = max(abs(a), abs(b))
    scale = 32767.0 / coeff_max
    return np.asarray(
        [
            a * scale,
            b * scale,
            c * scale * float(1 << total_exponent_offset),
        ],
        dtype=np.float32,
    )


class _CompatClassificationRegFile:
    """
    Stand-in for `e7awghal.capunit._CapParamClassificationRegFile`.

    The e7awghal of quel_ic_config 0.11 builds the classification registers
    without the class. This temporary layer puts it back and routes the
    register building through it, so that the scoped patch below keeps
    working. Remove it when e7awghal accepts raw line equations.
    """

    def __init__(self) -> None:
        self.p0 = np.zeros(3, dtype=np.float32)
        self.p1 = np.zeros(3, dtype=np.float32)

    @classmethod
    def fromcapparam(cls, cp: Any) -> _CompatClassificationRegFile:
        from e7awghal.classification import calc_abc_main_sub

        reg_file = cls()
        reg_file.p0, reg_file.p1 = calc_abc_main_sub(
            cp.classification_param.pivot_x,
            cp.classification_param.pivot_y,
            cp.classification_param.angle_main,
            cp.classification_param.angle_sub,
            cp.total_exponent_offset(),
        )
        return reg_file


def _install_compat_classification_reg_file() -> None:
    try:
        from e7awghal import capunit
    except ImportError:  # pragma: no cover - depends on quelware extras
        return
    if hasattr(capunit, "_CapParamClassificationRegFile"):
        return
    original_build = getattr(capunit, "_build_classification_regs", None)
    param_half = getattr(capunit, "_ClassificationParamHalf", None)
    if original_build is None or param_half is None:
        return

    @wraps(original_build)
    def _build_classification_regs(cp: Any) -> npt.NDArray[np.uint32]:
        reg_file = _CompatClassificationRegFile.fromcapparam(cp)
        param_half.validate_python(reg_file.p0)
        param_half.validate_python(reg_file.p1)
        t = np.hstack((reg_file.p0, reg_file.p1)).astype(np.float32)
        return np.frombuffer(t.tobytes(), dtype=np.uint32)

    target: Any = capunit
    target._CapParamClassificationRegFile = _CompatClassificationRegFile
    target._build_classification_regs = _build_classification_regs


_install_compat_classification_reg_file()


def _classification_reg_file_class() -> type:
    try:
        from e7awghal import capunit
    except ImportError as exc:  # pragma: no cover - depends on quelware extras
        raise RuntimeError(
            "e7awghal is required for raw DSP classification line conversion."
        ) from exc

    classification_reg_file_cls = getattr(
        capunit,
        "_CapParamClassificationRegFile",
        None,
    )
    if classification_reg_file_cls is None:
        raise RuntimeError(
            "e7awghal.capunit._CapParamClassificationRegFile is required for "
            "raw DSP classification line conversion."
        )
    return classification_reg_file_cls


@contextmanager
def patched_classification_register_builder(
    *,
    required: bool,
) -> Iterator[None]:
    """
    Temporarily patch e7awghal classification register generation.

    e7awghal normally derives classification registers from pivot/angle values.
    That conversion cannot preserve parallel separated decision lines, so the
    direct driver stores raw line equations on `CapParam` and this scoped patch
    writes those equations to the private register file while `config_runit`
    builds hardware registers.
    """
    if not required:
        yield
        return

    with _PATCH_LOCK:
        classification_reg_file_cls = _classification_reg_file_class()
        original_fromcapparam = classification_reg_file_cls.fromcapparam

        def _classification_fromcapparam(cls: type, cp: Any) -> Any:
            lines = getattr(cp, RAW_CLASSIFICATION_LINES_ATTR, None)
            if lines is None:
                return original_fromcapparam(cp)
            line0, line1 = lines
            reg_file = cls()
            total_exponent_offset = cp.total_exponent_offset()
            reg_file.p0 = _line_to_reg_half(line0, total_exponent_offset)
            reg_file.p1 = _line_to_reg_half(line1, total_exponent_offset)
            return reg_file

        classification_reg_file_cls.fromcapparam = classmethod(
            _classification_fromcapparam
        )
        try:
            yield
        finally:
            classification_reg_file_cls.fromcapparam = original_fromcapparam
