from __future__ import annotations

import inspect

import numpy as np
import pytest
from e7awghal import CapParam, capunit
from e7awghal.classification import ClassificationParam

from qxdriver_quel1.driver.compat import convert_captureparam
from qxdriver_quel1.driver.e7awghal_classification_patch import (
    patched_classification_register_builder,
)
from qxdriver_quel1.e7awg.compat import CaptureParam, DspUnit


def _as_float32(regs: np.ndarray) -> list[float]:
    return np.frombuffer(regs.tobytes(), dtype=np.float32).tolist()


def test_raw_lines_reach_classification_registers() -> None:
    """Given raw decision lines, the registers built by e7awghal hold them."""
    cprm = CaptureParam()
    cprm.add_sum_section(12, num_post_blank_words=4)
    cprm.sel_dsp_units_to_enable(DspUnit.CLASSIFICATION)
    for func_sel, c in ((0, -2.0), (1, -3.0)):
        cprm.set_decision_func_params(
            func_sel=func_sel,
            coef_a=np.float32(1.0),
            coef_b=np.float32(0.0),
            const_c=np.float32(c),
        )
    converted = convert_captureparam(cprm)

    with patched_classification_register_builder(required=True):
        regs = capunit._build_classification_regs(converted)

    assert _as_float32(regs) == pytest.approx(
        [32767.0, 0.0, -65534.0, 32767.0, 0.0, -98301.0]
    )


def test_registers_without_raw_lines_are_unchanged() -> None:
    """Given no raw lines, the registers equal those of e7awghal itself."""
    cp = CapParam(
        classification_enable=True,
        classification_param=ClassificationParam(
            pivot_x=100.0, pivot_y=-50.0, angle_main=30.0, angle_sub=120.0
        ),
    )
    original = inspect.unwrap(capunit._build_classification_regs)

    assert capunit._build_classification_regs(cp).tolist() == original(cp).tolist()
