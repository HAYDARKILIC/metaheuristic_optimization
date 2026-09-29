"""Tiny assertion helpers so every notebook validates what it builds."""

import numpy as np


def check_close(actual, expected, name="value", atol=1e-6, rtol=1e-6):
    ok = np.allclose(actual, expected, atol=atol, rtol=rtol)
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got {np.round(actual, 6)}, expected {np.round(expected, 6)}")
    assert ok, f"{name} mismatch"


def check_true(condition, name="condition"):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}")
    assert condition, name
