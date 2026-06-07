"""Profile auto-selection from synthetic hardware (no real hardware/ML)."""

from __future__ import annotations

from bestee_chat.wiki.hardware import HardwareProfile
from bestee_chat.wiki.profiles import list_profiles, resolve_profile


def _hw(ram: float, accel: str, torch: bool = True) -> HardwareProfile:
    return HardwareProfile(
        cores=8, ram_gb=ram, free_disk_gb=500.0, accel=accel, torch_available=torch
    )


def test_no_torch_is_lexical() -> None:
    assert resolve_profile("auto", _hw(128, "cuda", torch=False)).name == "lexical"
    # Even an explicit semantic request downgrades when torch is missing.
    assert resolve_profile("quality", _hw(128, "cuda", torch=False)).name == "lexical"


def test_cpu_and_small_ram_is_economy() -> None:
    assert resolve_profile("auto", _hw(16, "cpu")).name == "economy"
    assert resolve_profile("auto", _hw(16, "mps")).name == "economy"


def test_midrange_accel_is_standard() -> None:
    assert resolve_profile("auto", _hw(32, "mps")).name == "standard"


def test_large_unified_memory_is_quality() -> None:
    # Apple Silicon (mps) cannot reach 'max' (cuda-gated) but reaches 'quality'.
    assert resolve_profile("auto", _hw(128, "mps")).name == "quality"
    assert resolve_profile("auto", _hw(64, "cuda")).name == "quality"


def test_big_cuda_box_is_max() -> None:
    assert resolve_profile("auto", _hw(128, "cuda")).name == "max"


def test_explicit_profile_is_honoured() -> None:
    assert resolve_profile("standard", _hw(16, "cpu")).name == "standard"
    assert set(list_profiles()) == {
        "lexical",
        "economy",
        "standard",
        "quality",
        "max",
    }
