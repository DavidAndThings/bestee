"""Probe the host machine's compute capabilities.

The probe is intentionally dependency-light: CPU/RAM/disk come from the
standard library, and the accelerator is detected by reusing
:func:`bestee_chat.embeddings.detect_device` (which knows about CUDA and Apple
Silicon / MPS). If ``torch`` is not installed the probe degrades gracefully to
a CPU-only profile instead of raising.

The resulting :class:`HardwareProfile` is a plain record, so profile-selection
logic can be unit-tested with synthetic profiles and no real hardware.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

_BYTES_PER_GB = 1024**3

#: Relative one-time build throughput by accelerator class. Querying never
#: depends on these; they only gate how heavy an index is worth *building*.
ACCEL_THROUGHPUT = {"cpu": 1.0, "mps": 4.0, "cuda": 12.0}


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """A snapshot of the machine's relevant compute resources.

    Attributes
    ----------
    cores:
        Logical CPU count.
    ram_gb:
        Total system RAM in GiB.
    free_disk_gb:
        Free space in GiB at the probed cache location.
    accel:
        Best available accelerator: ``"cuda"``, ``"mps"`` or ``"cpu"``.
    torch_available:
        Whether ``torch`` could be imported (gates semantic profiles).
    """

    cores: int
    ram_gb: float
    free_disk_gb: float
    accel: str
    torch_available: bool

    @property
    def throughput_weight(self) -> float:
        """Relative embedding-build throughput for :attr:`accel`."""
        return ACCEL_THROUGHPUT.get(self.accel, 1.0)

    @property
    def unified_memory(self) -> bool:
        """True on Apple Silicon, where RAM doubles as GPU-accessible VRAM."""
        return self.accel == "mps"


def _total_ram_gb() -> float:
    """Best-effort total RAM in GiB using stdlib ``sysconf`` where available."""
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except ValueError, OSError, AttributeError:
        return 0.0
    return (pages * page_size) / _BYTES_PER_GB


def _detect_accelerator() -> tuple[str, bool]:
    """Return ``(accel, torch_available)`` without hard-depending on torch."""
    try:
        from bestee_chat.embeddings import detect_device
    except Exception:
        return "cpu", False
    try:
        return detect_device(), True
    except Exception:
        return "cpu", False


def probe_hardware(cache_dir: str | os.PathLike[str] | None = None) -> HardwareProfile:
    """Inspect the current machine and return a :class:`HardwareProfile`.

    ``cache_dir`` only influences the free-disk measurement; it falls back to
    the user's home directory when not given or not yet created.
    """
    disk_target = Path(cache_dir).expanduser() if cache_dir else Path.home()
    while not disk_target.exists() and disk_target != disk_target.parent:
        disk_target = disk_target.parent
    try:
        free_disk_gb = shutil.disk_usage(disk_target).free / _BYTES_PER_GB
    except OSError:
        free_disk_gb = 0.0

    accel, torch_available = _detect_accelerator()
    return HardwareProfile(
        cores=os.cpu_count() or 1,
        ram_gb=round(_total_ram_gb(), 1),
        free_disk_gb=round(free_disk_gb, 1),
        accel=accel,
        torch_available=torch_available,
    )
