from __future__ import annotations

from pathlib import Path
import sys


EFFICIENTSAM3_ROOT = Path("/slow_disk/ccl/codes/efficientsam3")
EFFICIENTSAM3_PACKAGE = EFFICIENTSAM3_ROOT / "sam3"
DEFAULT_SOURCE_CHECKPOINT = EFFICIENTSAM3_ROOT / "download/efficient_sam3_tinyvit_s.pt"


def activate_efficientsam3() -> None:
    """Make the read-only EfficientSAM3 fork win over this repo's sam3 package."""
    package_path = str(EFFICIENTSAM3_PACKAGE)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    # Some reusable DETR experiment utilities import video-only helpers that
    # are absent from the EfficientSAM3 fork. Keep EfficientSAM3 authoritative
    # while allowing missing modules to fall back to this checkout, read-only.
    import sam3

    local_package = str(Path(__file__).resolve().parents[1] / "sam3")
    if local_package not in sam3.__path__:
        sam3.__path__.append(local_package)
    import sam3.model

    local_model_package = str(Path(local_package) / "model")
    if local_model_package not in sam3.model.__path__:
        sam3.model.__path__.append(local_model_package)
