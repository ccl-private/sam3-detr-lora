from __future__ import annotations

from pathlib import Path
import sys


EFFICIENTSAM3_ROOT = Path("/slow_disk/ccl/codes/efficientsam3")
EFFICIENTSAM3_PACKAGE = EFFICIENTSAM3_ROOT / "sam3"


def activate_efficientsam3() -> None:
    """只读加载相邻 EfficientSAM3 源码，并为缺失模块保留本地回退路径。"""
    package_path = str(EFFICIENTSAM3_PACKAGE)
    if package_path not in sys.path:
        sys.path.insert(0, package_path)
    import sam3

    local_package = str(Path(__file__).resolve().parents[1] / "sam3")
    if local_package not in sam3.__path__:
        sam3.__path__.append(local_package)
    import sam3.model

    local_model_package = str(Path(local_package) / "model")
    if local_model_package not in sam3.model.__path__:
        sam3.model.__path__.append(local_model_package)
