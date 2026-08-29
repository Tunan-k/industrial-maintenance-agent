"""
功能：
    创建并缓存 DiagnosisService，避免模型反复加载。
因为模型服务初始化会：创建网络，读取 checkpoint，加载参数，不能每调用一次 Tool 都重新做一次。
"""

###### V2版本
from functools import lru_cache
from pathlib import Path

from app.services.diagnosis_service import (
    DiagnosisService,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)


ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "drilling_pump"
    / "90spm"
)


CHECKPOINT_PATH = (
    ARTIFACT_DIR
    / "best_model10_dsmt1dcnn_pump10.pt"
)


SCALER_PATH = (
    ARTIFACT_DIR
    / "scaler.joblib"
)


CLASS_MAP_PATH = (
    ARTIFACT_DIR
    / "class_map_10.json"
)

#过缓存的 service factory 保证单进程内模型只初始化一次，避免每次 Tool/API 请求重复加载 checkpoint 带来的延迟和内存开销
@lru_cache(maxsize=1)
def get_diagnosis_service() -> DiagnosisService:
    """
    Create the diagnosis service only once per Python process.

    The cached service keeps the PyTorch model loaded in memory.
    """

    return DiagnosisService(
        checkpoint_path=CHECKPOINT_PATH,
        scaler_path=SCALER_PATH,
        class_map_path=CLASS_MAP_PATH,
    )

