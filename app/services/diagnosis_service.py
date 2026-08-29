"""
功能：
    把 preprocessing + model inference 组合成一个完整业务能力
输入：
    file_path
    spm
    start_time
    end_time
输出：
    完整 diagnosis dict
"""

###### V2版本
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

from app.services.model_service import (
    DSMTModelService,
    ModelServiceError,
)

from app.services.preprocessing_service import (
    PreprocessingError,
    preprocess_file,
)


class DiagnosisServiceError(RuntimeError):
    """Unified error raised by the diagnosis application service."""


class DiagnosisService:
    """
    Application-level diagnosis service.

    It combines:
        raw file
        -> preprocessing
        -> model inference
        -> aggregated diagnosis

    This service is shared by:
        1. FastAPI
        2. Agent Diagnosis Tool
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        scaler_path: Union[str, Path],
        class_map_path: Union[str, Path],
    ):

        self.scaler_path = Path(
            scaler_path
        )

        self.model_service = DSMTModelService(
            checkpoint_path=checkpoint_path,
            class_map_path=class_map_path,
        )

    def diagnose_file(
        self,
        file_path: Union[str, Path],
        spm: int = 90,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> Dict:
        """
        Diagnose one uploaded/local industrial data file.

        V2.0 currently supports only drilling-pump 90SPM.
        """

        file_path = Path(file_path)

        if spm != 90:
            raise DiagnosisServiceError(
                "Current V2.0 diagnosis service supports only 90 SPM."
            )

        if not file_path.exists():
            raise DiagnosisServiceError(
                f"Input file does not exist: {file_path}"
            )

        if file_path.suffix.lower() not in {
            ".csv",
            ".xlsx",
        }:
            raise DiagnosisServiceError(
                "Unsupported file format. Use CSV or XLSX."
            )

        try:

            preprocessed = preprocess_file(
                file_path=file_path,
                scaler_path=self.scaler_path,
                start_time=start_time,
                end_time=end_time,
            )

            result = (
                self.model_service
                .predict_preprocessed(
                    preprocessed
                )
            )

            return result

        except PreprocessingError as exc:

            raise DiagnosisServiceError(
                f"Preprocessing failed: {exc}"
            ) from exc

        except ModelServiceError as exc:

            raise DiagnosisServiceError(
                f"Model inference failed: {exc}"
            ) from exc