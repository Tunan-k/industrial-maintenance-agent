"""
功能：把 DiagnosisService 包成 LangChain Tool。
它不应该：自己切窗口；自己加载 checkpoint；自己算 softmax。

它只负责：
Agent Tool Schema
↓
DiagnosisService
↓
结构化 Tool Result
"""
###### V2版本
# from __future__ import annotations

# from typing import Optional

# from langchain.tools import tool

# from pydantic import (
#     BaseModel,
#     Field,
# )

# from app.dependencies import (
#     get_diagnosis_service,
# )

# from app.services.diagnosis_service import (
#     DiagnosisServiceError,
# )


# class DiagnosisToolInput(BaseModel):
#     """
#     Input schema for the industrial fault diagnosis tool.
#     """

#     file_path: str = Field(
#         description=(
#             "Path to a local drilling-pump "
#             "CSV or XLSX raw data file."
#         )
#     )

#     spm: int = Field(
#         default=90,
#         description=(
#             "Drilling pump operating speed. "
#             "V2.0 currently supports 90 SPM."
#         ),
#     )

#     start_time: Optional[str] = Field(
#         default=None,
#         description=(
#             "Optional diagnosis start time, "
#             "for example 15:35:00."
#         ),
#     )

#     end_time: Optional[str] = Field(
#         default=None,
#         description=(
#             "Optional diagnosis end time, "
#             "for example 16:02:00."
#         ),
#     )


# @tool(
#     "diagnose_industrial_equipment",
#     args_schema=DiagnosisToolInput,
# )
# def diagnose_industrial_equipment(
#     file_path: str,
#     spm: int = 90,
#     start_time: Optional[str] = None,
#     end_time: Optional[str] = None,
# ) -> dict:
#     """
#     Diagnose drilling-pump faults from raw industrial time-series data.

#     Use this tool when fault diagnosis must be performed from a CSV/XLSX
#     sensor-data file. It runs deterministic preprocessing and a trained
#     PyTorch DSMT-1DCNN model, then returns cylinder-level fault results,
#     model scores, window agreement, and top candidate classes.

#     Do not use this tool for general maintenance-document questions.
#     """

#     service = (
#         get_diagnosis_service()
#     )

#     try:

#         return service.diagnose_file(
#             file_path=file_path,
#             spm=spm,
#             start_time=start_time,
#             end_time=end_time,
#         )

#     except DiagnosisServiceError as exc:

#         return {
#             "status": "error",
#             "error_type": (
#                 "diagnosis_service_error"
#             ),
#             "message": str(exc),
#         }

######V2.O版本
from __future__ import annotations

from typing import Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from app.dependencies import get_diagnosis_service
from app.services.diagnosis_service import (
    DiagnosisServiceError,
)


# ============================================================
# Tool Input Schema
# ============================================================

class DiagnosisToolInput(BaseModel):
    """
    Input schema for the industrial diagnosis tool.
    """

    file_path: str = Field(
        description=(
            "Local path of the raw drilling-pump "
            "sensor data file. "
            "Supported formats: CSV and XLSX."
        )
    )

    spm: Literal[90] = Field(
    default=90,
    description=(
        "Drilling-pump operating speed. "
        "Current model supports only 90 SPM."
    ),
)

    start_time: Optional[str] = Field(
        default=None,
        description=(
            "Optional diagnosis start time. "
            "Example: 15:35:00."
        ),
    )

    end_time: Optional[str] = Field(
        default=None,
        description=(
            "Optional diagnosis end time. "
            "Example: 16:02:00."
        ),
    )


# ============================================================
# Diagnosis Tool
# ============================================================

@tool(
    "diagnose_industrial_equipment",
    args_schema=DiagnosisToolInput,
)
def diagnose_industrial_equipment(
    file_path: str,
    spm: Literal[90] = 90,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> dict:
    """
    Diagnose drilling-pump faults from raw industrial
    time-series sensor data.

    Use this tool when a drilling-pump CSV or XLSX sensor
    data file needs fault diagnosis.

    The tool runs deterministic preprocessing and a trained
    PyTorch DSMT-1DCNN model.

    It returns cylinder-level fault types, model scores,
    window agreement, window counts, and top candidate
    fault classes.

    Do not use this tool for general maintenance-document
    questions or knowledge retrieval.
    """

    diagnosis_service = (
        get_diagnosis_service()
    )

    try:

        result = (
            diagnosis_service
            .diagnose_file(
                file_path=file_path,
                spm=spm,
                start_time=start_time,
                end_time=end_time,
            )
        )

        return result

    except DiagnosisServiceError as exc:

        return {
            "status": "error",
            "error_type": (
                "diagnosis_service_error"
            ),
            "message": str(exc),
        }