"""
功能：负责三件事。收到 HTTP 请求 -> 调用 model_service -> 把结果返回
"""
###### V0版本
# from fastapi import FastAPI
# from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse
# from app.services.model_service import predict_fault

# app=FastAPI(title="Industrial Maintenance Agent", 
#             description="工业设备智能故障诊断与运维Agent", 
#             version="0.1.0")


# @app.post("/diagnose",
#            response_model=DiagnosisResponse, 
#            tags=["Diagnosis"],)
# def diagnose(req: DiagnosisRequest):
#     """
#     工业设备故障诊断接口
#     """
#     # 调用模型服务进行故障诊断
#     result = predict_fault(req.sample_id)
#     # 返回诊断结果
#     return result

###### V1.2版本

# """
# 功能：
# 1. 创建 FastAPI 应用
# 2. 初始化模型服务
# 3. 定义 HTTP 路由

# 代码框架：imports
# ↓
# a = FastAPI(...)
# ↓
# 路径配置
# ↓
# model_service 初始化
# ↓
# @app.post(...)
# ↓
# 路由函数
# """
# import shutil
# from pathlib import Path
# from tempfile import TemporaryDirectory

# from fastapi import (
#     FastAPI,
#     File,
#     Form,
#     HTTPException,
#     UploadFile,
# )

# from app.schemas.diagnosis import DiagnosisFileResponse

# from app.services.preprocessing_service import (
#     PreprocessingError,
#     preprocess_file,
# )

# from app.services.model_service import (
#     DSMTModelService,
#     ModelServiceError,
# )

# app = FastAPI(
#     title="Industrial Maintenance Agent",
#     version="1.2.0",
#     description="工业设备智能故障诊断与运维 Agent",
# )

# # 路径
# PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ARTIFACT_DIR = (
#     PROJECT_ROOT
#     / "artifacts"
#     / "drilling_pump"
#     / "90SPM"
# )

# SCALER_PATH = (
#     ARTIFACT_DIR
#     / "scaler.joblib"
# )

# CHECKPOINT_PATH = (
#     ARTIFACT_DIR
#     / "best_model10_dsmt1dcnn_pump10.pt"
# )

# CLASS_MAP_PATH = (
#     ARTIFACT_DIR
#     / "class_map_10.json"
# )

# #模型：模型只在服务启动时加载一次
# model_service = DSMTModelService(
#     checkpoint_path=CHECKPOINT_PATH,
#     class_map_path=CLASS_MAP_PATH,
# )

# @app.post(
#     "/diagnose/file",
#     response_model=DiagnosisFileResponse,
#     tags=["Diagnosis"],
# )
# def diagnose_file(
#     file: UploadFile = File(...),
#     spm: int = Form(90),
#     start_time: str | None = Form(None),
#     end_time: str | None = Form(None),
# ):
#     """
#     Upload raw drilling-pump data and perform
#     real fault diagnosis.
#     """

#     # -----------------------------------------
#     # V1.2 currently supports only 90 SPM
#     # -----------------------------------------

#     if spm != 90:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Current V1.2 model supports "
#                 "only 90 SPM."
#             ),
#         )

#     original_name = (
#         file.filename
#         or "uploaded_file"
#     )

#     suffix = Path(
#         original_name
#     ).suffix.lower()

#     if suffix not in {
#         ".csv",
#         ".xlsx",
#     }:
#         raise HTTPException(
#             status_code=400,
#             detail=(
#                 "Unsupported file format. "
#                 "Use CSV or XLSX."
#             ),
#         )

#     try:

#         # TemporaryDirectory is automatically removed
#         # after diagnosis finishes.
#         with TemporaryDirectory() as tmp_dir:

#             temp_path = (
#                 Path(tmp_dir)
#                 / f"upload{suffix}"
#             )

#             # Save uploaded file temporarily.
#             with temp_path.open("wb") as buffer:

#                 shutil.copyfileobj(
#                     file.file,
#                     buffer,
#                 )

#             # ---------------------------------
#             # Step 1: preprocessing
#             # ---------------------------------

#             preprocessed = preprocess_file(
#                 file_path=temp_path,
#                 scaler_path=SCALER_PATH,
#                 start_time=start_time,
#                 end_time=end_time,
#             )

#             # The preprocessing service sees
#             # the temporary filename.
#             # Restore the user's real filename.
#             preprocessed["file_name"] = (
#                 original_name
#             )

#             # ---------------------------------
#             # Step 2: model inference
#             # ---------------------------------

#             result = (
#                 model_service
#                 .predict_preprocessed(
#                     preprocessed
#                 )
#             )

#             return result

#     except PreprocessingError as exc:

#         raise HTTPException(
#             status_code=422,
#             detail=str(exc),
#         ) from exc

#     except ModelServiceError as exc:

#         raise HTTPException(
#             status_code=500,
#             detail=(
#                 "Model inference failed: "
#                 f"{exc}"
#             ),
#         ) from exc

###### v2.0版本
from __future__ import annotations

import logging
import shutil

from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)

from app.dependencies import (
    get_diagnosis_service,
)

from app.schemas.diagnosis import (
    DiagnosisFileResponse,
)

from app.services.diagnosis_service import (
    DiagnosisServiceError,
)


# ============================================================
# Basic configuration
# ============================================================

APP_NAME = "Industrial Maintenance Agent"

APP_VERSION = "2.0.0"

SUPPORTED_EQUIPMENT = "drilling_pump"

SUPPORTED_SPMS = [90]

ALLOWED_EXTENSIONS = {
    ".csv",
    ".xlsx",
}


logger = logging.getLogger(__name__)


# ============================================================
# FastAPI lifespan
#
# Load the shared diagnosis service before the API starts
# accepting requests.
#
# get_diagnosis_service() is cached, so the PyTorch model
# will not be reloaded for every request.
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info(
        "Initializing industrial diagnosis service..."
    )

    # Warm up / initialize the shared diagnosis service.
    #
    # Internally this loads:
    #   DSMT model structure
    #   checkpoint
    #   class map
    #   scaler path
    get_diagnosis_service()

    logger.info(
        "Industrial diagnosis service initialized."
    )

    yield

    # Release cached service reference when application stops.
    #
    # This is especially useful later when GPU models
    # or larger model registries are introduced.
    get_diagnosis_service.cache_clear()

    logger.info(
        "Industrial diagnosis service released."
    )


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title=APP_NAME,
    version=APP_VERSION,
    description=(
        "工业设备智能故障诊断与运维 Agent。"
        "V2.0 基于真实工业时序数据、"
        "PyTorch DSMT-1DCNN 故障诊断模型"
        "和可复用 DiagnosisService 构建。"
    ),
    lifespan=lifespan,
)


# ============================================================
# System endpoints
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():
    """
    Basic service information.
    """

    return {
        "service": APP_NAME,
        "version": APP_VERSION,
        "status": "running",
        "supported_equipment": [
            SUPPORTED_EQUIPMENT
        ],
        "supported_spms": SUPPORTED_SPMS,
        "docs": "/docs",
    }


@app.get(
    "/health",
    tags=["System"],
)
def health():
    """
    Lightweight health endpoint.

    Also verifies that the cached diagnosis service
    can be obtained successfully.
    """

    service = get_diagnosis_service()

    return {
        "status": "ok",
        "service": APP_NAME,
        "version": APP_VERSION,
        "diagnosis_service": "ready",
        "model": "DSMT1DCNN10",
        "device": str(
            service.model_service.device
        ),
        "supported_spms": SUPPORTED_SPMS,
    }


# ============================================================
# Real industrial diagnosis API
# ============================================================

@app.post(
    "/diagnose/file",
    response_model=DiagnosisFileResponse,
    tags=["Diagnosis"],
    status_code=status.HTTP_200_OK,
)
def diagnose_file(
    file: Annotated[
        UploadFile,
        File(
            description=(
                "Raw drilling-pump sensor data. "
                "Supported formats: CSV or XLSX."
            )
        ),
    ],

    spm: Annotated[
        int,
        Form(
            description=(
                "Drilling pump operating speed. "
                "V2.0 currently supports 90 SPM."
            )
        ),
    ] = 90,

    start_time: Annotated[
        str | None,
        Form(
            description=(
                "Optional diagnosis start time. "
                "Example: 15:35:00"
            )
        ),
    ] = None,

    end_time: Annotated[
        str | None,
        Form(
            description=(
                "Optional diagnosis end time. "
                "Example: 16:02:00"
            )
        ),
    ] = None,
):
    """
    Diagnose faults from raw industrial time-series data.

    Pipeline:

        CSV / XLSX
        -> DiagnosisService
        -> Preprocessing
        -> DSMT-1DCNN inference
        -> Multi-window aggregation
        -> Structured diagnosis result
    """

    # --------------------------------------------------------
    # 1. Validate uploaded filename
    # --------------------------------------------------------

    original_name = (
        file.filename
        or "uploaded_file"
    )

    suffix = (
        Path(original_name)
        .suffix
        .lower()
    )

    if suffix not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=(
                status.HTTP_400_BAD_REQUEST
            ),
            detail=(
                "Unsupported file format. "
                "Current supported formats: "
                ".csv and .xlsx."
            ),
        )


    try:

        # ----------------------------------------------------
        # 2. Save uploaded data to a temporary file
        #
        # preprocessing_service currently works with a
        # filesystem path, so the HTTP upload is temporarily
        # materialized on disk.
        # ----------------------------------------------------

        with TemporaryDirectory(
            prefix="industrial_diagnosis_"
        ) as tmp_dir:

            temp_path = (
                Path(tmp_dir)
                / f"upload{suffix}"
            )

            with temp_path.open(
                "wb"
            ) as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer,
                )


            # ------------------------------------------------
            # 3. Obtain shared DiagnosisService
            #
            # The service is cached.
            # The DSMT checkpoint is NOT loaded again here.
            # ------------------------------------------------

            diagnosis_service = (
                get_diagnosis_service()
            )


            # ------------------------------------------------
            # 4. Run the complete real diagnosis pipeline
            # ------------------------------------------------

            result = (
                diagnosis_service
                .diagnose_file(
                    file_path=temp_path,
                    spm=spm,
                    start_time=start_time,
                    end_time=end_time,
                )
            )


            # ------------------------------------------------
            # 5. Restore user's original filename
            #
            # DiagnosisService only sees the temporary path.
            # The external response should show the real
            # uploaded filename.
            # ------------------------------------------------

            result["file_name"] = (
                original_name
            )


            # ------------------------------------------------
            # 6. FastAPI + Pydantic validate the response
            # ------------------------------------------------

            return result


    # ========================================================
    # Expected domain / business errors
    # ========================================================

    except DiagnosisServiceError as exc:

        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc


    # ========================================================
    # File-system errors
    # ========================================================

    except OSError as exc:

        logger.exception(
            "File IO error during diagnosis."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Failed to process uploaded file."
            ),
        ) from exc


    # ========================================================
    # Unexpected errors
    #
    # Do not return the full Python traceback to users.
    # Keep it in server logs for debugging.
    # ========================================================

    except Exception as exc:

        logger.exception(
            "Unexpected diagnosis error."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Unexpected internal diagnosis error."
            ),
        ) from exc


    finally:

        # Explicitly close the temporary uploaded-file handle.
        try:
            file.file.close()
        except Exception:
            pass