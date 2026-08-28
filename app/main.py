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
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from app.schemas.diagnosis import DiagnosisFileResponse

from app.services.preprocessing_service import (
    PreprocessingError,
    preprocess_file,
)

from app.services.model_service import (
    DSMTModelService,
    ModelServiceError,
)

# 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "drilling_pump"
    / "90SPM"
)

SCALER_PATH = (
    ARTIFACT_DIR
    / "scaler.joblib"
)

CHECKPOINT_PATH = (
    ARTIFACT_DIR
    / "best_model10_dsmt1dcnn_pump10.pt"
)

CLASS_MAP_PATH = (
    ARTIFACT_DIR
    / "class_map_10.json"
)

#模型：模型只在服务启动时加载一次
model_service = DSMTModelService(
    checkpoint_path=CHECKPOINT_PATH,
    class_map_path=CLASS_MAP_PATH,
)

@app.post(
    "/diagnose/file",
    response_model=DiagnosisFileResponse,
    tags=["Diagnosis"],
)
def diagnose_file(
    file: UploadFile = File(...),
    spm: int = Form(90),
    start_time: str | None = Form(None),
    end_time: str | None = Form(None),
):
    """
    Upload raw drilling-pump data and perform
    real fault diagnosis.
    """

    # -----------------------------------------
    # V1.2 currently supports only 90 SPM
    # -----------------------------------------

    if spm != 90:
        raise HTTPException(
            status_code=400,
            detail=(
                "Current V1.2 model supports "
                "only 90 SPM."
            ),
        )

    original_name = (
        file.filename
        or "uploaded_file"
    )

    suffix = Path(
        original_name
    ).suffix.lower()

    if suffix not in {
        ".csv",
        ".xlsx",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported file format. "
                "Use CSV or XLSX."
            ),
        )

    try:

        # TemporaryDirectory is automatically removed
        # after diagnosis finishes.
        with TemporaryDirectory() as tmp_dir:

            temp_path = (
                Path(tmp_dir)
                / f"upload{suffix}"
            )

            # Save uploaded file temporarily.
            with temp_path.open("wb") as buffer:

                shutil.copyfileobj(
                    file.file,
                    buffer,
                )

            # ---------------------------------
            # Step 1: preprocessing
            # ---------------------------------

            preprocessed = preprocess_file(
                file_path=temp_path,
                scaler_path=SCALER_PATH,
                start_time=start_time,
                end_time=end_time,
            )

            # The preprocessing service sees
            # the temporary filename.
            # Restore the user's real filename.
            preprocessed["file_name"] = (
                original_name
            )

            # ---------------------------------
            # Step 2: model inference
            # ---------------------------------

            result = (
                model_service
                .predict_preprocessed(
                    preprocessed
                )
            )

            return result

    except PreprocessingError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except ModelServiceError as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Model inference failed: "
                f"{exc}"
            ),
        ) from exc