"""
功能：负责三件事。收到 HTTP 请求 -> 调用 model_service -> 把结果返回
"""

from fastapi import FastAPI
from app.schemas.diagnosis import DiagnosisRequest, DiagnosisResponse
from app.services.model_service import predict_fault

app=FastAPI(title="Industrial Maintenance Agent", 
            description="工业设备智能故障诊断与运维Agent", 
            version="0.1.0")


@app.post("/diagnose",
           response_model=DiagnosisResponse, 
           tags=["Diagnosis"],)
def diagnose(req: DiagnosisRequest):
    """
    工业设备故障诊断接口
    """
    # 调用模型服务进行故障诊断
    result = predict_fault(req.sample_id)
    # 返回诊断结果
    return result