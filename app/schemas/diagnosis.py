"""
功能：
    1.规定别人调用诊断服务时，输入输出的格式

补充知识点：
①：Pydantic 可以做数据格式约束 + 类型校验 + 自动 API 文档
"""

###### V0版本
from typing import List
from pydantic import BaseModel, Field

class DiagnosisRequest(BaseModel):
    """
    【接口输入】诊断请求参数，前端或者 Agent 给 FastAPI 什么数据
    """
    sample_id: str = Field(min_length=1, description="待诊断的设备样本ID")

class DiagnosisResponse(BaseModel):
    """
    【接口输出】诊断结果返回参数，FastAPI 承诺给调用方返回什么数据
    """
    fault_type: str = Field(description="模型预测的故障类别")
    confidence: float = Field(ge=0.0, le=1.0, description="模型预测的置信度")

###### V1.2版本
from typing import List
from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_id: int = Field(
        ge=0,
        le=9,
        description="故障类别 ID"
    )

    fault_type: str = Field(
        description="故障类别名称"
    )

    probability: float = Field(
        ge=0.0,
        le=1.0,
        description="该类别的平均模型概率"
    )


class CylinderDiagnosis(BaseModel):
    cylinder_id: int = Field(
        ge=1,
        le=3,
        description="液缸编号"
    )

    class_id: int = Field(
        ge=0,
        le=9
    )

    fault_type: str

    model_score: float = Field(
        ge=0.0,
        le=1.0,
        description="多窗口平均概率中的最高类别得分"
    )

    window_agreement: float = Field(
        ge=0.0,
        le=1.0,
        description="窗口预测与最终类别的一致率"
    )

    window_count: int = Field(
        ge=1
    )

    dropped_nonfinite: int = Field(
        ge=0,
        description="因 NaN/Inf 被删除的窗口数"
    )

    top_predictions: List[TopPrediction]


class DiagnosisFileResponse(BaseModel):
    equipment_type: str

    spm: int

    model_name: str

    device: str

    file_name: str

    rows_used: int = Field(
        ge=1
    )

    window_size: int = Field(
        ge=1
    )

    stride: int = Field(
        ge=1
    )

    cylinders: List[CylinderDiagnosis]