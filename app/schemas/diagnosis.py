"""
功能：
    1.规定别人调用诊断服务时，输入输出的格式

补充知识点：
①：Pydantic 可以做数据格式约束 + 类型校验 + 自动 API 文档
"""

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