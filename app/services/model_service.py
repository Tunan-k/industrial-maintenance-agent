"""
功能：加载模型，数据预处理，model(...)，softmax，故障类别，置信度
    此处重要的工程思想是：
    main.py不是模型，而是接口的入口
"""

MOCK_RESULTS={
    "sample_001": {"fault_type": "轴承内圈故障", "confidence": 0.95},
    "sample_002": {"fault_type": "轴承外圈故障", "confidence": 0.87},
    "sample_003": {"fault_type": "正常情况", "confidence": 0.92},
}

def predict_fault(sample_id: str)-> dict:
    """
    V0 mock diagnosis service.
    后续这里会替换成真实故障诊断模型：
    数据读取 -> 预处理 -> 模型推理 -> 类别与置信度
    """
    # 模拟模型预测结果
    return MOCK_RESULTS.get(sample_id, {"fault_type": "未知故障", "confidence": 0.50, },)