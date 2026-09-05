# Retriever 质量评估

初始标签由 Codex 对照原文审阅，待人工复核；以下是暂定指标，不是人工验收结果。

Top1 命中率：3/12 = 25.00%。
Top3 相关率（Precision@3）：5/36 = 13.89%。
标签：2=正文直接支持问题；1=背景或部分支持；0=不支持。仅 2 计入命中；空结果计未命中。
索引覆盖：46/63；排除项在 JSON 中逐条记录。

## q01：钻井泵吸入阀故障原因和检查方法

期望 equipment_type=drilling_pump；source_type=troubleshooting。
原始语料存在直接答案：False。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.908986 | 钻井泵.pptx | null | 1 | 说明液力端磨损腐蚀背景，未给出吸入阀检查方法。 |
| 2 | chk_fed75cb96f7b6e9a1f00 | 0.882074 | 钻井泵.pptx | null | 0 | 未提供本题所需的吸入阀原因及检查步骤；同设备或同词不足以判相关。 |
| 3 | chk_c127fd4722c47b8806f6 | 0.869885 | 钻井泵.pptx | null | 0 | 未提供本题所需的吸入阀原因及检查步骤；同设备或同词不足以判相关。 |

## q02：压力异常可能原因

期望 equipment_type=drilling_pump；source_type=troubleshooting。
原始语料存在直接答案：False。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.863402 | 钻井泵.pptx | null | 1 | 提到压力波动监测，未解释异常压力的原因。 |
| 2 | chk_fed75cb96f7b6e9a1f00 | 0.836080 | 钻井泵.pptx | null | 0 | 未提供本题所需的压力异常的因果解释；同设备或同词不足以判相关。 |
| 3 | chk_c127fd4722c47b8806f6 | 0.823185 | 钻井泵.pptx | null | 0 | 未提供本题所需的压力异常的因果解释；同设备或同词不足以判相关。 |

## q03：设备振动异常处理措施

期望 equipment_type=drilling_pump；source_type=maintenance_sop。
原始语料存在直接答案：False。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.861673 | 钻井泵.pptx | null | 1 | 描述振动传感器监测，未提供现场处置。 |
| 2 | chk_fed75cb96f7b6e9a1f00 | 0.842974 | 钻井泵.pptx | null | 0 | 未提供本题所需的现场振动异常处理步骤；同设备或同词不足以判相关。 |
| 3 | chk_c127fd4722c47b8806f6 | 0.829434 | 钻井泵.pptx | null | 0 | 未提供本题所需的现场振动异常处理步骤；同设备或同词不足以判相关。 |

## q04：检修泥浆泵前如何隔离驱动并上锁挂牌？

期望 equipment_type=drilling_pump；source_type=safety_procedure。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.862709 | 钻井泵.pptx | null | 0 | 未提供本题所需的检修前隔离和上锁挂牌；同设备或同词不足以判相关。 |
| 2 | chk_c127fd4722c47b8806f6 | 0.847843 | 钻井泵.pptx | null | 0 | 未提供本题所需的检修前隔离和上锁挂牌；同设备或同词不足以判相关。 |
| 3 | chk_fed75cb96f7b6e9a1f00 | 0.845702 | 钻井泵.pptx | null | 0 | 未提供本题所需的检修前隔离和上锁挂牌；同设备或同词不足以判相关。 |

## q05：泥浆泵剪切式泄压阀更换销钉前要做什么，能用内六角扳手替代销钉吗？

期望 equipment_type=drilling_pump；source_type=safety_procedure。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.856331 | 钻井泵.pptx | null | 0 | 未提供本题所需的泄压阀卸压及销钉替代限制；同设备或同词不足以判相关。 |
| 2 | chk_c127fd4722c47b8806f6 | 0.851255 | 钻井泵.pptx | null | 0 | 未提供本题所需的泄压阀卸压及销钉替代限制；同设备或同词不足以判相关。 |
| 3 | chk_fed75cb96f7b6e9a1f00 | 0.841380 | 钻井泵.pptx | null | 0 | 未提供本题所需的泄压阀卸压及销钉替代限制；同设备或同词不足以判相关。 |

## q06：钻井泵阀盒部署哪些传感器，采样频率是多少？

期望 equipment_type=drilling_pump；source_type=other。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.901062 | 钻井泵.pptx | null | 2 | 明确振动、应变、压力传感器及1kHz采样。 |
| 2 | chk_c127fd4722c47b8806f6 | 0.869151 | 钻井泵.pptx | null | 0 | 未提供本题所需的传感器类型和采样频率；同设备或同词不足以判相关。 |
| 3 | chk_fed75cb96f7b6e9a1f00 | 0.864032 | 钻井泵.pptx | null | 0 | 未提供本题所需的传感器类型和采样频率；同设备或同词不足以判相关。 |

## q07：钻井泵项目如何准备数据集、划分训练验证测试集并评估模型？

期望 equipment_type=drilling_pump；source_type=other。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_fed75cb96f7b6e9a1f00 | 0.920847 | 钻井泵.pptx | null | 2 | 明确标准化预处理、训练验证测试划分、验证调参和测试评估。 |
| 2 | chk_0eac7d37996da0f982f7 | 0.882519 | 钻井泵.pptx | null | 0 | 未提供本题所需的数据准备和模型评估流程；同设备或同词不足以判相关。 |
| 3 | chk_c127fd4722c47b8806f6 | 0.881899 | 钻井泵.pptx | null | 0 | 未提供本题所需的数据准备和模型评估流程；同设备或同词不足以判相关。 |

## q08：五缸泵振动信号用VMD和FastICA抑制缸间串扰的具体步骤是什么？

期望 equipment_type=drilling_pump；source_type=research_paper。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.842245 | 钻井泵.pptx | null | 0 | 未提供本题所需的VMD-FastICA处理步骤；同设备或同词不足以判相关。 |
| 2 | chk_99390fd9604ec62a8607 | 0.837839 | sensors-26-04917.pdf | 4 | 1 | 概述VMD-FastICA抑制串扰，但未列完整执行步骤。 |
| 3 | chk_fed75cb96f7b6e9a1f00 | 0.835411 | 钻井泵.pptx | null | 0 | 未提供本题所需的VMD-FastICA处理步骤；同设备或同词不足以判相关。 |

## q09：五缸泵转速波动时，为什么要对振动信号做角域重采样？

期望 equipment_type=drilling_pump；source_type=research_paper。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_0eac7d37996da0f982f7 | 0.863456 | 钻井泵.pptx | null | 0 | 未提供本题所需的角域重采样的动机；同设备或同词不足以判相关。 |
| 2 | chk_fed75cb96f7b6e9a1f00 | 0.844378 | 钻井泵.pptx | null | 0 | 未提供本题所需的角域重采样的动机；同设备或同词不足以判相关。 |
| 3 | chk_c127fd4722c47b8806f6 | 0.836746 | 钻井泵.pptx | null | 0 | 未提供本题所需的角域重采样的动机；同设备或同词不足以判相关。 |

## q10：论文VICA-DA-ViT在五缸泵跨工况诊断中的平均准确率是多少？

期望 equipment_type=drilling_pump；source_type=research_paper。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_fed75cb96f7b6e9a1f00 | 0.862951 | 钻井泵.pptx | null | 0 | 未提供本题所需的跨工况平均准确率；同设备或同词不足以判相关。 |
| 2 | chk_0eac7d37996da0f982f7 | 0.853046 | 钻井泵.pptx | null | 0 | 未提供本题所需的跨工况平均准确率；同设备或同词不足以判相关。 |
| 3 | chk_f1aa5656e240fb410568 | 0.852009 | sensors-26-04917.pdf | 23 | 2 | 本块直接报告跨工况平均准确率87.42%，不能误用消融实验准确率。 |

## q11：How should a mud pump drive be isolated and locked out before maintenance?

期望 equipment_type=drilling_pump；source_type=safety_procedure。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_35b68889cb9915ab3cd8 | 0.872104 | Working on Mud Pumps - IADC.org.html | null | 2 | 明确关闭控制阀、无安全阀时断开气源、电动设备上锁挂牌。 |
| 2 | chk_3207237ed0d6c610af2a | 0.870327 | Working on Mud Pumps - IADC.org.html | null | 0 | 未提供本题所需的检修前隔离和上锁挂牌；同设备或同词不足以判相关。 |
| 3 | chk_ab47dc9d28a04db91c6d | 0.840400 | Working on Mud Pumps - IADC.org.html | null | 0 | 未提供本题所需的检修前隔离和上锁挂牌；同设备或同词不足以判相关。 |

## q12：What are the VMD and FastICA processing steps to suppress inter-cylinder crosstalk in a quintuplex pump?

期望 equipment_type=drilling_pump；source_type=research_paper。
原始语料存在直接答案：True。

| Rank | chunk_id | score | source | page | 标签 | 理由 |
|---|---|---|---|---|---|---|
| 1 | chk_99390fd9604ec62a8607 | 0.911954 | sensors-26-04917.pdf | 4 | 1 | 概述VMD-FastICA抑制串扰，但未列完整执行步骤。 |
| 2 | chk_6ae2c287109bace3e61e | 0.882979 | sensors-26-04917.pdf | 3 | 0 | 未提供本题所需的VMD-FastICA处理步骤；同设备或同词不足以判相关。 |
| 3 | chk_8602416ac77da5d4cb4d | 0.882361 | sensors-26-04917.pdf | 5 | 2 | Table 1给出预处理、VMD分解、选模、虚拟通道及FastICA六步。 |

## 可观测诊断统计

```json
{
  "corpus_format_counts": {
    "html": 7,
    "pdf": 53,
    "pptx": 3
  },
  "indexed_format_counts": {
    "html": 7,
    "pdf": 36,
    "pptx": 3
  },
  "top3_format_counts": {
    "pptx": 28,
    "pdf": 5,
    "html": 3
  },
  "irrelevant_top3_format_counts": {
    "pptx": 23,
    "html": 2,
    "pdf": 1
  },
  "source_type_counts": {
    "industry_safety_guidance": 7,
    "research_paper": 53,
    "internal_project_note": 3
  },
  "equipment_matches": 36,
  "source_type_matches": 5,
  "returned_top3_slots": 36,
  "missing_page_top3": 31,
  "no_direct_answer_queries": [
    "q01",
    "q02",
    "q03"
  ]
}
```
