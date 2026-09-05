# from pathlib import Path

# from app.tools.diagnosis_tool import (
#     diagnose_industrial_equipment,
# )
# # diagnosis_tool.py
# from typing import Literal

# spm: Literal[90] = Field(
#     default=90,
#     description="Current model supports 90 SPM."
# )

# PROJECT_ROOT = (
#     Path(__file__)
#     .resolve()
#     .parent
# )


# RAW_FILE = (
#     PROJECT_ROOT
#     / "demo_data"
#     / "raw_data1.xlsx"
# )


# def main():

#     print(
#         "================================="
#     )
#     print(
#         "Diagnosis Tool Test"
#     )
#     print(
#         "================================="
#     )


#     # --------------------------------------------------------
#     # 1. Inspect Tool metadata
#     # --------------------------------------------------------

#     print(
#         "\n[1] Tool name:"
#     )

#     print(
#         diagnose_industrial_equipment.name
#     )


#     print(
#         "\n[2] Tool description:"
#     )

#     print(
#         diagnose_industrial_equipment.description
#     )


#     print(
#         "\n[3] Tool input schema:"
#     )

#     schema = (
#         diagnose_industrial_equipment
#         .args_schema
#         .model_json_schema()
#     )

#     print(schema)


#     # --------------------------------------------------------
#     # 2. Invoke Tool directly
#     # --------------------------------------------------------

#     print(
#         "\n[4] Running Diagnosis Tool..."
#     )

#     result = (
#         diagnose_industrial_equipment
#         .invoke(
#             {
#                 "file_path": str(
#                     RAW_FILE
#                 ),
#                 "spm": 90,
#                 "start_time": (
#                     "15:35:00"
#                 ),
#                 "end_time": (
#                     "16:02:00"
#                 ),
#             }
#         )
#     )


#     # --------------------------------------------------------
#     # 3. Check Tool result
#     # --------------------------------------------------------

#     if result.get(
#         "status"
#     ) == "error":

#         print(
#             "\nDiagnosis Tool failed:"
#         )

#         print(result)

#         return


#     print(
#         "\n================================="
#     )

#     print(
#         "Diagnosis Result"
#     )

#     print(
#         "================================="
#     )


#     for item in result[
#         "cylinders"
#     ]:

#         print(
#             f"C{item['cylinder_id']} | "
#             f"fault={item['fault_type']} | "
#             f"score="
#             f"{item['model_score']:.4f} | "
#             f"agreement="
#             f"{item['window_agreement']:.4f} | "
#             f"windows="
#             f"{item['window_count']}"
#         )


# if __name__ == "__main__":
#     main()

from pathlib import Path

import pytest
from pydantic import ValidationError

from app.tools.diagnosis_tool import (
    DiagnosisToolInput,
    diagnose_industrial_equipment,
)


from pathlib import Path

# #PROJECT_ROOT和RAW_FILE是为了找到F:\git-demo-file\industrial-maintenance-agent\ demo_data\ raw_data1.xlsx

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

RAW_FILE = (
    PROJECT_ROOT
    / "demo_data"
    / "raw_data1.xlsx"
)



###测试1：输入契约是否正确
def test_tool_schema_rejects_unsupported_spm():

    with pytest.raises(ValidationError):

        DiagnosisToolInput(
            file_path="test.xlsx",
            spm=110,
        )

###测试2：异常输入能不能优雅失败，而不是系统崩溃
#即验证整个 Tool → DiagnosisService → PyTorch 真实链路有没有被改坏

def test_tool_missing_file_returns_error():

    result = diagnose_industrial_equipment.invoke(
        {
            "file_path": "not_exist.xlsx",
            "spm": 90,
        }
    )

    assert result["status"] == "error"


def test_real_diagnosis_tool():

    if not RAW_FILE.exists():
        pytest.skip(
            "Local raw_data1.xlsx is not available."
        )

    result = diagnose_industrial_equipment.invoke(
        {
            "file_path": str(RAW_FILE),
            "spm": 90,
            "start_time": "15:35:00",
            "end_time": "16:02:00",
        }
    )

    faults = {
        item["cylinder_id"]:
        item["fault_type"]
        for item in result["cylinders"]
    }

    assert faults == {
        1: "suction_light",
        2: "suction_moderate",
        3: "suction_severe",
    }
    from app.tools.diagnosis_tool import DiagnosisEvidence
    from uuid import UUID
    evidence = [DiagnosisEvidence.model_validate(e) for e in result["evidence"]]
    assert len(evidence) == 3
    assert len({e.trace_id for e in evidence}) == 1
    UUID(evidence[0].trace_id)
    for item, cylinder in zip(evidence, result["cylinders"]):
        assert item.cylinder_id == cylinder["cylinder_id"]
        assert item.fault_type == cylinder["fault_type"]
        assert item.confidence == cylinder["model_score"]
        assert item.model_name == result["model_name"]
        assert item.model_version.startswith("sha256:")
        assert item.latency_ms > 0 and item.warnings
