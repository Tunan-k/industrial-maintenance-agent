from pathlib import Path

from app.services.preprocessing_service import preprocess_file
from app.services.model_service import DSMTModelService


PROJECT_ROOT = Path(__file__).resolve().parent

RAW_FILE = PROJECT_ROOT / "demo_data" / "raw_data1.xlsx"

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "drilling_pump"
    / "90SPM"
)

SCALER_PATH = ARTIFACT_DIR / "scaler.joblib"
CHECKPOINT_PATH = (
    ARTIFACT_DIR
    / "best_model10_dsmt1dcnn_pump10.pt"
)
CLASS_MAP_PATH = ARTIFACT_DIR / "class_map_10.json"


def main():

    print("1. Preprocessing raw data...")

    preprocessed = preprocess_file(
        file_path=RAW_FILE,
        scaler_path=SCALER_PATH,

        # raw_data1 = 第2组，90SPM
        # audit 中对应有效区间
        start_time="15:35:00",
        end_time="16:02:00",
    )

    for cylinder_id, data in preprocessed["cylinders"].items():
        print(
            f"C{cylinder_id}: "
            f"X.shape={data['X'].shape}, "
            f"windows={data['window_count']}"
        )

    print("\n2. Loading DSMT model...")

    model_service = DSMTModelService(
        checkpoint_path=CHECKPOINT_PATH,
        class_map_path=CLASS_MAP_PATH,
    )

    print("\n3. Running real inference...")

    result = model_service.predict_preprocessed(
        preprocessed
    )

    print("\n===== Diagnosis Result =====")

    for item in result["cylinders"]:

        print(
            f"C{item['cylinder_id']} | "
            f"fault={item['fault_type']} | "
            f"score={item['model_score']:.4f} | "
            f"agreement={item['window_agreement']:.4f} | "
            f"windows={item['window_count']}"
        )


if __name__ == "__main__":
    main()