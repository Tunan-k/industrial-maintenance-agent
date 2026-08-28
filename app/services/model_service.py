# """
# 功能：加载模型，数据预处理，model(...)，softmax，故障类别，置信度
#     此处重要的工程思想是：
#     main.py不是模型，而是接口的入口
# """

# MOCK_RESULTS={
#     "sample_001": {"fault_type": "轴承内圈故障", "confidence": 0.95},
#     "sample_002": {"fault_type": "轴承外圈故障", "confidence": 0.87},
#     "sample_003": {"fault_type": "正常情况", "confidence": 0.92},
# }

# def predict_fault(sample_id: str)-> dict:
#     """
#     V0 mock diagnosis service.
#     后续这里会替换成真实故障诊断模型：
#     数据读取 -> 预处理 -> 模型推理 -> 类别与置信度
#     """
#     # 模拟模型预测结果
#     return MOCK_RESULTS.get(sample_id, {"fault_type": "未知故障", "confidence": 0.50, },)

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

from app.models.dsmt1dcnn import DSMT1DCNN10


class ModelServiceError(RuntimeError):
    """Raised when model loading or inference fails."""


class DSMTModelService:
    """
    Runtime inference service for the
    drilling-pump DSMT1DCNN10 model.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        class_map_path: Union[str, Path],
        batch_size: int = 256,
        device: str | None = None,
    ):

        self.checkpoint_path = Path(
            checkpoint_path
        )

        self.class_map_path = Path(
            class_map_path
        )

        self.batch_size = int(
            batch_size
        )

        # ---------------------------------
        # Device selection
        # ---------------------------------

        if device is None:

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        else:

            self.device = torch.device(
                device
            )

        # ---------------------------------
        # Load class map
        # ---------------------------------

        self.class_names = (
            self._load_class_map()
        )

        # ---------------------------------
        # Load model
        # ---------------------------------

        self.model = (
            self._load_model()
        )

    # ========================================================
    # Model loading
    # ========================================================

    def _load_class_map(
        self,
    ) -> Dict[int, str]:

        if not self.class_map_path.exists():

            raise ModelServiceError(
                "Class map does not exist: "
                f"{self.class_map_path}"
            )

        try:

            with self.class_map_path.open(
                "r",
                encoding="utf-8",
            ) as f:

                payload = json.load(f)

        except Exception as exc:

            raise ModelServiceError(
                f"Failed to load class map: {exc}"
            ) from exc

        class_names_raw = payload.get(
            "class_names"
        )

        if not class_names_raw:

            raise ModelServiceError(
                "class_map_10.json does not contain "
                "'class_names'."
            )

        class_names = {
            int(key): value
            for key, value
            in class_names_raw.items()
        }

        if len(class_names) != 10:

            raise ModelServiceError(
                "Expected 10 classes, "
                f"got {len(class_names)}."
            )

        return class_names

    def _load_model(
        self,
    ) -> DSMT1DCNN10:

        if not self.checkpoint_path.exists():

            raise ModelServiceError(
                "Checkpoint does not exist: "
                f"{self.checkpoint_path}"
            )

        model = DSMT1DCNN10(
            in_ch=3,
            n_cls=10,
        )

        try:

            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.device,
            )

        except Exception as exc:

            raise ModelServiceError(
                f"Failed to load checkpoint: {exc}"
            ) from exc

        if "model_state" not in checkpoint:

            raise ModelServiceError(
                "Checkpoint does not contain "
                "'model_state'."
            )

        try:

            model.load_state_dict(
                checkpoint["model_state"]
            )

        except Exception as exc:

            raise ModelServiceError(
                "Checkpoint parameters do not match "
                f"DSMT1DCNN10: {exc}"
            ) from exc

        model.to(
            self.device
        )

        # VERY IMPORTANT:
        # Disable training behavior such as BatchNorm updates.
        model.eval()

        return model

    # ========================================================
    # Window-level inference
    # ========================================================

    def predict_windows(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Predict probabilities for all windows.

        Input:
            X: (N, 3, 1024)

        Output:
            probabilities: (N, 10)
        """

        if not isinstance(
            X,
            np.ndarray,
        ):

            raise ModelServiceError(
                "X must be a numpy.ndarray."
            )

        if X.ndim != 3:

            raise ModelServiceError(
                "X must have shape "
                "(N, 3, 1024). "
                f"Got {X.shape}."
            )

        if X.shape[1] != 3:

            raise ModelServiceError(
                "Model requires exactly "
                "3 input channels."
            )

        if X.shape[2] != 1024:

            raise ModelServiceError(
                "Model requires window length 1024. "
                f"Got {X.shape[2]}."
            )

        if len(X) == 0:

            raise ModelServiceError(
                "No windows available for inference."
            )

        all_probabilities = []

        # torch.inference_mode is ideal for deployment:
        # no gradient graph is constructed.
        with torch.inference_mode():

            for start in range(
                0,
                len(X),
                self.batch_size,
            ):

                end = min(
                    start + self.batch_size,
                    len(X),
                )

                batch_np = X[
                    start:end
                ]

                batch = torch.from_numpy(
                    batch_np
                ).float().to(
                    self.device
                )

                _, logits = self.model(
                    batch
                )

                probabilities = F.softmax(
                    logits,
                    dim=1,
                )

                all_probabilities.append(
                    probabilities
                    .cpu()
                    .numpy()
                )

        return np.concatenate(
            all_probabilities,
            axis=0,
        )

    # ========================================================
    # Multi-window aggregation
    # ========================================================

    def aggregate_windows(
        self,
        probabilities: np.ndarray,
        top_k: int = 3,
    ) -> Dict:
        """
        Aggregate many sliding-window predictions
        into one cylinder-level diagnosis.

        Final class:
            argmax(mean class probability)

        model_score:
            mean probability of the final class

        window_agreement:
            percentage of windows whose own argmax
            agrees with the final class
        """

        if probabilities.ndim != 2:

            raise ModelServiceError(
                "Probabilities must have "
                "shape (N, num_classes)."
            )

        if probabilities.shape[1] != 10:

            raise ModelServiceError(
                "Expected 10 class probabilities."
            )

        if len(probabilities) == 0:

            raise ModelServiceError(
                "Cannot aggregate empty predictions."
            )

        # Average probabilities over all windows
        mean_probabilities = (
            probabilities.mean(
                axis=0
            )
        )

        final_class_id = int(
            np.argmax(
                mean_probabilities
            )
        )

        model_score = float(
            mean_probabilities[
                final_class_id
            ]
        )

        # Each window's independent prediction
        window_predictions = np.argmax(
            probabilities,
            axis=1,
        )

        window_agreement = float(
            np.mean(
                window_predictions
                == final_class_id
            )
        )

        # Top-K classes for interpretability
        top_indices = (
            np.argsort(
                mean_probabilities
            )[::-1][:top_k]
        )

        top_predictions = []

        for class_id in top_indices:

            top_predictions.append(
                {
                    "class_id": int(
                        class_id
                    ),
                    "fault_type": (
                        self.class_names[
                            int(class_id)
                        ]
                    ),
                    "probability": float(
                        mean_probabilities[
                            class_id
                        ]
                    ),
                }
            )

        return {
            "class_id": final_class_id,
            "fault_type": self.class_names[
                final_class_id
            ],
            "model_score": model_score,
            "window_agreement": (
                window_agreement
            ),
            "window_count": int(
                len(probabilities)
            ),
            "top_predictions": (
                top_predictions
            ),
        }

    # ========================================================
    # Cylinder-level prediction
    # ========================================================

    def predict_cylinder(
        self,
        X: np.ndarray,
    ) -> Dict:

        probabilities = (
            self.predict_windows(X)
        )

        return self.aggregate_windows(
            probabilities
        )

    # ========================================================
    # Full equipment prediction
    # ========================================================

    def predict_preprocessed(
        self,
        preprocessed: Dict,
    ) -> Dict:
        """
        Predict C1/C2/C3 from output produced by
        preprocessing_service.preprocess_file().
        """

        cylinder_results = []

        for cylinder_id, data in (
            preprocessed["cylinders"].items()
        ):

            X = data["X"]

            diagnosis = (
                self.predict_cylinder(X)
            )

            diagnosis.update(
                {
                    "cylinder_id": int(
                        cylinder_id
                    ),
                    "dropped_nonfinite": int(
                        data[
                            "dropped_nonfinite"
                        ]
                    ),
                }
            )

            cylinder_results.append(
                diagnosis
            )

        return {
            "equipment_type": (
                "drilling_pump"
            ),
            "spm": 90,
            "model_name": (
                "DSMT1DCNN10"
            ),
            "device": str(
                self.device
            ),
            "file_name": (
                preprocessed[
                    "file_name"
                ]
            ),
            "rows_used": (
                preprocessed[
                    "rows_used"
                ]
            ),
            "window_size": (
                preprocessed[
                    "window_size"
                ]
            ),
            "stride": (
                preprocessed[
                    "stride"
                ]
            ),
            "cylinders": (
                cylinder_results
            ),
        }