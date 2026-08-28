from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

import joblib
import numpy as np
import pandas as pd


# ============================================================
# V1.1 Drilling Pump Runtime Preprocessing Configuration
# ============================================================

WINDOW_SIZE = 1024
STRIDE = 512

CYLINDERS = (1, 2, 3)

SIGNALS = (
    "Stress",
    "Vibration",
    "Pressure",
)

CHANNEL_ORDER = (
    "stress",
    "vibration",
    "pressure",
)


class PreprocessingError(RuntimeError):
    """Raised when uploaded industrial data cannot be safely preprocessed."""


# ============================================================
# Column normalization:列名兼容逻辑
# Reused from the training preprocessing logic
# ============================================================

def clean_colname(name: str) -> str:
    return (
        str(name)
        .replace("\ufeff", "")
        .replace("\u3000", "")
        .strip()
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common column-name variants.

    Keep this behavior consistent with the training preprocessing code.
    """

    df = df.copy()

    df.columns = [
        clean_colname(column)
        for column in df.columns
    ]

    aliases = {
        "PC_Time": "Pc_Time",
        "pc_time": "Pc_Time",
        "PcTime": "Pc_Time",
        "Pc Time": "Pc_Time",

        "PLC_Time": "Plc_Time",
        "plc_time": "Plc_Time",
        "PlcTime": "Plc_Time",
        "Plc Time": "Plc_Time",

        "Crankshaft Angle": "Crankshaft_Angle",
        "CrankShaft_Angle": "Crankshaft_Angle",
        "crankshaft_angle": "Crankshaft_Angle",

        "sequence": "Sequence",
        "Seq": "Sequence",
        "SEQ": "Sequence",
    }

    for old_name, new_name in aliases.items():
        if old_name in df.columns and new_name not in df.columns:
            df.rename(
                columns={old_name: new_name},
                inplace=True,
            )

    return df


# ============================================================
# Input schema validation
# ============================================================

def required_columns() -> list[str]:
    """
    Columns required by the current drilling-pump diagnosis model.
    """

    columns = ["Pc_Time"]

    for cylinder in CYLINDERS:
        for signal in SIGNALS:
            columns.append(
                f"Cylinder{cylinder}_{signal}"
            )

    return columns


def validate_columns(df: pd.DataFrame) -> None:
    missing = [
        column
        for column in required_columns()
        if column not in df.columns
    ]

    if missing:
        raise PreprocessingError(
            "Uploaded file is missing required columns: "
            + ", ".join(missing)
        )


# ============================================================
# CSV / XLSX input adapter
# ============================================================

def read_raw_file(
    file_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Read industrial raw data from CSV or XLSX.

    External user format:
        CSV / XLSX

    Internal format:
        pandas.DataFrame
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise PreprocessingError(
            f"Input file does not exist: {file_path}"
        )

    suffix = file_path.suffix.lower()

    if suffix == ".xlsx":
        try:
            df = pd.read_excel(file_path)
        except Exception as exc:
            raise PreprocessingError(
                f"Failed to read XLSX file: {exc}"
            ) from exc

    elif suffix == ".csv":

        last_error = None

        for encoding in (
            "gbk",
            "gb2312",
            "utf-8-sig",
            "utf-8",
            "latin-1",
        ):
            try:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                )
                break

            except Exception as exc:
                last_error = exc

        else:
            raise PreprocessingError(
                f"Failed to read CSV file: {last_error}"
            )

    else:
        raise PreprocessingError(
            "Unsupported file format. "
            "Current supported formats: .csv, .xlsx"
        )

    return df


# ============================================================
# Time handling
# ============================================================

def parse_pc_time(
    series: pd.Series,
) -> pd.Series:

    timestamps = pd.to_datetime(
        series,
        errors="coerce",
    )

    valid_ratio = float(
        timestamps.notna().mean()
    )

    if valid_ratio < 0.8:
        raise PreprocessingError(
            "Pc_Time parsing quality is too low. "
            f"Valid ratio = {valid_ratio:.3f}"
        )

    return timestamps


def _parse_time_boundary(
    value: str,
    reference_date: pd.Timestamp,
) -> pd.Timestamp:
    """
    Support both:

        15:35:00

    and:

        2021-03-28 15:35:00
    """

    text = str(value).strip()

    # Time-only format
    if (
        ":" in text
        and "-" not in text
        and "/" not in text
    ):
        try:
            delta = pd.to_timedelta(text)
        except Exception as exc:
            raise PreprocessingError(
                f"Invalid time value: {value}"
            ) from exc

        return reference_date.normalize() + delta

    timestamp = pd.to_datetime(
        text,
        errors="coerce",
    )

    if pd.isna(timestamp):
        raise PreprocessingError(
            f"Invalid datetime value: {value}"
        )

    return timestamp


def filter_time_range(
    df: pd.DataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Optional runtime time-range filtering.

    This is useful when the uploaded original file contains
    more data than the actual diagnostic interval.

    IMPORTANT:
    This is NOT train/val/test splitting.
    """

    if start_time is None and end_time is None:
        return df

    df = df.copy()

    timestamps = df["_Pc_Time_Parsed"]

    valid_times = timestamps.dropna()

    if len(valid_times) == 0:
        raise PreprocessingError(
            "No valid Pc_Time values found."
        )

    reference_date = valid_times.iloc[0]

    mask = pd.Series(
        True,
        index=df.index,
    )

    if start_time is not None:
        start = _parse_time_boundary(
            start_time,
            reference_date,
        )

        mask &= timestamps >= start

    if end_time is not None:
        end = _parse_time_boundary(
            end_time,
            reference_date,
        )

        mask &= timestamps < end

    selected = df.loc[mask].copy()

    if selected.empty:
        raise PreprocessingError(
            "No data remains after time-range filtering."
        )

    return selected


# ============================================================
# Runtime raw-data preparation
# ============================================================

def prepare_dataframe(
    df: pd.DataFrame,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:

    df = normalize_columns(df)

    validate_columns(df)

    df = df.copy()

    # Parse timestamps
    df["_Pc_Time_Parsed"] = parse_pc_time(
        df["Pc_Time"]
    )

    # Sequence is used only to maintain deterministic ordering
    if "Sequence" in df.columns:

        df["_Sequence_Numeric"] = pd.to_numeric(
            df["Sequence"],
            errors="coerce",
        )

    else:

        df["_Sequence_Numeric"] = np.arange(
            len(df),
            dtype=np.int64,
        )

    # Match training-time ordering
    df = df.sort_values(
        [
            "_Pc_Time_Parsed",
            "_Sequence_Numeric",
        ],
        kind="mergesort",
    ).reset_index(drop=True)

    # Optional diagnostic interval
    df = filter_time_range(
        df,
        start_time=start_time,
        end_time=end_time,
    )

    # Convert model input channels to numeric
    for cylinder in CYLINDERS:

        for signal in SIGNALS:

            column = (
                f"Cylinder{cylinder}_{signal}"
            )

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df.reset_index(drop=True)


# ============================================================
# Sliding-window generation
# ============================================================

def iter_window_starts(
    length: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
):

    last = length - window_size

    if last < 0:
        return []

    return range(
        0,
        last + 1,
        stride,
    )


def build_windows_for_cylinder(
    df: pd.DataFrame,
    cylinder: int,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Dict:

    columns = [
        f"Cylinder{cylinder}_Stress",
        f"Cylinder{cylinder}_Vibration",
        f"Cylinder{cylinder}_Pressure",
    ]

    # (L, 3) -> (3, L)
    signal_array = (
        df[columns]
        .to_numpy(dtype=np.float32)
        .T
    )

    windows = []
    window_starts = []

    dropped_nonfinite = 0

    for start in iter_window_starts(
        signal_array.shape[1],
        window_size,
        stride,
    ):

        end = start + window_size

        window = signal_array[
            :,
            start:end,
        ]

        if window.shape != (
            3,
            window_size,
        ):
            continue

        # Match final training preprocessing:
        # no interpolation;
        # windows containing NaN / Inf are discarded.
        if not np.isfinite(window).all():

            dropped_nonfinite += 1
            continue

        windows.append(window)

        window_starts.append(start)

    if windows:

        X = np.stack(
            windows
        ).astype(np.float32)

    else:

        X = np.empty(
            (
                0,
                3,
                window_size,
            ),
            dtype=np.float32,
        )

    return {
        "X": X,
        "window_starts": window_starts,
        "window_count": int(len(X)),
        "dropped_nonfinite": int(
            dropped_nonfinite
        ),
    }


# ============================================================
# StandardScaler
# Must reproduce the training-time transform exactly
# ============================================================

def load_scaler(
    scaler_path: Union[str, Path],
):

    scaler_path = Path(scaler_path)

    if not scaler_path.exists():
        raise PreprocessingError(
            f"Scaler does not exist: {scaler_path}"
        )

    try:
        scaler = joblib.load(
            scaler_path
        )

    except Exception as exc:
        raise PreprocessingError(
            f"Failed to load scaler: {exc}"
        ) from exc

    # Our model requires exactly:
    # stress, vibration, pressure
    if getattr(
        scaler,
        "n_features_in_",
        None,
    ) != 3:

        raise PreprocessingError(
            "Scaler feature number mismatch. "
            "Expected 3 channels: "
            "stress, vibration, pressure."
        )

    return scaler


def transform_with_scaler(
    X: np.ndarray,
    scaler,
) -> np.ndarray:
    """
    Reproduce data_step1.py transform_scaler exactly.

    Input:
        X.shape == (N, 3, 1024)

    StandardScaler expects:
        (N * 1024, 3)
    """

    if X.size == 0:
        return X.astype(
            np.float32
        )

    n, c, length = X.shape

    flat = (
        np.transpose(
            X,
            (0, 2, 1),
        )
        .reshape(-1, c)
    )

    transformed = scaler.transform(
        flat
    ).astype(np.float32)

    transformed = (
        transformed
        .reshape(
            n,
            length,
            c,
        )
    )

    X_scaled = np.transpose(
        transformed,
        (0, 2, 1),
    ).astype(np.float32)

    return X_scaled


# ============================================================
# Main runtime preprocessing API
# ============================================================

def preprocess_file(
    file_path: Union[str, Path],
    scaler_path: Union[str, Path],
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
) -> Dict:
    """
    Convert uploaded drilling-pump raw data into
    model-ready arrays.

    Pipeline:

        CSV / XLSX
        -> DataFrame
        -> column validation
        -> time ordering
        -> optional interval selection
        -> C1/C2/C3 signals
        -> sliding windows
        -> StandardScaler transform
        -> (N, 3, 1024)

    No label information is used here.
    """

    file_path = Path(file_path)

    raw_df = read_raw_file(
        file_path
    )

    df = prepare_dataframe(
        raw_df,
        start_time=start_time,
        end_time=end_time,
    )

    if len(df) < window_size:

        raise PreprocessingError(
            "Diagnostic data is too short. "
            f"At least {window_size} samples are required, "
            f"but only {len(df)} samples are available."
        )

    scaler = load_scaler(
        scaler_path
    )

    cylinders = {}

    for cylinder in CYLINDERS:

        result = build_windows_for_cylinder(
            df=df,
            cylinder=cylinder,
            window_size=window_size,
            stride=stride,
        )

        X_scaled = transform_with_scaler(
            result["X"],
            scaler,
        )

        cylinders[cylinder] = {
            "X": X_scaled,
            "window_starts": result[
                "window_starts"
            ],
            "window_count": result[
                "window_count"
            ],
            "dropped_nonfinite": result[
                "dropped_nonfinite"
            ],
        }

    total_valid_windows = sum(
        item["window_count"]
        for item in cylinders.values()
    )

    if total_valid_windows == 0:

        raise PreprocessingError(
            "No valid diagnostic windows were generated."
        )

    return {
        "file_name": file_path.name,
        "rows_used": int(len(df)),
        "window_size": int(
            window_size
        ),
        "stride": int(
            stride
        ),
        "channel_order": list(
            CHANNEL_ORDER
        ),
        "cylinders": cylinders,
    }