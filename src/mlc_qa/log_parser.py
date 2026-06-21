"""
Treatment Log CSV Parser.

Parses linear accelerator treatment log files in CSV format.
Extracts: timestamps, actual leaf positions, gantry angles, dose rates.

Handles:
- Multiple CSV formats
- Non-uniform sampling intervals
- Gantry angle wraparound (0/360 degrees)
- Missing data interpolation
- Data validation and quality checks
"""
import csv
import io
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class LogParserError(Exception):
    """Custom exception for log parsing errors."""
    pass


class MissingDataError(LogParserError):
    """Raised when required data columns are missing."""
    pass


class LeafCountMismatchError(LogParserError):
    """Raised when leaf count is inconsistent."""
    pass


class UnevenSamplingWarning(UserWarning):
    """Warning for uneven sampling intervals."""
    pass


@dataclass
class LogSample:
    """Single log sample data."""
    timestamp_sec: float
    dose_rate: float
    gantry_angle: float
    leaf_positions_bank_a: np.ndarray
    leaf_positions_bank_b: np.ndarray
    meterset_weight: Optional[float] = None


@dataclass
class TreatmentLog:
    """Complete treatment log data."""
    filename: str
    samples: List[LogSample] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    @property
    def num_leaves(self) -> int:
        if self.samples:
            return len(self.samples[0].leaf_positions_bank_a)
        return 0

    @property
    def duration_sec(self) -> float:
        if not self.samples:
            return 0.0
        return self.samples[-1].timestamp_sec - self.samples[0].timestamp_sec

    def get_timestamps(self) -> np.ndarray:
        """Get timestamps array in seconds."""
        return np.array([s.timestamp_sec for s in self.samples], dtype=np.float64)

    def get_dose_rates(self) -> np.ndarray:
        """Get dose rates array."""
        return np.array([s.dose_rate for s in self.samples], dtype=np.float64)

    def get_gantry_angles(self, unwrap: bool = True) -> np.ndarray:
        """Get gantry angles array.

        Args:
            unwrap: If True, unwrap angles to avoid 0/360 discontinuity.
        """
        angles = np.array([s.gantry_angle for s in self.samples], dtype=np.float64)
        if unwrap:
            angles = np.unwrap(np.deg2rad(angles))
            angles = np.rad2deg(angles)
        return angles

    def get_leaf_positions_bank_a(self) -> np.ndarray:
        """Get bank A leaf positions array (num_samples x num_leaves)."""
        if not self.samples:
            return np.array([])
        return np.vstack([s.leaf_positions_bank_a for s in self.samples])

    def get_leaf_positions_bank_b(self) -> np.ndarray:
        """Get bank B leaf positions array (num_samples x num_leaves)."""
        if not self.samples:
            return np.array([])
        return np.vstack([s.leaf_positions_bank_b for s in self.samples])

    def get_meterset_weights(self) -> np.ndarray:
        """Get meterset weights array."""
        weights = [s.meterset_weight for s in self.samples]
        if any(w is None for w in weights):
            ts = self.get_timestamps()
            if len(ts) > 1:
                weights = (ts - ts[0]) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else np.zeros_like(ts)
            else:
                weights = np.zeros_like(ts)
        return np.array(weights, dtype=np.float64)

    def get_sampling_statistics(self) -> Dict[str, Any]:
        """Calculate sampling interval statistics."""
        if self.num_samples < 2:
            return {
                "num_samples": self.num_samples,
                "mean_interval_sec": 0.0,
                "min_interval_sec": 0.0,
                "max_interval_sec": 0.0,
                "std_interval_sec": 0.0,
                "is_uniform": True,
            }

        timestamps = self.get_timestamps()
        intervals = np.diff(timestamps)
        mean_interval = np.mean(intervals)
        std_interval = np.std(intervals)
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        is_uniform = cv < 0.1  # Coefficient of variation < 10%

        return {
            "num_samples": self.num_samples,
            "mean_interval_sec": float(mean_interval),
            "min_interval_sec": float(np.min(intervals)),
            "max_interval_sec": float(np.max(intervals)),
            "std_interval_sec": float(std_interval),
            "cv_pct": float(cv * 100),
            "is_uniform": bool(is_uniform),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "filename": self.filename,
            "num_samples": self.num_samples,
            "num_leaves": self.num_leaves,
            "duration_sec": self.duration_sec,
            "sampling_stats": self.get_sampling_statistics(),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


class TreatmentLogParser:
    """Parser for treatment log CSV files."""

    TIMESTAMP_COLUMNS = ["timestamp", "time", "time_sec", "datetime", "date_time"]
    DOSE_RATE_COLUMNS = ["dose_rate", "doserate", "dose rate", "dr", "doserate"]
    GANTRY_COLUMNS = ["gantry", "gantry_angle", "gantryangle", "angle"]
    METERSET_COLUMNS = ["meterset", "mu", "monitor_units", "cumulative_mu", "weight"]
    BANK_A_PATTERN = re.compile(r"(bank_a|banka|bank_a_|a_)\d+", re.IGNORECASE)
    BANK_B_PATTERN = re.compile(r"(bank_b|bankb|bank_b_|b_)\d+", re.IGNORECASE)
    LEAF_PATTERN = re.compile(r"leaf_?(\d+)(?:_bank_?([ab]))?", re.IGNORECASE)
    MLCA_PATTERN = re.compile(r"mlc_?a?_?(\d+)", re.IGNORECASE)
    MLCB_PATTERN = re.compile(r"mlc_?b?_?(\d+)", re.IGNORECASE)

    @classmethod
    def parse(cls, file_path: str, num_leaves: Optional[int] = None) -> TreatmentLog:
        """
        Parse a treatment log CSV file.

        Args:
            file_path: Path to the CSV file.
            num_leaves: Expected number of leaves per bank (auto-detected if None).

        Returns:
            TreatmentLog object.

        Raises:
            LogParserError: If parsing fails.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        filename = path.name

        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise LogParserError(f"Failed to read CSV file: {e}")

        return cls._parse_dataframe(df, filename, num_leaves)

    @classmethod
    def parse_string(cls, content: str, filename: str = "log.csv",
                     num_leaves: Optional[int] = None) -> TreatmentLog:
        """Parse log data from CSV string content."""
        try:
            df = pd.read_csv(io.StringIO(content))
        except Exception as e:
            raise LogParserError(f"Failed to parse CSV content: {e}")

        return cls._parse_dataframe(df, filename, num_leaves)

    @classmethod
    def _parse_dataframe(cls, df: pd.DataFrame, filename: str,
                         num_leaves: Optional[int] = None) -> TreatmentLog:
        """Parse DataFrame into TreatmentLog."""
        if df.empty:
            raise LogParserError("CSV file is empty")

        log = TreatmentLog(filename=filename)

        columns = df.columns.tolist()
        col_lower = {col.lower(): col for col in columns}

        timestamp_col = cls._find_column(col_lower, cls.TIMESTAMP_COLUMNS)
        dose_rate_col = cls._find_column(col_lower, cls.DOSE_RATE_COLUMNS)
        gantry_col = cls._find_column(col_lower, cls.GANTRY_COLUMNS)

        if timestamp_col is None:
            raise MissingDataError(
                "Timestamp column not found. Expected one of: "
                f"{', '.join(cls.TIMESTAMP_COLUMNS)}"
            )

        if dose_rate_col is None:
            raise MissingDataError(
                "Dose rate column not found. Expected one of: "
                f"{', '.join(cls.DOSE_RATE_COLUMNS)}"
            )

        if gantry_col is None:
            raise MissingDataError(
                "Gantry angle column not found. Expected one of: "
                f"{', '.join(cls.GANTRY_COLUMNS)}"
            )

        meterset_col = cls._find_column(col_lower, cls.METERSET_COLUMNS)

        bank_a_cols, bank_b_cols = cls._identify_leaf_columns(columns)

        if len(bank_a_cols) == 0 or len(bank_b_cols) == 0:
            raise MissingDataError(
                f"Leaf position columns not found. Found {len(bank_a_cols)} bank A "
                f"and {len(bank_b_cols)} bank B columns"
            )

        if len(bank_a_cols) != len(bank_b_cols):
            raise LeafCountMismatchError(
                f"Bank A and B leaf count mismatch: bank A={len(bank_a_cols)}, "
                f"bank B={len(bank_b_cols)}"
            )

        if num_leaves is not None and len(bank_a_cols) != num_leaves:
            raise LeafCountMismatchError(
                f"Leaf count mismatch: expected {num_leaves}, got {len(bank_a_cols)}"
            )

        original_timestamps = cls._parse_timestamps(df[timestamp_col])
        if len(original_timestamps) >= 2:
            original_intervals = np.diff(original_timestamps)
            if np.any(original_intervals < 0):
                raise LogParserError(
                    "Timestamps are not strictly increasing. "
                    "Check for duplicate or out-of-order timestamps."
                )
            if np.any(original_intervals == 0):
                raise LogParserError(
                    "Timestamps are not strictly increasing. "
                    "Check for duplicate or out-of-order timestamps."
                )

        df = df.sort_values(timestamp_col).reset_index(drop=True)

        timestamps = cls._parse_timestamps(df[timestamp_col])
        df["_normalized_time"] = timestamps

        cls._validate_sampling(timestamps)

        for idx, row in df.iterrows():
            sample = LogSample(
                timestamp_sec=float(row["_normalized_time"]),
                dose_rate=float(row[dose_rate_col]) if pd.notna(row[dose_rate_col]) else 0.0,
                gantry_angle=cls._normalize_gantry_angle(
                    float(row[gantry_col]) if pd.notna(row[gantry_col]) else 0.0
                ),
                leaf_positions_bank_a=cls._extract_leaf_row(row, bank_a_cols),
                leaf_positions_bank_b=cls._extract_leaf_row(row, bank_b_cols),
                meterset_weight=float(row[meterset_col]) if meterset_col and pd.notna(
                    row[meterset_col]) else None,
            )
            log.samples.append(sample)

        if log.samples:
            log.start_time = datetime.now() - timedelta(seconds=log.duration_sec)
            log.end_time = datetime.now()

        cls._validate_log_data(log)

        return log

    @staticmethod
    def _find_column(col_lower: Dict[str, str], candidates: List[str]) -> Optional[str]:
        """Find a column by candidate names."""
        for candidate in candidates:
            if candidate in col_lower:
                return col_lower[candidate]

        for col_name, original_name in col_lower.items():
            for candidate in candidates:
                if candidate in col_name:
                    return original_name

        return None

    @classmethod
    def _identify_leaf_columns(cls, columns: List[str]) -> Tuple[List[str], List[str]]:
        """Identify leaf position columns for both banks."""
        bank_a_cols = []
        bank_b_cols = []

        seen = set()

        for col in columns:
            col_lower = col.lower()

            if col_lower in seen:
                continue

            bank_a_match = cls.BANK_A_PATTERN.match(col)
            if bank_a_match:
                bank_a_cols.append(col)
                seen.add(col_lower)
                continue

            bank_b_match = cls.BANK_B_PATTERN.match(col)
            if bank_b_match:
                bank_b_cols.append(col)
                seen.add(col_lower)
                continue

            mlca_match = cls.MLCA_PATTERN.match(col)
            if mlca_match:
                bank_a_cols.append(col)
                seen.add(col_lower)
                continue

            mlcb_match = cls.MLCB_PATTERN.match(col)
            if mlcb_match:
                bank_b_cols.append(col)
                seen.add(col_lower)
                continue

            leaf_match = cls.LEAF_PATTERN.match(col)
            if leaf_match:
                bank = leaf_match.group(2)
                if bank and bank.lower() == "a":
                    bank_a_cols.append(col)
                elif bank and bank.lower() == "b":
                    bank_b_cols.append(col)
                seen.add(col_lower)
                continue

        def sort_key(col_name: str) -> int:
            digits = re.findall(r"\d+", col_name)
            return int(digits[-1]) if digits else 0

        bank_a_cols.sort(key=sort_key)
        bank_b_cols.sort(key=sort_key)

        return bank_a_cols, bank_b_cols

    @staticmethod
    def _parse_timestamps(timestamp_series: pd.Series) -> np.ndarray:
        """Parse timestamps into seconds since start."""
        series = timestamp_series.copy()

        if pd.api.types.is_numeric_dtype(series):
            values = series.values.astype(np.float64)
            return values - values[0] if len(values) > 0 else values

        try:
            datetimes = pd.to_datetime(series)
            if pd.api.types.is_datetime64_any_dtype(datetimes):
                start = datetimes.iloc[0]
                deltas = datetimes - start
                return deltas.dt.total_seconds().values.astype(np.float64)
        except (ValueError, TypeError):
            pass

        try:
            return series.astype(np.float64)
        except (ValueError, TypeError) as e:
            raise LogParserError(f"Unable to parse timestamps: {e}")

    @staticmethod
    def _normalize_gantry_angle(angle: float) -> float:
        """Normalize gantry angle to [0, 360) range."""
        angle = angle % 360.0
        if angle < 0:
            angle += 360.0
        return angle

    @staticmethod
    def _extract_leaf_row(row: pd.Series, columns: List[str]) -> np.ndarray:
        """Extract leaf positions from a row."""
        values = []
        for col in columns:
            val = row[col]
            values.append(float(val) if pd.notna(val) else 0.0)
        return np.array(values, dtype=np.float64)

    @classmethod
    def _validate_sampling(cls, timestamps: np.ndarray) -> None:
        """Validate sampling intervals and warn if uneven."""
        if len(timestamps) < 2:
            return

        intervals = np.diff(timestamps)

        if np.any(intervals <= 0):
            raise LogParserError(
                "Timestamps are not strictly increasing. "
                "Check for duplicate or out-of-order timestamps."
            )

        mean_interval = np.mean(intervals)
        if mean_interval > 0:
            cv = np.std(intervals) / mean_interval
            if cv > 0.3:
                import warnings
                warnings.warn(
                    f"Uneven sampling detected (CV={cv*100:.1f}%). "
                    f"Intervals range from {np.min(intervals):.3f}s to {np.max(intervals):.3f}s.",
                    UnevenSamplingWarning
                )

    @classmethod
    def _validate_log_data(cls, log: TreatmentLog) -> None:
        """Validate parsed log data."""
        if log.num_samples < 2:
            raise LogParserError(
                f"Insufficient samples: got {log.num_samples}, need at least 2"
            )

        if log.num_leaves == 0:
            raise LeafCountMismatchError("No leaf positions found in log")

        gantry_angles = log.get_gantry_angles(unwrap=False)
        if np.any(gantry_angles < 0) or np.any(gantry_angles > 360):
            raise LogParserError("Gantry angles out of valid range [0, 360]")


def create_sample_log_csv(
    num_samples: int = 100,
    num_leaves: int = 60,
    duration_sec: float = 30.0,
    max_leaf_position: float = 100.0,
    noise_std: float = 0.5,
    uneven_sampling: bool = False,
) -> str:
    """
    Create sample treatment log CSV content for testing.

    Args:
        num_samples: Number of log samples.
        num_leaves: Number of leaves per bank.
        duration_sec: Total duration in seconds.
        max_leaf_position: Maximum leaf opening in mm.
        noise_std: Standard deviation of noise to add to leaf positions.
        uneven_sampling: If True, create uneven sampling intervals.

    Returns:
        CSV content as string.
    """
    if uneven_sampling:
        base_intervals = np.random.uniform(0.05, 0.5, num_samples - 1)
        timestamps = np.concatenate([[0], np.cumsum(base_intervals)])
        timestamps = timestamps * duration_sec / timestamps[-1]
    else:
        timestamps = np.linspace(0, duration_sec, num_samples)

    meterset_weights = timestamps / duration_sec

    gantry_angles = np.linspace(0, 359, num_samples)

    dose_rates = np.full(num_samples, 600.0)
    dose_rates[:10] = np.linspace(0, 600, 10)
    dose_rates[-10:] = np.linspace(600, 0, 10)

    output = io.StringIO()
    writer = csv.writer(output)

    header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
    for i in range(num_leaves):
        header.append(f"bank_a_{i}")
    for i in range(num_leaves):
        header.append(f"bank_b_{i}")
    writer.writerow(header)

    for idx, t in enumerate(timestamps):
        weight = meterset_weights[idx]
        leaf_pos = max_leaf_position * weight
        noise = np.random.normal(0, noise_std, num_leaves)

        row = [
            f"{t:.4f}",
            f"{dose_rates[idx]:.1f}",
            f"{gantry_angles[idx]:.1f}",
            f"{weight:.6f}",
        ]

        for i in range(num_leaves):
            row.append(f"{-leaf_pos + noise[i]:.3f}")
        for i in range(num_leaves):
            row.append(f"{leaf_pos + noise[i]:.3f}")

        writer.writerow(row)

    return output.getvalue()


def create_log_with_gantry_wraparound(
    num_samples: int = 100,
    num_leaves: int = 60,
) -> str:
    """
    Create a log with gantry angle wrapping around 360 degrees.

    Args:
        num_samples: Number of samples.
        num_leaves: Number of leaves per bank.

    Returns:
        CSV content as string.
    """
    timestamps = np.linspace(0, 30, num_samples)

    gantry_angles = np.linspace(350, 370, num_samples)
    gantry_angles = gantry_angles % 360

    output = io.StringIO()
    writer = csv.writer(output)

    header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
    for i in range(num_leaves):
        header.append(f"bank_a_{i}")
    for i in range(num_leaves):
        header.append(f"bank_b_{i}")
    writer.writerow(header)

    for idx, t in enumerate(timestamps):
        weight = idx / (num_samples - 1)
        row = [
            f"{t:.4f}",
            "600.0",
            f"{gantry_angles[idx]:.1f}",
            f"{weight:.6f}",
        ]
        for i in range(num_leaves):
            row.append(f"{-50.0 * weight:.3f}")
        for i in range(num_leaves):
            row.append(f"{50.0 * weight:.3f}")
        writer.writerow(row)

    return output.getvalue()
