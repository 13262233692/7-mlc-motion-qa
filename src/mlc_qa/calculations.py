"""
NumPy-based calculation core for MLC motion QA.

Performs core computations:
- Control point interpolation and alignment
- Leaf position deviation calculations
- Dose rate deviation analysis
- Control point pass rate evaluation
- Gantry angle handling (including wraparound)

All computations use vectorized NumPy operations for performance.
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from mlc_qa.dicom_parser import BeamData
from mlc_qa.log_parser import TreatmentLog
from mlc_qa.config import MAX_LEAF_DEVIATION_THRESHOLD, CONTROL_POINT_PASS_THRESHOLD


class InterpolationMethod(str, Enum):
    """Interpolation method for alignment."""
    LINEAR = "linear"
    NEAREST = "nearest"
    CUBIC = "cubic"


class QAPassStatus(str, Enum):
    """QA pass/fail status."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"


@dataclass
class LeafDeviation:
    """Leaf deviation data for a single control point and leaf."""
    control_point_index: int
    leaf_index: int
    bank: str
    planned_position_mm: float
    actual_position_mm: float
    deviation_mm: float
    timestamp_sec: float


@dataclass
class ControlPointResult:
    """QA result for a single control point."""
    control_point_index: int
    max_deviation_mm: float
    mean_deviation_mm: float
    rmse_mm: float
    pass_status: bool
    num_failed_leaves: int
    total_leaves: int
    dose_rate_deviation_pct: float
    gantry_angle_deg: float


@dataclass
class QAAnalysisResult:
    """Complete QA analysis result."""
    max_leaf_deviation_mm: float
    mean_leaf_deviation_mm: float
    rmse_mm: float
    dose_rate_deviation_pct: float
    control_point_pass_rate_pct: float
    num_control_points: int
    num_failed_control_points: int
    num_leaves: int
    gantry_angle_start: float
    gantry_angle_end: float
    overall_pass: bool
    max_deviation_location: Optional[LeafDeviation] = None
    control_point_results: List[ControlPointResult] = field(default_factory=list)
    leaf_deviations: List[LeafDeviation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "max_leaf_deviation_mm": float(self.max_leaf_deviation_mm),
            "mean_leaf_deviation_mm": float(self.mean_leaf_deviation_mm),
            "rmse_mm": float(self.rmse_mm),
            "dose_rate_deviation_pct": float(self.dose_rate_deviation_pct),
            "control_point_pass_rate_pct": float(self.control_point_pass_rate_pct),
            "num_control_points": int(self.num_control_points),
            "num_failed_control_points": int(self.num_failed_control_points),
            "num_leaves": int(self.num_leaves),
            "gantry_angle_start": float(self.gantry_angle_start),
            "gantry_angle_end": float(self.gantry_angle_end),
            "overall_pass": bool(self.overall_pass),
            "warnings": self.warnings,
        }


class CalculationError(Exception):
    """Custom exception for calculation errors."""
    pass


class MLCQACalculator:
    """Core calculator for MLC motion QA analysis."""

    def __init__(
        self,
        leaf_deviation_threshold_mm: float = MAX_LEAF_DEVIATION_THRESHOLD,
        control_point_pass_threshold_pct: float = CONTROL_POINT_PASS_THRESHOLD,
        interpolation_method: InterpolationMethod = InterpolationMethod.LINEAR,
    ):
        """
        Initialize the QA calculator.

        Args:
            leaf_deviation_threshold_mm: Maximum allowed leaf deviation in mm.
            control_point_pass_threshold_pct: Minimum pass rate percentage.
            interpolation_method: Method for interpolating log data to control points.
        """
        self.leaf_deviation_threshold_mm = leaf_deviation_threshold_mm
        self.control_point_pass_threshold_pct = control_point_pass_threshold_pct
        self.interpolation_method = interpolation_method

    def analyze(
        self,
        beam: BeamData,
        log: TreatmentLog,
        sample_errors: bool = True,
        max_error_samples: int = 100,
    ) -> QAAnalysisResult:
        """
        Perform complete QA analysis comparing plan to log.

        Args:
            beam: Beam data from DICOM plan.
            log: Treatment log data.
            sample_errors: Whether to sample individual leaf errors.
            max_error_samples: Maximum number of error samples to retain.

        Returns:
            QAAnalysisResult with all computed metrics.

        Raises:
            CalculationError: If analysis fails.
        """
        self._validate_inputs(beam, log)

        warnings = []

        plan_weights = beam.get_cumulative_weights()
        log_weights = log.get_meterset_weights()

        plan_dose_rates = beam.get_dose_rates()
        plan_gantry = beam.get_gantry_angles()
        plan_leaves_a = beam.get_leaf_positions_bank_a()
        plan_leaves_b = beam.get_leaf_positions_bank_b()

        log_timestamps = log.get_timestamps()
        log_dose_rates = log.get_dose_rates()
        log_gantry = log.get_gantry_angles(unwrap=True)
        log_leaves_a = log.get_leaf_positions_bank_a()
        log_leaves_b = log.get_leaf_positions_bank_b()

        if not np.all(np.diff(log_weights) >= 0):
            warnings.append("Log meterset weights are not monotonically increasing")
            log_weights = np.maximum.accumulate(log_weights)

        plan_to_log_time = self._create_time_weight_mapping(
            log_timestamps, log_weights
        )

        plan_times = plan_to_log_time(plan_weights)

        interp_log_leaves_a = self._interpolate_to_control_points(
            log_timestamps, log_leaves_a, plan_times
        )
        interp_log_leaves_b = self._interpolate_to_control_points(
            log_timestamps, log_leaves_b, plan_times
        )
        interp_log_dose_rates = self._interpolate_to_control_points(
            log_timestamps, log_dose_rates[:, np.newaxis], plan_times
        ).flatten()
        interp_log_gantry = self._interpolate_gantry(
            log_timestamps, log_gantry, plan_times
        )

        deviation_a = np.abs(interp_log_leaves_a - plan_leaves_a)
        deviation_b = np.abs(interp_log_leaves_b - plan_leaves_b)
        all_deviations = np.concatenate([deviation_a, deviation_b], axis=1)

        max_deviation = float(np.max(all_deviations))
        mean_deviation = float(np.mean(all_deviations))
        rmse = float(np.sqrt(np.mean(all_deviations ** 2)))

        leaf_pass_mask = all_deviations <= self.leaf_deviation_threshold_mm
        cp_leaf_pass_rate = np.mean(leaf_pass_mask, axis=1) * 100
        cp_pass_status = cp_leaf_pass_rate >= self.control_point_pass_threshold_pct
        num_failed_cp = int(np.sum(~cp_pass_status))
        overall_pass_rate = float(np.mean(cp_pass_status) * 100)
        overall_pass = overall_pass_rate >= self.control_point_pass_threshold_pct

        plan_dose_rates_safe = np.where(
            np.abs(plan_dose_rates) < 1e-6, 1.0, plan_dose_rates
        )
        dose_rate_deviation_pct = float(
            np.mean(
                np.abs(interp_log_dose_rates - plan_dose_rates)
                / plan_dose_rates_safe
                * 100
            )
        )

        control_point_results = []
        for i in range(len(plan_weights)):
            cp_result = ControlPointResult(
                control_point_index=i,
                max_deviation_mm=float(np.max(all_deviations[i])),
                mean_deviation_mm=float(np.mean(all_deviations[i])),
                rmse_mm=float(np.sqrt(np.mean(all_deviations[i] ** 2))),
                pass_status=bool(cp_pass_status[i]),
                num_failed_leaves=int(np.sum(~leaf_pass_mask[i])),
                total_leaves=all_deviations.shape[1],
                dose_rate_deviation_pct=float(
                    abs(interp_log_dose_rates[i] - plan_dose_rates[i])
                    / max(plan_dose_rates[i], 1e-6)
                    * 100
                ),
                gantry_angle_deg=float(interp_log_gantry[i] % 360),
            )
            control_point_results.append(cp_result)

        leaf_deviations = []
        if sample_errors:
            leaf_deviations = self._sample_leaf_errors(
                deviation_a,
                deviation_b,
                plan_leaves_a,
                plan_leaves_b,
                interp_log_leaves_a,
                interp_log_leaves_b,
                plan_times,
                max_error_samples,
            )

        max_dev_idx = np.unravel_index(np.argmax(all_deviations), all_deviations.shape)
        max_cp, max_leaf_idx = max_dev_idx
        if max_leaf_idx < beam.num_leaves:
            bank = "A"
            planned = float(plan_leaves_a[max_cp, max_leaf_idx])
            actual = float(interp_log_leaves_a[max_cp, max_leaf_idx])
            leaf_idx = max_leaf_idx
        else:
            bank = "B"
            leaf_idx = max_leaf_idx - beam.num_leaves
            planned = float(plan_leaves_b[max_cp, leaf_idx])
            actual = float(interp_log_leaves_b[max_cp, leaf_idx])

        max_deviation_location = LeafDeviation(
            control_point_index=int(max_cp),
            leaf_index=int(leaf_idx),
            bank=bank,
            planned_position_mm=planned,
            actual_position_mm=actual,
            deviation_mm=float(all_deviations[max_cp, max_leaf_idx]),
            timestamp_sec=float(plan_times[max_cp]),
        )

        gantry_start = float(plan_gantry[0]) if len(plan_gantry) > 0 else 0.0
        gantry_end = float(plan_gantry[-1]) if len(plan_gantry) > 0 else 0.0

        if not log.get_sampling_statistics()["is_uniform"]:
            warnings.append(
                f"Log sampling is uneven (CV="
                f"{log.get_sampling_statistics()['cv_pct']:.1f}%)"
            )

        result = QAAnalysisResult(
            max_leaf_deviation_mm=max_deviation,
            mean_leaf_deviation_mm=mean_deviation,
            rmse_mm=rmse,
            dose_rate_deviation_pct=dose_rate_deviation_pct,
            control_point_pass_rate_pct=overall_pass_rate,
            num_control_points=len(plan_weights),
            num_failed_control_points=num_failed_cp,
            num_leaves=beam.num_leaves,
            gantry_angle_start=gantry_start,
            gantry_angle_end=gantry_end,
            overall_pass=overall_pass,
            max_deviation_location=max_deviation_location,
            control_point_results=control_point_results,
            leaf_deviations=leaf_deviations,
            warnings=warnings,
        )

        return result

    @staticmethod
    def _validate_inputs(beam: BeamData, log: TreatmentLog) -> None:
        """Validate input data for consistency."""
        if beam.num_leaves == 0:
            raise CalculationError("Beam has no leaf positions")

        if beam.num_control_points < 2:
            raise CalculationError(
                f"Beam must have at least 2 control points, got {beam.num_control_points}"
            )

        if log.num_leaves == 0:
            raise CalculationError("Log has no leaf positions")

        if log.num_samples < 2:
            raise CalculationError(
                f"Log must have at least 2 samples, got {log.num_samples}"
            )

        if beam.num_leaves != log.num_leaves:
            raise CalculationError(
                f"Leaf count mismatch: plan has {beam.num_leaves}, "
                f"log has {log.num_leaves}"
            )

        plan_weights = beam.get_cumulative_weights()
        if plan_weights[0] != 0.0:
            raise CalculationError(
                f"First control point weight must be 0.0, got {plan_weights[0]}"
            )
        if plan_weights[-1] < 0.9:
            raise CalculationError(
                f"Last control point weight must be >= 0.9, got {plan_weights[-1]}"
            )

    @staticmethod
    def _create_time_weight_mapping(
        timestamps: np.ndarray, weights: np.ndarray
    ):
        """
        Create a function mapping meterset weight to timestamp.

        Uses linear interpolation from weight to time.
        """
        if len(weights) < 2:
            return lambda w: np.zeros_like(w) if hasattr(w, "__len__") else 0.0

        weights_sorted_idx = np.argsort(weights)
        weights_sorted = weights[weights_sorted_idx]
        timestamps_sorted = timestamps[weights_sorted_idx]

        def map_weight_to_time(target_weights):
            return np.interp(
                target_weights,
                weights_sorted,
                timestamps_sorted,
                left=timestamps_sorted[0],
                right=timestamps_sorted[-1],
            )

        return map_weight_to_time

    def _interpolate_to_control_points(
        self,
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate log data to plan control point times.

        Args:
            source_times: Source timestamps (1D array).
            source_values: Source values (n_samples x n_leaves).
            target_times: Target timestamps (n_control_points).

        Returns:
            Interpolated values (n_control_points x n_leaves).
        """
        if self.interpolation_method == InterpolationMethod.NEAREST:
            return self._nearest_interpolation(source_times, source_values, target_times)
        elif self.interpolation_method == InterpolationMethod.CUBIC:
            return self._cubic_interpolation(source_times, source_values, target_times)
        else:
            return self._linear_interpolation(source_times, source_values, target_times)

    @staticmethod
    def _linear_interpolation(
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Linear interpolation."""
        if source_values.ndim == 1:
            source_values = source_values[:, np.newaxis]

        n_target = len(target_times)
        n_features = source_values.shape[1]
        result = np.zeros((n_target, n_features), dtype=np.float64)

        for j in range(n_features):
            result[:, j] = np.interp(
                target_times,
                source_times,
                source_values[:, j],
                left=source_values[0, j],
                right=source_values[-1, j],
            )

        return result

    @staticmethod
    def _nearest_interpolation(
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Nearest neighbor interpolation."""
        if source_values.ndim == 1:
            source_values = source_values[:, np.newaxis]

        indices = np.searchsorted(source_times, target_times, side="left")
        indices = np.clip(indices, 0, len(source_times) - 1)

        prev_indices = np.maximum(indices - 1, 0)
        dist_to_prev = target_times - source_times[prev_indices]
        dist_to_curr = source_times[indices] - target_times
        nearest = np.where(dist_to_prev < dist_to_curr, prev_indices, indices)

        return source_values[nearest]

    @staticmethod
    def _cubic_interpolation(
        source_times: np.ndarray,
        source_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """Cubic spline interpolation."""
        try:
            from scipy.interpolate import interp1d
            has_scipy = True
        except ImportError:
            has_scipy = False

        if not has_scipy:
            return MLCQACalculator._linear_interpolation(
                source_times, source_values, target_times
            )

        if source_values.ndim == 1:
            source_values = source_values[:, np.newaxis]

        n_target = len(target_times)
        n_features = source_values.shape[1]
        result = np.zeros((n_target, n_features), dtype=np.float64)

        for j in range(n_features):
            f = interp1d(
                source_times,
                source_values[:, j],
                kind="cubic",
                fill_value="extrapolate",
            )
            result[:, j] = f(target_times)

        return result

    @staticmethod
    def _interpolate_gantry(
        source_times: np.ndarray,
        gantry_angles: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate gantry angles, handling wraparound properly.

        Uses unwrapped angles for interpolation, then wraps back to [0, 360).
        """
        unwrapped = np.unwrap(np.deg2rad(gantry_angles))
        interpolated = np.interp(
            target_times,
            source_times,
            unwrapped,
            left=unwrapped[0],
            right=unwrapped[-1],
        )
        return np.rad2deg(interpolated) % 360

    def _sample_leaf_errors(
        self,
        deviation_a: np.ndarray,
        deviation_b: np.ndarray,
        plan_a: np.ndarray,
        plan_b: np.ndarray,
        actual_a: np.ndarray,
        actual_b: np.ndarray,
        timestamps: np.ndarray,
        max_samples: int,
    ) -> List[LeafDeviation]:
        """
        Sample leaf errors for detailed analysis.

        Prioritizes largest deviations, then samples uniformly across control points.
        """
        samples: List[LeafDeviation] = []
        n_cp, n_leaves = deviation_a.shape

        all_errors = []
        for cp in range(n_cp):
            for leaf in range(n_leaves):
                for bank, dev, plan_pos, actual_pos in [
                    ("A", deviation_a[cp, leaf], plan_a[cp, leaf], actual_a[cp, leaf]),
                    ("B", deviation_b[cp, leaf], plan_b[cp, leaf], actual_b[cp, leaf]),
                ]:
                    all_errors.append((
                        dev, cp, leaf, bank, plan_pos, actual_pos
                    ))

        all_errors.sort(key=lambda x: x[0], reverse=True)

        for dev, cp, leaf, bank, plan_pos, actual_pos in all_errors[:max_samples]:
            samples.append(LeafDeviation(
                control_point_index=int(cp),
                leaf_index=int(leaf),
                bank=bank,
                planned_position_mm=float(plan_pos),
                actual_position_mm=float(actual_pos),
                deviation_mm=float(dev),
                timestamp_sec=float(timestamps[cp]),
            ))

        return samples


def calculate_gamma_index(
    planned: np.ndarray,
    actual: np.ndarray,
    dose_to_agreement_pct: float = 3.0,
    distance_to_agreement_mm: float = 3.0,
) -> float:
    """
    Calculate gamma index pass rate.

    Args:
        planned: Planned dose/position array.
        actual: Actual measured array.
        dose_to_agreement_pct: Dose difference tolerance (%).
        distance_to_agreement_mm: Distance tolerance (mm).

    Returns:
        Gamma pass rate (0-100).
    """
    delta_dose = np.abs(actual - planned)
    if np.max(planned) > 0:
        delta_dose_normalized = delta_dose / (np.max(planned) * dose_to_agreement_pct / 100.0)
    else:
        delta_dose_normalized = delta_dose

    delta_pos = np.abs(actual - planned) / distance_to_agreement_mm

    gamma = np.sqrt(delta_dose_normalized ** 2 + delta_pos ** 2)
    pass_rate = np.mean(gamma <= 1.0) * 100.0

    return float(pass_rate)


def compute_statistics(values: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for an array of values.

    Args:
        values: Input array.

    Returns:
        Dictionary with mean, std, min, max, median, p25, p75, p95, p99.
    """
    if len(values) == 0:
        return {
            "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
            "median": 0.0, "p25": 0.0, "p75": 0.0, "p95": 0.0, "p99": 0.0,
        }

    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }
