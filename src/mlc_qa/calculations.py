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


def circular_angle_distance_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum circular distance between two angles (in degrees).

    Handles 0/360 wraparound correctly. Returns the smaller of the
    clockwise or counter-clockwise arc between the two angles.

    Args:
        angle1: First angle(s) in degrees [0, 360).
        angle2: Second angle(s) in degrees [0, 360).

    Returns:
        Minimum distance in degrees, always in [0, 180].
    """
    diff = np.abs(np.asarray(angle1, dtype=np.float64) - np.asarray(angle2, dtype=np.float64))
    return np.minimum(diff, 360.0 - diff)


def circular_abs_diff_deg(angle1: np.ndarray, angle2: np.ndarray) -> np.ndarray:
    """
    Calculate the signed circular difference (angle2 - angle1) in degrees.

    Handles wraparound. Result is in [-180, 180].
    Positive means clockwise from angle1 to angle2.
    """
    a1 = np.asarray(angle1, dtype=np.float64) % 360.0
    a2 = np.asarray(angle2, dtype=np.float64) % 360.0
    diff = a2 - a1
    diff = np.where(diff > 180.0, diff - 360.0, diff)
    diff = np.where(diff < -180.0, diff + 360.0, diff)
    return diff


def unwrap_angles_deg(angles: np.ndarray) -> np.ndarray:
    """
    Unwrap angles in degrees by detecting 0/360 jumps.

    This is more robust than np.unwrap when:
    - Angles are sparsely sampled
    - There is small jitter around the 0/360 boundary
    - Direction needs to be auto-detected

    Args:
        angles: Angles in degrees (wrapped, [0, 360)).

    Returns:
        Unwrapped angles in degrees (monotonic if angles change smoothly).
    """
    angles = np.asarray(angles, dtype=np.float64)
    if len(angles) < 2:
        return angles.copy()

    wrapped = angles % 360.0
    result = wrapped.copy()

    for i in range(1, len(result)):
        step = wrapped[i] - wrapped[i - 1]
        if step > 180.0:
            step -= 360.0
        elif step < -180.0:
            step += 360.0
        result[i] = result[i - 1] + step

    return result


def align_angles_to_reference(
    angles_to_align: np.ndarray,
    reference_angles: np.ndarray,
) -> np.ndarray:
    """
    Align wrapped angles to be on the same "unwrapped branch" as reference.

    This ensures that both plan and log angles are unwrapped consistently
    before interpolation, so that small wraparound jumps (e.g., 359.8 → 0.2)
    are treated as the small step (~0.4°) they actually are.

    Args:
        angles_to_align: Wrapped angles [0, 360) to be aligned.
        reference_angles: Wrapped reference angles [0, 360) for direction context.

    Returns:
        Unwrapped angles for `angles_to_align` that are consistent with
        the trajectory implied by `reference_angles`.
    """
    ref_unwrapped = unwrap_angles_deg(np.asarray(reference_angles, dtype=np.float64))
    to_align_wrapped = np.asarray(angles_to_align, dtype=np.float64) % 360.0

    if len(ref_unwrapped) < 2 or len(to_align_wrapped) < 2:
        return unwrap_angles_deg(to_align_wrapped)

    direction = 1.0 if ref_unwrapped[-1] >= ref_unwrapped[0] else -1.0

    aligned = to_align_wrapped.copy()
    for i in range(1, len(aligned)):
        step = aligned[i] - aligned[i - 1]
        if direction > 0 and step < -180.0:
            step += 360.0
        elif direction < 0 and step > 180.0:
            step -= 360.0
        aligned[i] = aligned[i - 1] + step

    ref_start = ref_unwrapped[0]
    start_diff = circular_abs_diff_deg(aligned[0], ref_start)
    aligned = aligned + (ref_start - (aligned[0] - start_diff))

    return aligned


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
        plan_gantry_wrapped = beam.get_gantry_angles(unwrap=False)
        plan_leaves_a = beam.get_leaf_positions_bank_a()
        plan_leaves_b = beam.get_leaf_positions_bank_b()

        log_timestamps = log.get_timestamps()
        log_dose_rates = log.get_dose_rates()
        log_gantry_wrapped = log.get_gantry_angles(unwrap=False)
        log_leaves_a = log.get_leaf_positions_bank_a()
        log_leaves_b = log.get_leaf_positions_bank_b()

        if not np.all(np.diff(log_weights) >= 0):
            warnings.append("Log meterset weights are not monotonically increasing")
            log_weights = np.maximum.accumulate(log_weights)

        plan_gantry_unwrapped = align_angles_to_reference(
            angles_to_align=plan_gantry_wrapped,
            reference_angles=log_gantry_wrapped,
        )
        log_gantry_unwrapped = unwrap_angles_deg(log_gantry_wrapped)

        gantry_raw_steps = np.abs(np.diff(log_gantry_wrapped))
        gantry_max_raw_step = float(np.max(gantry_raw_steps)) if len(gantry_raw_steps) >= 1 else 0.0
        gantry_max_step_unwrapped = float(np.max(np.abs(
            np.diff(log_gantry_unwrapped)
        ))) if len(log_gantry_unwrapped) >= 2 else 0.0
        if gantry_max_raw_step > 90.0 and gantry_max_step_unwrapped < 5.0:
            warnings.append(
                f"Gantry angle wraparound detected (max raw step="
                f"{gantry_max_raw_step:.1f}°, max unwrapped step="
                f"{gantry_max_step_unwrapped:.1f}°). Using circular interpolation."
            )

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
        interp_log_gantry_unwrapped = self._interpolate_unwrapped(
            log_timestamps, log_gantry_unwrapped, plan_times
        )
        interp_log_gantry_wrapped = interp_log_gantry_unwrapped % 360.0

        gantry_discrepancies = circular_angle_distance_deg(
            interp_log_gantry_wrapped, plan_gantry_wrapped
        )
        max_gantry_discrepancy = float(np.max(gantry_discrepancies))
        mean_gantry_discrepancy = float(np.mean(gantry_discrepancies))
        if max_gantry_discrepancy > 5.0:
            warnings.append(
                f"Large gantry angle discrepancy detected: max="
                f"{max_gantry_discrepancy:.1f}°, mean="
                f"{mean_gantry_discrepancy:.1f}°"
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
                gantry_angle_deg=float(interp_log_gantry_wrapped[i]),
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

        gantry_start = float(plan_gantry_wrapped[0]) if len(plan_gantry_wrapped) > 0 else 0.0
        gantry_end = float(plan_gantry_wrapped[-1]) if len(plan_gantry_wrapped) > 0 else 0.0

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
    def _interpolate_unwrapped(
        source_times: np.ndarray,
        unwrapped_values: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate pre-unwrapped values (angles in degrees) linearly.

        This method works on already-unwrapped angle data to avoid
        0/360 discontinuities during interpolation. Values are kept
        in unwrapped form; caller should apply % 360 if wrapped output is needed.

        Args:
            source_times: Source timestamps (1D).
            unwrapped_values: Pre-unwrapped angles in degrees (1D).
            target_times: Target timestamps (1D).

        Returns:
            Interpolated unwrapped angles in degrees (1D).
        """
        interpolated = np.interp(
            target_times,
            source_times,
            unwrapped_values,
            left=unwrapped_values[0],
            right=unwrapped_values[-1],
        )
        return interpolated

    @staticmethod
    def _interpolate_gantry(
        source_times: np.ndarray,
        gantry_angles: np.ndarray,
        target_times: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate gantry angles, handling wraparound properly.

        Uses unwrapped angles for interpolation, then wraps back to [0, 360).
        Kept for backward compatibility.
        """
        unwrapped = unwrap_angles_deg(gantry_angles)
        interpolated = np.interp(
            target_times,
            source_times,
            unwrapped,
            left=unwrapped[0],
            right=unwrapped[-1],
        )
        return interpolated % 360.0

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


class TrendLabel(str, Enum):
    """Trend classification labels."""
    STABLE_NORMAL = "STABLE_NORMAL"
    GRADUAL_INCREASE = "GRADUAL_INCREASE"
    SHARP_INCREASE = "SHARP_INCREASE"
    SINGLE_SPIKE = "SINGLE_SPIKE"
    IMPROVING = "IMPROVING"
    ERRATIC = "ERRATIC"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ALL_PASS = "ALL_PASS"
    ALL_FAIL = "ALL_FAIL"


@dataclass
class FractionMetrics:
    """Metrics for a single fraction (used in trend analysis)."""
    fraction_number: int
    plan_version: int
    max_leaf_deviation_mm: float
    mean_leaf_deviation_mm: float
    rmse_mm: float
    pass_rate_pct: float
    overall_pass: bool
    qa_date: Optional[Any] = None


@dataclass
class TrendAnalysisResult:
    """Complete trend analysis result."""
    trend_label: TrendLabel
    trend_confidence: float
    overall_description: str

    total_fractions: int
    plan_versions: List[int]
    latest_fraction: int

    max_deviation_slope: Optional[float] = None
    pass_rate_slope: Optional[float] = None
    latest_max_deviation: Optional[float] = None
    latest_pass_rate: Optional[float] = None

    anomaly_flags: List[str] = field(default_factory=list)
    chart_data: Dict[str, Any] = field(default_factory=dict)

    fraction_metrics: List[FractionMetrics] = field(default_factory=list)


class TrendAnalyzer:
    """Analyzer for detecting trends and anomalies in per-fraction QA data.

    Detects patterns like:
    - Gradual increase in leaf deviation (drift)
    - Sharp increase / sudden degradation
    - Single outlier spike
    - Stable / normal pattern
    - Improving trend
    - Erratic / unstable pattern
    """

    def __init__(
        self,
        deviation_threshold_mm: float = 1.0,
        pass_rate_threshold_pct: float = 95.0,
        spike_threshold_mm: float = 1.5,
        spike_confidence_factor: float = 1.8,
        min_fractions_for_trend: int = 3,
    ):
        self.deviation_threshold_mm = deviation_threshold_mm
        self.pass_rate_threshold_pct = pass_rate_threshold_pct
        self.spike_threshold_mm = spike_threshold_mm
        self.spike_confidence_factor = spike_confidence_factor
        self.min_fractions_for_trend = min_fractions_for_trend

    def analyze(self, fraction_metrics: List[FractionMetrics]) -> TrendAnalysisResult:
        """Analyze trend across fractions.

        Args:
            fraction_metrics: List of per-fraction metrics, sorted by fraction number.

        Returns:
            TrendAnalysisResult with classification and details.
        """
        fraction_metrics = sorted(
            fraction_metrics, key=lambda m: (m.plan_version, m.fraction_number)
        )

        if len(fraction_metrics) < 2:
            return self._insufficient_data_result(fraction_metrics)

        max_devs = np.array([m.max_leaf_deviation_mm for m in fraction_metrics])
        pass_rates = np.array([m.pass_rate_pct for m in fraction_metrics])
        fraction_numbers = np.array([m.fraction_number for m in fraction_metrics])
        overall_pass = np.array([m.overall_pass for m in fraction_metrics])
        plan_versions = sorted(set(m.plan_version for m in fraction_metrics))

        slope_deviation, _ = self._linear_regression_slope(fraction_numbers, max_devs)
        slope_pass_rate, _ = self._linear_regression_slope(fraction_numbers, pass_rates)

        anomaly_flags = []
        trend_label = TrendLabel.STABLE_NORMAL
        confidence = 0.5

        if np.all(overall_pass):
            trend_label = TrendLabel.ALL_PASS
            confidence = 0.9
        elif not np.any(overall_pass):
            trend_label = TrendLabel.ALL_FAIL
            confidence = 0.95
            anomaly_flags.append("All fractions failed QA")

        if trend_label in (TrendLabel.ALL_PASS, TrendLabel.ALL_FAIL) and len(fraction_metrics) >= 3:
            pass

        if len(fraction_metrics) >= self.min_fractions_for_trend:
            spike_idx = self._detect_single_spike(max_devs)
            if spike_idx is not None:
                trend_label = TrendLabel.SINGLE_SPIKE
                confidence = min(0.95, 0.6 + 0.1 * len(fraction_metrics))
                anomaly_flags.append(
                    f"Single spike detected at fraction "
                    f"{fraction_metrics[spike_idx].fraction_number}"
                )

        if trend_label not in (TrendLabel.SINGLE_SPIKE, TrendLabel.ALL_FAIL) and len(fraction_metrics) >= self.min_fractions_for_trend:
            mean_dev = np.mean(max_devs)
            std_dev = np.std(max_devs)

            if slope_deviation > 0.05 and std_dev < mean_dev * 0.5:
                if slope_deviation > 0.15:
                    trend_label = TrendLabel.SHARP_INCREASE
                    confidence = min(0.9, 0.5 + slope_deviation * 2)
                else:
                    trend_label = TrendLabel.GRADUAL_INCREASE
                    confidence = min(0.85, 0.5 + slope_deviation)
                anomaly_flags.append(
                    f"Deviation trend increasing: {slope_deviation:.4f}mm/fraction"
                )

            elif slope_deviation < -0.05 and std_dev < mean_dev * 0.5:
                trend_label = TrendLabel.IMPROVING
                confidence = min(0.8, 0.5 + abs(slope_deviation))
                anomaly_flags.append(
                    f"Deviation trend decreasing: {slope_deviation:.4f}mm/fraction"
                )

            elif std_dev > mean_dev * 0.5 and len(fraction_metrics) >= 5:
                trend_label = TrendLabel.ERRATIC
                confidence = min(0.8, 0.5 + std_dev / mean_dev * 0.3)
                anomaly_flags.append(
                    f"Erratic pattern detected (CV={std_dev/mean_dev*100:.1f}%)"
                )

        if trend_label == TrendLabel.STABLE_NORMAL and len(fraction_metrics) >= self.min_fractions_for_trend:
            mean_dev = np.mean(max_devs)
            if mean_dev < self.deviation_threshold_mm:
                confidence = 0.85
            else:
                confidence = 0.6

        max_dev_baseline = max_devs[0] if len(max_devs) > 0 else 0.0
        delta_from_first = float(max_devs[-1] - max_devs[0]) if len(max_devs) >= 2 else None

        if delta_from_first is not None and abs(delta_from_first) > self.deviation_threshold_mm * 0.5:
            if delta_from_first > 0:
                anomaly_flags.append(
                    f"Deviation increased by {delta_from_first:.2f}mm from first fraction"
                )
            else:
                anomaly_flags.append(
                    f"Deviation decreased by {abs(delta_from_first):.2f}mm from first fraction"
                )

        version_changes = [
            m.plan_version for i, m in enumerate(fraction_metrics)
            if i == 0 or m.plan_version != fraction_metrics[i-1].plan_version
        ]
        if len(version_changes) > 1:
            anomaly_flags.append(
                f"Plan re-planned {len(version_changes)-1} time(s): versions {version_changes}"
            )

        description = self._build_description(trend_label, anomaly_flags, len(fraction_metrics))

        chart_data = {
            "fraction_numbers": fraction_numbers.tolist(),
            "max_deviation_mm": max_devs.tolist(),
            "pass_rate_pct": pass_rates.tolist(),
            "overall_pass": overall_pass.tolist(),
            "plan_versions": [m.plan_version for m in fraction_metrics],
            "deviation_threshold_mm": self.deviation_threshold_mm,
            "pass_rate_threshold_pct": self.pass_rate_threshold_pct,
            "slope_deviation_mm_per_fraction": float(slope_deviation) if slope_deviation is not None else None,
            "slope_pass_rate_pct_per_fraction": float(slope_pass_rate) if slope_pass_rate is not None else None,
        }

        latest = fraction_metrics[-1] if fraction_metrics else None

        return TrendAnalysisResult(
            trend_label=trend_label,
            trend_confidence=float(confidence),
            overall_description=description,
            total_fractions=len(fraction_metrics),
            plan_versions=plan_versions,
            max_deviation_slope=float(slope_deviation) if slope_deviation is not None else None,
            pass_rate_slope=float(slope_pass_rate) if slope_pass_rate is not None else None,
            latest_fraction=latest.fraction_number if latest else 0,
            latest_max_deviation=float(latest.max_leaf_deviation_mm) if latest else None,
            latest_pass_rate=float(latest.pass_rate_pct) if latest else None,
            anomaly_flags=anomaly_flags,
            chart_data=chart_data,
            fraction_metrics=fraction_metrics,
        )

    @staticmethod
    def _linear_regression_slope(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
        """Simple linear regression returning slope and r_value.

        Returns (slope, r_value) or (None, None) if insufficient data.
        """
        if len(x) < 2 or len(y) < 2 or len(x) != len(y):
            return None, None

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        x_mean = np.mean(x)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if abs(denominator) < 1e-10:
            return None, None

        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)

        if ss_tot < 1e-10:
            r_value = 1.0
        else:
            r_value = float(1.0 - ss_res / ss_tot)
            r_value = np.sign(slope) * np.sqrt(abs(r_value))

        return float(slope), float(r_value)

    def _detect_single_spike(self, values: np.ndarray) -> Optional[int]:
        """Detect a single outlier spike in a series.

        Uses the spike_confidence_factor * std_dev rule.
        Returns the index of the spike, or None if no spike detected.
        """
        if len(values) < 4:
            return None

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val < 1e-6:
            return None

        deviations = np.abs(values - mean_val) / std_val
        spike_indices = np.where(deviations > self.spike_confidence_factor)[0]

        if len(spike_indices) == 1:
            spike_idx = int(spike_indices[0])
            spike_val = values[spike_idx]
            others_mask = np.ones(len(values), dtype=bool)
            others_mask[spike_idx] = False
            others_mean = np.mean(values[others_mask])
            if spike_val > others_mean and spike_val > self.spike_threshold_mm:
                return spike_idx

        return None

    @staticmethod
    def _insufficient_data_result(fraction_metrics: List[FractionMetrics]) -> TrendAnalysisResult:
        """Build result for insufficient data."""
        plan_versions = sorted(set(m.plan_version for m in fraction_metrics))
        latest = fraction_metrics[-1] if fraction_metrics else None

        return TrendAnalysisResult(
            trend_label=TrendLabel.INSUFFICIENT_DATA,
            trend_confidence=1.0,
            overall_description=(
                f"Insufficient data for trend analysis "
                f"({len(fraction_metrics)} fraction(s)). "
                f"At least 3 fractions are recommended."
            ),
            total_fractions=len(fraction_metrics),
            plan_versions=plan_versions,
            latest_fraction=latest.fraction_number if latest else 0,
            latest_max_deviation=float(latest.max_leaf_deviation_mm) if latest else None,
            latest_pass_rate=float(latest.pass_rate_pct) if latest else None,
            chart_data={
                "fraction_numbers": [m.fraction_number for m in fraction_metrics],
                "max_deviation_mm": [m.max_leaf_deviation_mm for m in fraction_metrics],
                "pass_rate_pct": [m.pass_rate_pct for m in fraction_metrics],
            },
            fraction_metrics=fraction_metrics,
        )

    @staticmethod
    def _build_description(
        trend_label: TrendLabel,
        anomaly_flags: List[str],
        num_fractions: int,
    ) -> str:
        """Build human-readable trend description."""
        descriptions = {
            TrendLabel.STABLE_NORMAL: (
                f"QA results across {num_fractions} fractions are stable and within normal range."
            ),
            TrendLabel.GRADUAL_INCREASE: (
                f"Leaf deviation shows a gradual increasing trend across {num_fractions} fractions. "
                f"May indicate wear or misalignment."
            ),
            TrendLabel.SHARP_INCREASE: (
                f"Leaf deviation shows a sharp increase across {num_fractions} fractions. "
                f"Investigation recommended."
            ),
            TrendLabel.SINGLE_SPIKE: (
                f"Single outlier fraction detected among {num_fractions} total. "
                f"Most likely a one-time issue."
            ),
            TrendLabel.IMPROVING: (
                f"Leaf deviation is trending downward across {num_fractions} fractions. "
                f"Performance is improving."
            ),
            TrendLabel.ERRATIC: (
                f"QA results are erratic and inconsistent across {num_fractions} fractions. "
                f"May indicate unstable delivery conditions."
            ),
            TrendLabel.ALL_PASS: (
                f"All {num_fractions} fractions passed QA. Plan performance is consistent."
            ),
            TrendLabel.ALL_FAIL: (
                f"All {num_fractions} fractions failed QA. Critical issue requires attention."
            ),
            TrendLabel.INSUFFICIENT_DATA: (
                "Not enough data for reliable trend analysis."
            ),
        }

        desc = descriptions.get(trend_label, "Trend analysis complete.")
        if anomaly_flags:
            desc += " Anomalies: " + "; ".join(anomaly_flags[:3])
            if len(anomaly_flags) > 3:
                desc += f" (+{len(anomaly_flags)-3} more)"
        return desc
