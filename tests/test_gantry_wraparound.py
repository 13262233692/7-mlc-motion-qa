"""
Tests for gantry angle wraparound handling.

Covers:
- Small wraparound jumps (359.8 -> 0.2 degrees)
- Clockwise and counter-clockwise rotation
- Small jitter around the 0/360 boundary
- Plan and log angle consistency validation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import json
import csv
import io
import pytest

from mlc_qa.dicom_parser import (
    DicomRTParser,
    create_simplified_plan_json,
    BeamData,
    ControlPoint,
)
from mlc_qa.log_parser import (
    TreatmentLogParser,
    create_sample_log_csv,
)
from mlc_qa.calculations import (
    MLCQACalculator,
    circular_angle_distance_deg,
    circular_abs_diff_deg,
    unwrap_angles_deg,
    align_angles_to_reference,
)


class TestCircularAngleDistance:
    """Tests for circular angle distance utility functions."""

    def test_small_positive_step(self):
        """359.8 -> 0.2 is 0.4 degrees, not 359.6"""
        dist = circular_angle_distance_deg(np.array([359.8]), np.array([0.2]))
        assert dist[0] == pytest.approx(0.4, abs=1e-4)

    def test_small_negative_step(self):
        """0.2 -> 359.8 is 0.4 degrees, not -359.6"""
        dist = circular_angle_distance_deg(np.array([0.2]), np.array([359.8]))
        assert dist[0] == pytest.approx(0.4, abs=1e-4)

    def test_180_degree_step(self):
        """Exactly 180 degrees should return 180"""
        dist = circular_angle_distance_deg(np.array([0.0]), np.array([180.0]))
        assert dist[0] == pytest.approx(180.0, abs=1e-4)

    def test_large_clockwise_step(self):
        """10 -> 350 is 20 degrees (the shorter way)"""
        dist = circular_angle_distance_deg(np.array([10.0]), np.array([350.0]))
        assert dist[0] == pytest.approx(20.0, abs=1e-4)

    def test_normal_non_wrapping_step(self):
        """Normal steps should work as expected"""
        dist = circular_angle_distance_deg(np.array([30.0]), np.array([50.0]))
        assert dist[0] == pytest.approx(20.0, abs=1e-4)

    def test_signed_diff_clockwise_wraparound(self):
        """Clockwise wraparound: 359.8 -> 0.2 should be +0.4"""
        diff = circular_abs_diff_deg(np.array([359.8]), np.array([0.2]))
        assert diff[0] == pytest.approx(0.4, abs=1e-4)

    def test_signed_diff_counter_clockwise_wraparound(self):
        """Counter-clockwise wraparound: 0.2 -> 359.8 should be -0.4"""
        diff = circular_abs_diff_deg(np.array([0.2]), np.array([359.8]))
        assert diff[0] == pytest.approx(-0.4, abs=1e-4)

    def test_signed_diff_normal_positive(self):
        """Normal positive step"""
        diff = circular_abs_diff_deg(np.array([30.0]), np.array([50.0]))
        assert diff[0] == pytest.approx(20.0, abs=1e-4)

    def test_signed_diff_normal_negative(self):
        """Normal negative step"""
        diff = circular_abs_diff_deg(np.array([50.0]), np.array([30.0]))
        assert diff[0] == pytest.approx(-20.0, abs=1e-4)

    def test_distance_vectorized(self):
        """Vectorized distance calculation"""
        a1 = np.array([359.8, 0.0, 10.0, 180.0])
        a2 = np.array([0.2, 359.8, 350.0, 0.0])
        dists = circular_angle_distance_deg(a1, a2)
        expected = [0.4, 0.2, 20.0, 180.0]
        for i, (d, e) in enumerate(zip(dists, expected)):
            assert d == pytest.approx(e, abs=1e-4), f"Index {i}: got {d}, expected {e}"


class TestUnwrapAnglesDeg:
    """Tests for degree-based angle unwrapping."""

    def test_clockwise_through_360(self):
        """Smooth clockwise rotation through 0/360 boundary"""
        wrapped = np.array([358.0, 359.0, 359.8, 0.2, 1.0, 2.0])
        unwrapped = unwrap_angles_deg(wrapped)
        steps = np.diff(unwrapped)
        assert np.all(steps > 0), f"Should be monotonic increasing, steps: {steps}"
        max_step = np.max(np.abs(steps))
        assert max_step < 5.0, f"Max step should be small, got {max_step}"
        assert unwrapped[0] == pytest.approx(358.0, abs=1e-4)
        assert unwrapped[-1] == pytest.approx(362.0, abs=1e-4)

    def test_counter_clockwise_through_360(self):
        """Smooth counter-clockwise rotation through 0/360 boundary"""
        wrapped = np.array([2.0, 1.0, 0.2, 359.8, 359.0, 358.0])
        unwrapped = unwrap_angles_deg(wrapped)
        steps = np.diff(unwrapped)
        assert np.all(steps < 0), f"Should be monotonic decreasing, steps: {steps}"
        max_step = np.max(np.abs(steps))
        assert max_step < 5.0, f"Max step should be small, got {max_step}"
        assert unwrapped[0] == pytest.approx(2.0, abs=1e-4)
        assert unwrapped[-1] == pytest.approx(-2.0, abs=1e-4)

    def test_small_boundary_jitter(self):
        """Small jitter around 0/360 boundary should not cause unwrap errors"""
        wrapped = np.array([0.1, 359.9, 0.05, 359.95, 0.12, 359.88])
        unwrapped = unwrap_angles_deg(wrapped)
        max_abs_step = np.max(np.abs(np.diff(unwrapped)))
        assert max_abs_step < 5.0, f"Jitter should produce small steps, got {max_abs_step}"

    def test_full_clockwise_rotation(self):
        """Full clockwise 360 rotation"""
        angles = np.linspace(0, 360, 100, endpoint=False) % 360
        unwrapped = unwrap_angles_deg(angles)
        assert unwrapped[0] == pytest.approx(0.0)
        assert unwrapped[-1] == pytest.approx(360.0 - 3.6, abs=0.1)

    def test_full_counter_clockwise_rotation(self):
        """Full counter-clockwise 360 rotation"""
        angles = np.linspace(360, 0, 100, endpoint=False) % 360
        unwrapped = unwrap_angles_deg(angles)
        assert unwrapped[0] == pytest.approx(0.0, abs=0.1)
        assert unwrapped[-1] == pytest.approx(-360.0 + 3.6, abs=0.1)

    def test_identical_to_np_unwrap(self):
        """For smooth data, should match np.unwrap behavior"""
        wrapped = np.array([0.0, 90.0, 180.0, 270.0, 359.0, 1.0, 10.0])
        via_np = np.rad2deg(np.unwrap(np.deg2rad(wrapped)))
        via_custom = unwrap_angles_deg(wrapped)
        assert np.allclose(via_np, via_custom, atol=0.1)


class TestAlignAnglesToReference:
    """Tests for aligning plan angles to log reference direction."""

    def test_plan_log_clockwise_wraparound(self):
        """Both plan and log go clockwise through 360"""
        log_angles = np.array([350.0, 355.0, 359.8, 0.2, 5.0, 10.0])
        plan_angles = np.array([350.0, 355.0, 359.8, 0.2, 5.0, 10.0])
        aligned = align_angles_to_reference(plan_angles, log_angles)
        steps = np.diff(aligned)
        assert np.all(steps > 0)
        max_step = np.max(np.abs(steps))
        assert max_step < 10.0

    def test_plan_log_counter_clockwise(self):
        """Both plan and log go counter-clockwise through 360"""
        log_angles = np.array([10.0, 5.0, 0.2, 359.8, 355.0, 350.0])
        plan_angles = np.array([10.0, 5.0, 0.2, 359.8, 355.0, 350.0])
        aligned = align_angles_to_reference(plan_angles, log_angles)
        steps = np.diff(aligned)
        assert np.all(steps < 0)
        max_step = np.max(np.abs(steps))
        assert max_step < 10.0

    def test_plan_misaligned_wrap(self):
        """Plan angles wrapped, log angles unwrapped"""
        log_angles = np.array([350.0, 355.0, 359.5, 0.5, 5.0, 10.0])
        plan_angles = np.array([350.0, 355.0, 359.5, 0.5, 5.0, 10.0])
        aligned = align_angles_to_reference(plan_angles, log_angles)
        assert aligned[0] == pytest.approx(350.0, abs=0.1)
        assert aligned[-1] > 350.0


def _create_qa_case(
    plan_gantry_raw: np.ndarray,
    log_gantry_raw: np.ndarray,
    num_cp: int = 20,
    num_samples: int = 400,
    max_leaf_pos: float = 50.0,
    noise_std: float = 0.0,
):
    """Helper: create plan+log for QA with matching meterset progression."""
    plan_dict = create_simplified_plan_json(
        plan_uid=f"WRAP-TEST-{np.random.randint(1000)}",
        beam_name="WraparoundBeam",
        num_leaves=60,
        num_control_points=num_cp,
    )

    for i, cp in enumerate(plan_dict["beams"][0]["control_points"]):
        cp["gantry_angle"] = float(plan_gantry_raw[i] % 360.0)
        # Leaf positions scale with meterset weight for predictable pattern
        w = cp["cumulative_meterset_weight"]
        for li in range(60):
            cp["leaf_positions_bank_a"][li] = float(-max_leaf_pos * w)
            cp["leaf_positions_bank_b"][li] = float(max_leaf_pos * w)

    output = io.StringIO()
    writer = csv.writer(output)
    header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
    for i in range(60):
        header.append(f"bank_a_{i}")
    for i in range(60):
        header.append(f"bank_b_{i}")
    writer.writerow(header)

    for idx in range(num_samples):
        t = idx * 30.0 / (num_samples - 1)
        w = idx / (num_samples - 1)
        gantry = (log_gantry_raw[idx] + np.random.normal(0, noise_std)) % 360.0

        # Same leaf pattern as plan
        leaves_a = [-max_leaf_pos * w + np.random.normal(0, noise_std) for _ in range(60)]
        leaves_b = [max_leaf_pos * w + np.random.normal(0, noise_std) for _ in range(60)]
        row = [f"{t:.6f}", "600.0", f"{gantry:.4f}", f"{w:.6f}"]
        row += [f"{v:.6f}" for v in leaves_a]
        row += [f"{v:.6f}" for v in leaves_b]
        writer.writerow(row)

    plan = DicomRTParser.parse_string(json.dumps(plan_dict))
    log = TreatmentLogParser.parse_string(output.getvalue())
    return plan.beams[0], log


class TestQAWithGantryWraparound:
    """End-to-end QA tests with gantry angle wraparound."""

    def test_small_wraparound_clockwise_perfect_match(self):
        """359.8 -> 0.2 degrees clockwise, leaves should match perfectly."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        # Angle from 358 to 362 (wrapping to 358->2) over the course of delivery
        plan_gantry = 358.0 + 4.0 * cp_weights
        log_gantry = 358.0 + 4.0 * log_weights

        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
            noise_std=0.0,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        assert result.max_leaf_deviation_mm < 0.1, (
            f"Expected near-perfect match with wraparound, got max deviation "
            f"{result.max_leaf_deviation_mm:.4f}mm"
        )
        assert result.overall_pass is True
        assert result.control_point_pass_rate_pct > 99.0

        wraparound_warnings = [w for w in result.warnings if "wraparound" in w.lower()]
        assert len(wraparound_warnings) >= 1, "Should detect gantry wraparound"

    def test_small_wraparound_counter_clockwise(self):
        """0.2 -> 359.8 degrees counter-clockwise (decreasing)."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        plan_gantry = 2.0 - 4.0 * cp_weights
        log_gantry = 2.0 - 4.0 * log_weights

        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
            noise_std=0.0,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        assert result.max_leaf_deviation_mm < 0.1, (
            f"Counter-clockwise wraparound: max deviation "
            f"{result.max_leaf_deviation_mm:.4f}mm should be < 0.1mm"
        )
        assert result.overall_pass is True

    def test_boundary_jitter(self):
        """Small jitter around 0/360 boundary, not a true wraparound."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        # Gantry essentially at 0, but with small jitter crossing boundary
        plan_gantry = np.array([0.0 if np.random.random() < 0.5 else 359.9 for _ in range(num_cp)])
        log_gantry = np.array([0.02 if np.random.random() < 0.5 else 359.98 for _ in range(num_samples)])

        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
            noise_std=0.0,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        assert result.max_leaf_deviation_mm < 0.5, (
            f"Jitter case: max deviation {result.max_leaf_deviation_mm:.4f}mm"
        )

    def test_wraparound_with_small_noise(self):
        """Realistic scenario: wraparound with small leaf noise (~0.05mm)."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        plan_gantry = 359.0 + 2.0 * cp_weights
        log_gantry = 359.0 + 2.0 * log_weights

        np.random.seed(42)
        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
            noise_std=0.05,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        assert result.max_leaf_deviation_mm < 1.0, (
            f"Noise case: expected small deviations, got {result.max_leaf_deviation_mm:.4f}mm"
        )
        assert result.rmse_mm < 0.2
        assert result.overall_pass is True

    def test_wraparound_detected_in_warnings(self):
        """Wraparound scenario should produce a warning."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        plan_gantry = 355.0 + 10.0 * cp_weights
        log_gantry = 355.0 + 10.0 * log_weights

        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        has_wrap_warning = any("wraparound" in w.lower() for w in result.warnings)
        assert has_wrap_warning, (
            f"Should detect wraparound, got warnings: {result.warnings}"
        )

    def test_no_wraparound_no_warning(self):
        """Non-wrapping scenario should NOT produce a wraparound warning."""
        num_cp = 20
        num_samples = 400
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        plan_gantry = 30.0 + 50.0 * cp_weights
        log_gantry = 30.0 + 50.0 * log_weights

        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
        )

        calc = MLCQACalculator()
        result = calc.analyze(beam, log)

        has_wrap_warning = any("wraparound" in w.lower() for w in result.warnings)
        assert not has_wrap_warning, (
            f"Normal scenario should not warn about wraparound, got: {result.warnings}"
        )


class TestCriticalRealWorldScenario:
    """The exact scenario from the bug report: 359.8 -> 0.2 causing false failures."""

    def test_359p8_to_0p2_perfect_log(self):
        """Exact user-reported scenario with tiny real noise."""
        num_cp = 25
        num_samples = 500
        cp_weights = np.linspace(0, 1, num_cp)
        log_weights = np.linspace(0, 1, num_samples)

        # Go from 355 to 365 (wraps to 355 -> 5)
        plan_gantry = 355.0 + 10.0 * cp_weights
        log_gantry = 355.0 + 10.0 * log_weights

        np.random.seed(123)
        beam, log = _create_qa_case(
            plan_gantry_raw=plan_gantry,
            log_gantry_raw=log_gantry,
            num_cp=num_cp,
            num_samples=num_samples,
            max_leaf_pos=50.0,
            noise_std=0.02,
        )

        calc = MLCQACalculator(
            leaf_deviation_threshold_mm=1.0,
            control_point_pass_threshold_pct=95.0,
        )
        result = calc.analyze(beam, log)

        # The critical assertions - before fix this would FAIL badly
        assert result.max_leaf_deviation_mm < 0.5, (
            f"BUG STILL PRESENT: 359.8->0.2 wraparound caused max deviation of "
            f"{result.max_leaf_deviation_mm:.4f}mm, should be < 0.5mm"
        )
        assert result.rmse_mm < 0.1, (
            f"RMSE {result.rmse_mm:.4f} should be tiny with small noise"
        )
        assert result.overall_pass is True, (
            f"Perfect-match scenario should pass, got pass rate "
            f"{result.control_point_pass_rate_pct:.1f}%"
        )
        assert result.control_point_pass_rate_pct >= 99.0, (
            f"Pass rate should be 99%+, got {result.control_point_pass_rate_pct:.1f}%"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
