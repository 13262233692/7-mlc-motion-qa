"""Tests for NumPy calculation core."""
import json

import pytest
import numpy as np

from mlc_qa.dicom_parser import (
    DicomRTParser,
    create_simplified_plan_json,
    BeamData,
    ControlPoint,
)
from mlc_qa.log_parser import (
    TreatmentLogParser,
    TreatmentLog,
    LogSample,
    create_sample_log_csv,
    create_log_with_gantry_wraparound,
)
from mlc_qa.calculations import (
    MLCQACalculator,
    CalculationError,
    InterpolationMethod,
    compute_statistics,
    calculate_gamma_index,
)


class TestCalculatorBasic:
    """Basic calculation core tests."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MLCQACalculator(
            leaf_deviation_threshold_mm=1.0,
            control_point_pass_threshold_pct=95.0,
        )

        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        self.beam = plan.beams[0]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            noise_std=0.2,
        )
        self.log = TreatmentLogParser.parse_string(log_csv)

    def test_basic_analysis(self):
        """Test basic QA analysis completes successfully."""
        result = self.calculator.analyze(self.beam, self.log)

        assert result is not None
        assert result.num_control_points == 10
        assert result.num_leaves == 60
        assert 0 <= result.control_point_pass_rate_pct <= 100
        assert result.max_leaf_deviation_mm >= 0
        assert result.mean_leaf_deviation_mm >= 0
        assert result.rmse_mm >= 0

    def test_analysis_with_zero_noise(self):
        """Test analysis with perfect log (zero noise)."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        result = self.calculator.analyze(self.beam, log)

        assert result.overall_pass is True
        assert result.max_leaf_deviation_mm == pytest.approx(0.0, abs=1e-3)
        assert result.rmse_mm == pytest.approx(0.0, abs=1e-3)
        assert result.control_point_pass_rate_pct == pytest.approx(100.0)

    def test_analysis_with_large_noise(self):
        """Test analysis with large noise (should fail)."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            noise_std=5.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        calculator = MLCQACalculator(
            leaf_deviation_threshold_mm=1.0,
            control_point_pass_threshold_pct=95.0,
        )
        result = calculator.analyze(self.beam, log)

        assert result.overall_pass is False
        assert result.max_leaf_deviation_mm > 1.0
        assert result.num_failed_control_points > 0

    def test_max_deviation_location(self):
        """Test max deviation location is correctly identified."""
        result = self.calculator.analyze(self.beam, self.log)

        assert result.max_deviation_location is not None
        assert result.max_deviation_location.deviation_mm == pytest.approx(
            result.max_leaf_deviation_mm
        )
        assert result.max_deviation_location.bank in ["A", "B"]
        assert 0 <= result.max_deviation_location.leaf_index < self.beam.num_leaves
        assert 0 <= result.max_deviation_location.control_point_index < self.beam.num_control_points

    def test_control_point_results(self):
        """Test control point results are computed correctly."""
        result = self.calculator.analyze(self.beam, self.log)

        assert len(result.control_point_results) == self.beam.num_control_points

        for cp_result in result.control_point_results:
            assert cp_result.max_deviation_mm >= 0
            assert cp_result.mean_deviation_mm >= 0
            assert cp_result.rmse_mm >= 0
            assert cp_result.total_leaves == self.beam.num_leaves * 2
            assert 0 <= cp_result.num_failed_leaves <= cp_result.total_leaves

    def test_leaf_error_sampling(self):
        """Test leaf error sampling."""
        result = self.calculator.analyze(
            self.beam, self.log, sample_errors=True, max_error_samples=50
        )

        assert len(result.leaf_deviations) <= 50
        for dev in result.leaf_deviations:
            assert dev.bank in ["A", "B"]
            assert 0 <= dev.leaf_index < self.beam.num_leaves
            assert dev.deviation_mm >= 0

        deviations = [d.deviation_mm for d in result.leaf_deviations]
        assert deviations == sorted(deviations, reverse=True)

    def test_no_error_sampling(self):
        """Test analysis without error sampling."""
        result = self.calculator.analyze(
            self.beam, self.log, sample_errors=False
        )

        assert len(result.leaf_deviations) == 0

    def test_result_to_dict(self):
        """Test result dictionary conversion."""
        result = self.calculator.analyze(self.beam, self.log)
        result_dict = result.to_dict()

        assert "max_leaf_deviation_mm" in result_dict
        assert "control_point_pass_rate_pct" in result_dict
        assert "overall_pass" in result_dict
        assert isinstance(result_dict["overall_pass"], bool)

    def test_interpolation_methods(self):
        """Test different interpolation methods."""
        for method in InterpolationMethod:
            calculator = MLCQACalculator(interpolation_method=method)
            result = calculator.analyze(self.beam, self.log)
            assert result is not None
            assert result.max_leaf_deviation_mm >= 0

    def test_calculate_gamma_index(self):
        """Test gamma index calculation."""
        planned = np.array([100.0, 80.0, 60.0, 40.0, 20.0])
        actual = np.array([102.0, 79.0, 61.0, 38.0, 21.0])

        pass_rate = calculate_gamma_index(planned, actual, 3.0, 3.0)
        assert 0 <= pass_rate <= 100

        perfect_pass = calculate_gamma_index(planned, planned, 3.0, 3.0)
        assert perfect_pass == pytest.approx(100.0)

    def test_compute_statistics(self):
        """Test statistics computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = compute_statistics(values)

        assert stats["mean"] == pytest.approx(3.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(5.0)
        assert stats["median"] == pytest.approx(3.0)
        assert "std" in stats
        assert "p95" in stats

    def test_compute_statistics_empty(self):
        """Test statistics with empty array."""
        stats = compute_statistics(np.array([]))
        assert stats["mean"] == 0.0
        assert stats["max"] == 0.0


class TestCalculationEdgeCases:
    """Edge case tests for calculation core."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = MLCQACalculator()

    def test_leaf_count_mismatch(self):
        """Test leaf count mismatch between plan and log raises error."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=30,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert "Leaf count mismatch" in str(exc_info.value)

    def test_insufficient_control_points(self):
        """Test insufficient control points raises error."""
        beam = BeamData(
            beam_number=1,
            beam_name="Test Beam",
            beam_type="DYNAMIC",
            energy="6MV",
        )
        beam.control_points = [
            ControlPoint(
                index=0,
                cumulative_meterset_weight=0.0,
                dose_rate=600.0,
                gantry_angle=0.0,
                leaf_positions_bank_a=np.zeros(60),
                leaf_positions_bank_b=np.zeros(60),
            )
        ]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert "at least 2 control points" in str(exc_info.value)

    def test_insufficient_log_samples(self):
        """Test insufficient log samples raises error."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log = TreatmentLog(filename="test.csv")
        log.samples = [
            LogSample(
                timestamp_sec=0.0,
                dose_rate=600.0,
                gantry_angle=0.0,
                leaf_positions_bank_a=np.zeros(60),
                leaf_positions_bank_b=np.zeros(60),
            )
        ]

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert "at least 2 samples" in str(exc_info.value)

    def test_zero_leaves_beam(self):
        """Test beam with zero leaves raises error."""
        beam = BeamData(
            beam_number=1,
            beam_name="Test Beam",
            beam_type="DYNAMIC",
            energy="6MV",
        )
        for i in range(10):
            beam.control_points.append(
                ControlPoint(
                    index=i,
                    cumulative_meterset_weight=i / 9,
                    dose_rate=600.0,
                    gantry_angle=0.0,
                    leaf_positions_bank_a=np.array([]),
                    leaf_positions_bank_b=np.array([]),
                )
            )

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert "no leaf positions" in str(exc_info.value).lower()

    def test_invalid_first_cp_weight(self):
        """Test invalid first control point weight raises error."""
        beam = BeamData(
            beam_number=1,
            beam_name="Test Beam",
            beam_type="DYNAMIC",
            energy="6MV",
        )
        for i in range(10):
            beam.control_points.append(
                ControlPoint(
                    index=i,
                    cumulative_meterset_weight=0.5 + i * 0.05,
                    dose_rate=600.0,
                    gantry_angle=0.0,
                    leaf_positions_bank_a=np.zeros(60),
                    leaf_positions_bank_b=np.zeros(60),
                )
            )

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert "must be 0.0" in str(exc_info.value)

    def test_invalid_last_cp_weight(self):
        """Test invalid last control point weight raises error."""
        beam = BeamData(
            beam_number=1,
            beam_name="Test Beam",
            beam_type="DYNAMIC",
            energy="6MV",
        )
        for i in range(10):
            beam.control_points.append(
                ControlPoint(
                    index=i,
                    cumulative_meterset_weight=i * 0.05,
                    dose_rate=600.0,
                    gantry_angle=0.0,
                    leaf_positions_bank_a=np.zeros(60),
                    leaf_positions_bank_b=np.zeros(60),
                )
            )

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        with pytest.raises(CalculationError) as exc_info:
            self.calculator.analyze(beam, log)
        assert ">= 0.9" in str(exc_info.value)

    def test_gantry_wraparound_analysis(self):
        """Test analysis with gantry wraparound in log."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        for i, cp in enumerate(plan_dict["beams"][0]["control_points"]):
            cp["gantry_angle"] = 350.0 + 20.0 * i / 9

        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log_csv = create_log_with_gantry_wraparound(
            num_samples=100,
            num_leaves=60,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        result = self.calculator.analyze(beam, log)
        assert result is not None
        assert result.num_control_points == 10

    def test_non_monotonic_log_weights(self):
        """Test analysis with non-monotonic log weights."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.1,
        )

        import csv
        import io
        lines = log_csv.split("\n")
        rows = list(csv.reader(io.StringIO(log_csv)))
        rows[50][3] = "0.3"
        rows[51][3] = "0.2"

        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(rows)
        modified_csv = output.getvalue()

        log = TreatmentLogParser.parse_string(modified_csv)

        result = self.calculator.analyze(beam, log)

        assert result is not None
        assert len(result.warnings) >= 1
        assert any("monotonic" in w.lower() for w in result.warnings)

    def test_pass_threshold_adjustment(self):
        """Test that adjusting thresholds changes pass/fail status."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            noise_std=2.0,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        strict_calc = MLCQACalculator(
            leaf_deviation_threshold_mm=0.5,
            control_point_pass_threshold_pct=99.0,
        )
        strict_result = strict_calc.analyze(beam, log)

        lenient_calc = MLCQACalculator(
            leaf_deviation_threshold_mm=10.0,
            control_point_pass_threshold_pct=50.0,
        )
        lenient_result = lenient_calc.analyze(beam, log)

        assert strict_result.overall_pass is False
        assert lenient_result.overall_pass is True
        assert strict_result.num_failed_control_points >= lenient_result.num_failed_control_points
