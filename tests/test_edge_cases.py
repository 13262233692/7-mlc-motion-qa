"""
Comprehensive edge case integration tests.

Covers:
1. Control point missing/incomplete
2. Leaf quantity inconsistency
3. Uneven log sampling
4. Gantry angle wraparound
"""
import json
import warnings

import pytest
import numpy as np

from mlc_qa.dicom_parser import (
    DicomRTParser,
    MissingControlPointError,
    LeafCountMismatchError as DicomLeafCountMismatchError,
    BeamData,
    ControlPoint,
    create_simplified_plan_json,
)
from mlc_qa.log_parser import (
    TreatmentLogParser,
    TreatmentLog,
    LogSample,
    LogParserError,
    LeafCountMismatchError as LogLeafCountMismatchError,
    UnevenSamplingWarning,
    create_sample_log_csv,
    create_log_with_gantry_wraparound,
)
from mlc_qa.calculations import MLCQACalculator, CalculationError


class TestEdgeCase1MissingControlPoints:
    """Test case 1: Missing/incomplete control points."""

    def test_empty_control_points_rejected(self):
        """Test that empty control points are rejected during parsing."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-001",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"] = []

        with pytest.raises(MissingControlPointError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "No control points" in str(exc_info.value)

    def test_single_control_point_rejected(self):
        """Test that single control point is rejected."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-002",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"] = plan_dict["beams"][0]["control_points"][:1]

        with pytest.raises(MissingControlPointError):
            DicomRTParser.parse_string(json.dumps(plan_dict))

    def test_calculator_rejects_insufficient_cp(self):
        """Test calculator rejects beam with insufficient control points."""
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

        calculator = MLCQACalculator()

        with pytest.raises(CalculationError) as exc_info:
            calculator.analyze(beam, log)

        assert "at least 2 control points" in str(exc_info.value)

    def test_missing_cp_weight_field(self):
        """Test missing cumulative_meterset_weight field in CP."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-004",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        del plan_dict["beams"][0]["control_points"][5]["cumulative_meterset_weight"]

        from mlc_qa.dicom_parser import DICOMParserError
        with pytest.raises(DICOMParserError):
            DicomRTParser.parse_string(json.dumps(plan_dict))

    def test_first_cp_nonzero_weight(self):
        """Test first CP with non-zero weight is rejected."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-005",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"][0]["cumulative_meterset_weight"] = 0.1

        from mlc_qa.dicom_parser import DICOMParserError
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "First control point" in str(exc_info.value)

    def test_last_cp_insufficient_weight(self):
        """Test last CP with weight < 0.9 is rejected."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-006",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"][-1]["cumulative_meterset_weight"] = 0.5

        from mlc_qa.dicom_parser import DICOMParserError
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "Last control point" in str(exc_info.value)

    def test_non_monotonic_cp_weights(self):
        """Test non-monotonic CP weights are rejected."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC1-007",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"][5]["cumulative_meterset_weight"] = 0.3
        plan_dict["beams"][0]["control_points"][6]["cumulative_meterset_weight"] = 0.2

        from mlc_qa.dicom_parser import DICOMParserError
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "non-decreasing" in str(exc_info.value)


class TestEdgeCase2LeafQuantityInconsistency:
    """Test case 2: Leaf quantity inconsistency."""

    def test_bank_ab_mismatch_same_cp(self):
        """Test bank A and B have different leaf counts in same CP."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC2-001",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"][0]["leaf_positions_bank_b"] = (
            plan_dict["beams"][0]["control_points"][0]["leaf_positions_bank_b"][:-5]
        )

        with pytest.raises(DicomLeafCountMismatchError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "Bank A and B leaf count mismatch" in str(exc_info.value)

    def test_leaf_count_variation_across_cp(self):
        """Test leaf count changes across control points."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC2-002",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        for i in range(5):
            plan_dict["beams"][0]["control_points"][5 + i]["leaf_positions_bank_a"] = (
                plan_dict["beams"][0]["control_points"][5 + i]["leaf_positions_bank_a"][:-10]
            )
            plan_dict["beams"][0]["control_points"][5 + i]["leaf_positions_bank_b"] = (
                plan_dict["beams"][0]["control_points"][5 + i]["leaf_positions_bank_b"][:-10]
            )

        with pytest.raises(DicomLeafCountMismatchError) as exc_info:
            DicomRTParser.parse_string(json.dumps(plan_dict))

        assert "Leaf count mismatch at control point 5" in str(exc_info.value)

    def test_plan_log_leaf_mismatch_calculator(self):
        """Test calculator rejects plan-log leaf count mismatch."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC2-003",
            beam_name="Test",
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

        calculator = MLCQACalculator()

        with pytest.raises(CalculationError) as exc_info:
            calculator.analyze(beam, log)

        assert "Leaf count mismatch" in str(exc_info.value)

    def test_zero_leaves_plan(self):
        """Test plan with zero leaves is rejected."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC2-004",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        for cp in plan_dict["beams"][0]["control_points"]:
            cp["leaf_positions_bank_a"] = []
            cp["leaf_positions_bank_b"] = []

        with pytest.raises(DicomLeafCountMismatchError):
            DicomRTParser.parse_string(json.dumps(plan_dict))

    def test_log_bank_ab_mismatch(self):
        """Test log with mismatched bank A/B leaf counts is rejected."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_a_1,bank_b_0
0.0,600.0,0.0,0.0,-50.0,-45.0,50.0
1.0,600.0,10.0,0.5,-40.0,-35.0,40.0
"""
        with pytest.raises(LogLeafCountMismatchError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)

        assert "bank A=2, bank B=1" in str(exc_info.value)

    def test_expected_leaf_count_mismatch_log(self):
        """Test log parsing with expected leaf count mismatch."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        with pytest.raises(LogLeafCountMismatchError) as exc_info:
            TreatmentLogParser.parse_string(log_csv, num_leaves=30)

        assert "expected 30, got 60" in str(exc_info.value)


class TestEdgeCase3UnevenLogSampling:
    """Test case 3: Uneven log sampling intervals."""

    def test_uneven_sampling_detection(self):
        """Test that uneven sampling is detected and warned."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            uneven_sampling=True,
            noise_std=0.0,
        )

        warnings.filterwarnings("always", category=UnevenSamplingWarning)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=UnevenSamplingWarning)
            log = TreatmentLogParser.parse_string(log_csv)

            uneven_warnings = [
                warning for warning in w
                if issubclass(warning.category, UnevenSamplingWarning)
            ]
            assert len(uneven_warnings) >= 1, f"Expected UnevenSamplingWarning, got: {[x.category.__name__ for x in w]}"

        stats = log.get_sampling_statistics()
        assert stats["is_uniform"] is False
        assert stats["cv_pct"] > 10.0

    def test_uneven_sampling_qa_analysis(self):
        """Test QA analysis completes with uneven sampling."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC3-001",
            beam_name="Test",
            num_leaves=60,
            num_control_points=10,
        )
        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            uneven_sampling=True,
            noise_std=0.1,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        calculator = MLCQACalculator()
        result = calculator.analyze(beam, log)

        assert result is not None
        assert result.num_control_points == 10
        assert len(result.warnings) >= 1
        assert any("uneven" in w.lower() for w in result.warnings)

    def test_extremely_uneven_sampling(self):
        """Test log with extremely uneven sampling (burst pattern)."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        timestamps = []
        t = 0.0
        for i in range(10):
            timestamps.append(t)
            t += 0.001

        t += 10.0
        for i in range(10):
            timestamps.append(t)
            t += 0.001

        t += 10.0
        for i in range(10):
            timestamps.append(t)
            t += 0.001

        for idx, t in enumerate(timestamps):
            weight = idx / (len(timestamps) - 1)
            row = [f"{t:.6f}", "600.0", f"{idx * 12:.1f}", f"{weight:.6f}"]
            for i in range(60):
                row.append(f"{-50.0 * weight:.3f}")
                row.append(f"{50.0 * weight:.3f}")
            writer.writerow(row)

        csv_content = output.getvalue()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TreatmentLogParser.parse_string(csv_content)
            assert len([
                warning for warning in w
                if issubclass(warning.category, UnevenSamplingWarning)
            ]) >= 1

        stats = log.get_sampling_statistics()
        assert stats["cv_pct"] > 200.0

    def test_duplicate_timestamps_rejected(self):
        """Test duplicate timestamps are rejected."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
        for i in range(60):
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        writer.writerow([0.0, 600.0, 0.0, 0.0] + [-50.0] * 120)
        writer.writerow([0.0, 600.0, 10.0, 0.1] + [-45.0] * 120)
        writer.writerow([1.0, 600.0, 20.0, 0.2] + [-40.0] * 120)

        csv_content = output.getvalue()

        with pytest.raises(LogParserError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)

        assert "strictly increasing" in str(exc_info.value)

    def test_decreasing_timestamps_rejected(self):
        """Test decreasing timestamps are rejected."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
        for i in range(60):
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        writer.writerow([2.0, 600.0, 0.0, 0.0] + [-50.0] * 120)
        writer.writerow([1.0, 600.0, 10.0, 0.1] + [-45.0] * 120)
        writer.writerow([3.0, 600.0, 20.0, 0.2] + [-40.0] * 120)

        csv_content = output.getvalue()

        with pytest.raises(LogParserError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)

        assert "strictly increasing" in str(exc_info.value)


class TestEdgeCase4GantryAngleWraparound:
    """Test case 4: Gantry angle wrapping around 0/360 degrees."""

    def test_gantry_wraparound_detection(self):
        """Test gantry angle wraparound in log."""
        log_csv = create_log_with_gantry_wraparound(
            num_samples=100,
            num_leaves=60,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        angles = log.get_gantry_angles(unwrap=False)
        assert np.any(angles > 350)
        assert np.any(angles < 10)

        wrap_points = np.where(np.diff(angles) < -180)[0]
        assert len(wrap_points) > 0

    def test_gantry_unwrap_correctness(self):
        """Test gantry angle unwrapping produces smooth angles."""
        log_csv = create_log_with_gantry_wraparound(
            num_samples=100,
            num_leaves=60,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        angles_wrapped = log.get_gantry_angles(unwrap=False)
        angles_unwrapped = log.get_gantry_angles(unwrap=True)

        max_step_wrapped = np.max(np.abs(np.diff(angles_wrapped)))
        max_step_unwrapped = np.max(np.abs(np.diff(angles_unwrapped)))

        assert max_step_unwrapped < max_step_wrapped
        assert max_step_unwrapped < 10.0

        assert np.all(np.diff(angles_unwrapped) > 0)

    def test_gantry_wraparound_qa_analysis(self):
        """Test QA analysis with gantry wraparound works correctly."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC4-001",
            beam_name="Test",
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

        calculator = MLCQACalculator()
        result = calculator.analyze(beam, log)

        assert result is not None
        assert result.gantry_angle_start == pytest.approx(350.0, abs=1.0)
        assert result.control_point_pass_rate_pct > 0
        assert result.num_control_points == 10

    def test_gantry_angle_normalization_negative(self):
        """Test negative gantry angles are normalized to [0, 360)."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
        for i in range(60):
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        writer.writerow([0.0, 600.0, -90.0, 0.0] + [-50.0] * 120)
        writer.writerow([1.0, 600.0, -45.0, 0.5] + [-45.0] * 120)
        writer.writerow([2.0, 600.0, 0.0, 1.0] + [-40.0] * 120)

        csv_content = output.getvalue()

        log = TreatmentLogParser.parse_string(csv_content)
        angles = log.get_gantry_angles(unwrap=False)

        assert angles[0] == pytest.approx(270.0)
        assert angles[1] == pytest.approx(315.0)
        assert angles[2] == pytest.approx(0.0)

    def test_gantry_angle_normalization_over_360(self):
        """Test gantry angles > 360 are normalized correctly."""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
        for i in range(60):
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        writer.writerow([0.0, 600.0, 370.0, 0.0] + [-50.0] * 120)
        writer.writerow([1.0, 600.0, 720.0, 0.5] + [-45.0] * 120)
        writer.writerow([2.0, 600.0, 1080.0, 1.0] + [-40.0] * 120)

        csv_content = output.getvalue()

        log = TreatmentLogParser.parse_string(csv_content)
        angles = log.get_gantry_angles(unwrap=False)

        assert angles[0] == pytest.approx(10.0)
        assert angles[1] == pytest.approx(0.0)
        assert angles[2] == pytest.approx(0.0)

    def test_gantry_wraparound_full_circle(self):
        """Test QA with gantry making full 360 rotation through wraparound."""
        plan_dict = create_simplified_plan_json(
            plan_uid="TEST-EC4-002",
            beam_name="Test",
            num_leaves=60,
            num_control_points=20,
        )
        for i, cp in enumerate(plan_dict["beams"][0]["control_points"]):
            cp["gantry_angle"] = (350.0 + 20.0 * i) % 360.0

        plan = DicomRTParser.parse_string(json.dumps(plan_dict))
        beam = plan.beams[0]

        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        num_samples = 200
        for idx in range(num_samples):
            t = 30.0 * idx / (num_samples - 1)
            weight = idx / (num_samples - 1)
            gantry = (350.0 + 20.0 * idx * 19 / (num_samples - 1)) % 360.0

            row = [f"{t:.4f}", "600.0", f"{gantry:.1f}", f"{weight:.6f}"]
            leaf_pos = 100.0 * weight
            for i in range(60):
                row.append(f"{-leaf_pos:.3f}")
                row.append(f"{leaf_pos:.3f}")
            writer.writerow(row)

        log_csv = output.getvalue()
        log = TreatmentLogParser.parse_string(log_csv)

        calculator = MLCQACalculator()
        result = calculator.analyze(beam, log)

        assert result is not None
        assert result.num_control_points == 20
        assert result.overall_pass is True


class TestEndToEndEdgeCases:
    """End-to-end tests for all edge cases through API."""

    def test_api_rejects_missing_control_points(self, client):
        """Test API rejects plan with missing CP."""
        plan_dict = create_simplified_plan_json(
            plan_uid="BAD-PLAN",
            beam_name="Bad",
            num_leaves=60,
            num_control_points=10,
        )
        plan_dict["beams"][0]["control_points"] = []

        log_csv = create_sample_log_csv(num_samples=100, num_leaves=60)

        files = {
            "plan_file": ("plan.json", json.dumps(plan_dict).encode(), "application/json"),
            "log_file": ("log.csv", log_csv.encode(), "text/csv"),
        }
        params = {"patient_anonymous_id": "PAT-001", "beam_name": "Bad"}

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 400

    def test_api_rejects_leaf_mismatch(self, client, sample_plan_json):
        """Test API rejects plan-log leaf mismatch."""
        log_csv = create_sample_log_csv(num_samples=100, num_leaves=30)

        files = {
            "plan_file": ("plan.json", json.dumps(sample_plan_json).encode(), "application/json"),
            "log_file": ("log.csv", log_csv.encode(), "text/csv"),
        }
        params = {"patient_anonymous_id": "PAT-001", "beam_name": "AP Field"}

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 400

    def test_api_handles_uneven_sampling(self, client, sample_plan_json):
        """Test API handles uneven sampling log."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            uneven_sampling=True,
            noise_std=0.1,
        )

        files = {
            "plan_file": ("plan.json", json.dumps(sample_plan_json).encode(), "application/json"),
            "log_file": ("log.csv", log_csv.encode(), "text/csv"),
        }
        params = {"patient_anonymous_id": "PAT-001", "beam_name": "AP Field"}

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 200

    def test_api_handles_gantry_wraparound(self, client, sample_plan_json):
        """Test API handles gantry wraparound log."""
        sample_plan_json["beams"][0]["control_points"][0]["gantry_angle"] = 350.0
        for i, cp in enumerate(sample_plan_json["beams"][0]["control_points"]):
            cp["gantry_angle"] = 350.0 + 20.0 * i / 9

        log_csv = create_log_with_gantry_wraparound(num_samples=100, num_leaves=60)

        files = {
            "plan_file": ("plan.json", json.dumps(sample_plan_json).encode(), "application/json"),
            "log_file": ("log.csv", log_csv.encode(), "text/csv"),
        }
        params = {"patient_anonymous_id": "PAT-001", "beam_name": "AP Field"}

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 200
