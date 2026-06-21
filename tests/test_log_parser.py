"""Tests for treatment log CSV parser."""
import tempfile
import os
import warnings

import pytest
import numpy as np

from mlc_qa.log_parser import (
    TreatmentLogParser,
    LogParserError,
    MissingDataError,
    LeafCountMismatchError,
    UnevenSamplingWarning,
    create_sample_log_csv,
    create_log_with_gantry_wraparound,
)


class TestLogParserBasic:
    """Basic log parser tests."""

    def test_parse_csv_file(self, sample_log_csv):
        """Test parsing a CSV log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(sample_log_csv)
            temp_path = f.name

        try:
            log = TreatmentLogParser.parse(temp_path)
            assert log.filename == os.path.basename(temp_path)
            assert log.num_samples == 100
            assert log.num_leaves == 60
            assert log.duration_sec > 0
        finally:
            os.unlink(temp_path)

    def test_parse_string(self, sample_log_csv):
        """Test parsing from string content."""
        log = TreatmentLogParser.parse_string(sample_log_csv, filename="test.csv")
        assert log.num_samples == 100
        assert log.num_leaves == 60
        assert log.filename == "test.csv"

    def test_timestamps_extraction(self, sample_log_csv):
        """Test timestamps extraction."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        timestamps = log.get_timestamps()
        assert len(timestamps) == 100
        assert timestamps[0] == 0.0
        assert timestamps[-1] == pytest.approx(30.0, rel=1e-3)
        assert np.all(np.diff(timestamps) > 0)

    def test_dose_rates_extraction(self, sample_log_csv):
        """Test dose rates extraction."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        dose_rates = log.get_dose_rates()
        assert len(dose_rates) == 100
        assert np.all(dose_rates >= 0)
        assert np.mean(dose_rates[10:-10]) == pytest.approx(600.0, rel=1e-3)

    def test_gantry_angles_extraction(self, sample_log_csv):
        """Test gantry angles extraction."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        angles = log.get_gantry_angles(unwrap=False)
        assert len(angles) == 100
        assert np.all(angles >= 0)
        assert np.all(angles <= 360)

    def test_leaf_positions_extraction(self, sample_log_csv):
        """Test leaf positions extraction."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        bank_a = log.get_leaf_positions_bank_a()
        bank_b = log.get_leaf_positions_bank_b()

        assert bank_a.shape == (100, 60)
        assert bank_b.shape == (100, 60)
        assert np.all(bank_a <= 0)
        assert np.all(bank_b >= 0)

    def test_meterset_weights_extraction(self, sample_log_csv):
        """Test meterset weights extraction."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        weights = log.get_meterset_weights()
        assert len(weights) == 100
        assert weights[0] == 0.0
        assert weights[-1] == pytest.approx(1.0, rel=1e-3)
        assert np.all(np.diff(weights) >= 0)

    def test_sampling_statistics(self, sample_log_csv):
        """Test sampling statistics calculation."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        stats = log.get_sampling_statistics()
        assert stats["num_samples"] == 100
        assert stats["is_uniform"] is True
        assert stats["mean_interval_sec"] == pytest.approx(30.0 / 99, rel=1e-3)

    def test_log_to_dict(self, sample_log_csv):
        """Test log to dictionary conversion."""
        log = TreatmentLogParser.parse_string(sample_log_csv)
        log_dict = log.to_dict()
        assert log_dict["num_samples"] == 100
        assert log_dict["num_leaves"] == 60
        assert "sampling_stats" in log_dict


class TestUnevenSampling:
    """Test case 3: Non-uniform log sampling intervals."""

    def test_detect_uneven_sampling(self):
        """Test that uneven sampling is detected."""
        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            uneven_sampling=True,
            noise_std=0.0,
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TreatmentLogParser.parse_string(log_csv)

            assert len(w) >= 1
            assert any(issubclass(warning.category, UnevenSamplingWarning) for warning in w)

        stats = log.get_sampling_statistics()
        assert stats["is_uniform"] is False
        assert stats["cv_pct"] > 10.0

    def test_uneven_sampling_analysis(self, sample_plan_json):
        """Test QA analysis with uneven sampling still works."""
        from mlc_qa.dicom_parser import DicomRTParser
        from mlc_qa.calculations import MLCQACalculator

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            duration_sec=30.0,
            uneven_sampling=True,
            noise_std=0.1,
        )

        plan_str = __import__("json").dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(plan_str)
        beam = plan.beams[0]

        log = TreatmentLogParser.parse_string(log_csv)
        calculator = MLCQACalculator()

        result = calculator.analyze(beam, log)
        assert result is not None
        assert result.num_control_points == 10
        assert len(result.warnings) >= 1
        assert any("uneven" in w.lower() for w in result.warnings)

    def test_duplicate_timestamps(self):
        """Test that duplicate timestamps raise error."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
0.0,600.0,0.0,0.0,-50.0,50.0
0.0,600.0,10.0,0.1,-45.0,45.0
1.0,600.0,20.0,0.2,-40.0,40.0
"""
        with pytest.raises(LogParserError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "strictly increasing" in str(exc_info.value)

    def test_out_of_order_timestamps(self):
        """Test that out-of-order timestamps raise error."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
2.0,600.0,0.0,0.0,-50.0,50.0
1.0,600.0,10.0,0.1,-45.0,45.0
3.0,600.0,20.0,0.2,-40.0,40.0
"""
        with pytest.raises(LogParserError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "strictly increasing" in str(exc_info.value)

    def test_large_cv_sampling(self):
        """Test log with very uneven sampling (high CV)."""
        import io
        import csv

        output = io.StringIO()
        writer = csv.writer(output)

        header = ["timestamp", "dose_rate", "gantry_angle", "meterset_weight"]
        for i in range(60):
            header.append(f"bank_a_{i}")
            header.append(f"bank_b_{i}")
        writer.writerow(header)

        timestamps = [0, 0.01, 0.02, 5.0, 10.0, 25.0, 25.1, 25.2, 25.3, 30.0]
        for idx, t in enumerate(timestamps):
            weight = idx / (len(timestamps) - 1)
            row = [f"{t:.4f}", "600.0", f"{idx * 36:.1f}", f"{weight:.6f}"]
            for i in range(60):
                row.append(f"{-50 * weight:.3f}")
                row.append(f"{50 * weight:.3f}")
            writer.writerow(row)

        csv_content = output.getvalue()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            log = TreatmentLogParser.parse_string(csv_content)
            assert len(w) >= 1

        stats = log.get_sampling_statistics()
        assert stats["cv_pct"] > 100.0
        assert stats["is_uniform"] is False


class TestGantryAngleWraparound:
    """Test case 4: Gantry angle wrapping around 0/360 degrees."""

    def test_gantry_wraparound_detection(self):
        """Test gantry angle wraparound in log data."""
        log_csv = create_log_with_gantry_wraparound(
            num_samples=100,
            num_leaves=60,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        angles_wrapped = log.get_gantry_angles(unwrap=False)
        assert np.any(angles_wrapped > 350)
        assert np.any(angles_wrapped < 10)

        wrap_idx = np.where(np.diff(angles_wrapped) < -180)[0]
        assert len(wrap_idx) > 0

    def test_gantry_unwrap(self):
        """Test that gantry angles are properly unwrapped."""
        log_csv = create_log_with_gantry_wraparound(
            num_samples=100,
            num_leaves=60,
        )
        log = TreatmentLogParser.parse_string(log_csv)

        angles_wrapped = log.get_gantry_angles(unwrap=False)
        angles_unwrapped = log.get_gantry_angles(unwrap=True)

        assert np.max(np.abs(np.diff(angles_unwrapped))) < 180

        assert np.all(np.diff(angles_unwrapped) > 0) or np.all(np.diff(angles_unwrapped) < 0)

    def test_gantry_wraparound_qa_analysis(self, sample_plan_json):
        """Test QA analysis with gantry wraparound."""
        from mlc_qa.dicom_parser import DicomRTParser
        from mlc_qa.calculations import MLCQACalculator

        sample_plan_json["beams"][0]["control_points"][0]["gantry_angle"] = 350.0
        for i, cp in enumerate(sample_plan_json["beams"][0]["control_points"]):
            angle = 350.0 + 20.0 * i / (len(sample_plan_json["beams"][0]["control_points"]) - 1)
            cp["gantry_angle"] = angle

        plan_str = __import__("json").dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(plan_str)
        beam = plan.beams[0]

        log_csv = create_log_with_gantry_wraparound(num_samples=100, num_leaves=60)
        log = TreatmentLogParser.parse_string(log_csv)

        calculator = MLCQACalculator()
        result = calculator.analyze(beam, log)

        assert result is not None
        assert result.gantry_angle_start == pytest.approx(350.0, abs=1.0)
        assert result.control_point_pass_rate_pct > 0

    def test_gantry_angle_normalization(self):
        """Test gantry angle normalization to [0, 360)."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
0.0,600.0,-10.0,0.0,-50.0,50.0
1.0,600.0,370.0,0.5,-40.0,40.0
2.0,600.0,720.0,1.0,-30.0,30.0
"""
        log = TreatmentLogParser.parse_string(csv_content)
        angles = log.get_gantry_angles(unwrap=False)

        assert angles[0] == pytest.approx(350.0)
        assert angles[1] == pytest.approx(10.0)
        assert angles[2] == pytest.approx(0.0)

    def test_gantry_angle_out_of_range(self):
        """Test that out-of-range gantry angles raise error after normalization."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
0.0,600.0,1000.0,0.0,-50.0,50.0
1.0,600.0,-500.0,0.5,-40.0,40.0
"""
        log = TreatmentLogParser.parse_string(csv_content)
        angles = log.get_gantry_angles(unwrap=False)
        assert np.all(angles >= 0)
        assert np.all(angles < 360)


class TestLogParserErrors:
    """Test log parser error conditions."""

    def test_missing_timestamp_column(self):
        """Test missing timestamp column raises error."""
        csv_content = """dose_rate,gantry_angle,bank_a_0,bank_b_0
600.0,0.0,-50.0,50.0
"""
        with pytest.raises(MissingDataError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "Timestamp column not found" in str(exc_info.value)

    def test_missing_dose_rate_column(self):
        """Test missing dose rate column raises error."""
        csv_content = """timestamp,gantry_angle,bank_a_0,bank_b_0
0.0,0.0,-50.0,50.0
"""
        with pytest.raises(MissingDataError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "Dose rate column not found" in str(exc_info.value)

    def test_missing_gantry_column(self):
        """Test missing gantry column raises error."""
        csv_content = """timestamp,dose_rate,bank_a_0,bank_b_0
0.0,600.0,-50.0,50.0
"""
        with pytest.raises(MissingDataError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "Gantry angle column not found" in str(exc_info.value)

    def test_missing_leaf_columns(self):
        """Test missing leaf columns raises error."""
        csv_content = """timestamp,dose_rate,gantry_angle
0.0,600.0,0.0
"""
        with pytest.raises(MissingDataError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "Leaf position columns not found" in str(exc_info.value)

    def test_bank_ab_count_mismatch(self):
        """Test bank A and B have different counts raises error."""
        csv_content = """timestamp,dose_rate,gantry_angle,bank_a_0,bank_a_1,bank_b_0
0.0,600.0,0.0,-50.0,-45.0,50.0
"""
        with pytest.raises(LeafCountMismatchError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "bank A=2, bank B=1" in str(exc_info.value)

    def test_insufficient_samples(self):
        """Test log with only one sample raises error."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
0.0,600.0,0.0,0.0,-50.0,50.0
"""
        with pytest.raises(LogParserError) as exc_info:
            TreatmentLogParser.parse_string(csv_content)
        assert "Insufficient samples" in str(exc_info.value)

    def test_expected_leaf_count_mismatch(self):
        """Test expected leaf count mismatch raises error."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,bank_a_0,bank_b_0
0.0,600.0,0.0,0.0,-50.0,50.0
1.0,600.0,10.0,0.5,-45.0,45.0
"""
        with pytest.raises(LeafCountMismatchError) as exc_info:
            TreatmentLogParser.parse_string(csv_content, num_leaves=60)
        assert "expected 60, got 1" in str(exc_info.value)

    def test_empty_csv(self):
        """Test empty CSV raises error."""
        with pytest.raises(LogParserError):
            TreatmentLogParser.parse_string("")


class TestLogHelperFunctions:
    """Test log helper functions."""

    def test_create_sample_log_default(self):
        """Test creating sample log with defaults."""
        csv_content = create_sample_log_csv()
        assert "timestamp" in csv_content
        assert "bank_a_0" in csv_content

        log = TreatmentLogParser.parse_string(csv_content)
        assert log.num_samples == 100
        assert log.num_leaves == 60

    def test_create_log_with_wraparound(self):
        """Test creating log with gantry wraparound."""
        csv_content = create_log_with_gantry_wraparound(num_samples=50, num_leaves=30)
        log = TreatmentLogParser.parse_string(csv_content)
        assert log.num_samples == 50
        assert log.num_leaves == 30

        angles = log.get_gantry_angles(unwrap=False)
        has_high = np.any(angles > 350)
        has_low = np.any(angles < 10)
        assert has_high and has_low

    def test_alternative_column_names(self):
        """Test parsing with alternative column names."""
        csv_content = """time_sec,doserate,gantry,mu,leaf_0_bank_a,leaf_0_bank_b
0.0,600.0,0.0,0.0,-50.0,50.0
1.0,600.0,10.0,10.0,-45.0,45.0
"""
        log = TreatmentLogParser.parse_string(csv_content)
        assert log.num_samples == 2
        assert log.num_leaves == 1

    def test_mlc_column_names(self):
        """Test parsing with MLC-prefixed column names."""
        csv_content = """timestamp,dose_rate,gantry_angle,meterset_weight,mlc_a_0,mlc_b_0
0.0,600.0,0.0,0.0,-50.0,50.0
1.0,600.0,10.0,0.5,-45.0,45.0
"""
        log = TreatmentLogParser.parse_string(csv_content)
        assert log.num_samples == 2
        assert log.num_leaves == 1
