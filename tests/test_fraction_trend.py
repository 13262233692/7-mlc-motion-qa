"""Tests for fraction trend analysis and fraction_qa_summary."""
import io
import json

import pytest
import numpy as np

from mlc_qa.calculations import (
    TrendAnalyzer,
    FractionMetrics,
    TrendLabel,
)
from mlc_qa.dicom_parser import create_simplified_plan_json
from mlc_qa.log_parser import create_sample_log_csv


class TestTrendAnalyzer:
    """Tests for the TrendAnalyzer core algorithm."""

    def test_stable_normal(self):
        """Test stable normal trend detection."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=0.5 + np.random.normal(0, 0.02),
                mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3,
                pass_rate_pct=90.0 + np.random.normal(0, 1.0),
                overall_pass=i % 3 != 0,
            )
            for i in range(1, 11)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.STABLE_NORMAL
        assert result.total_fractions == 10

    def test_gradual_increase(self):
        """Test gradual increase trend detection."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=0.3 + i * 0.08,
                mean_leaf_deviation_mm=0.15 + i * 0.04,
                rmse_mm=0.2 + i * 0.05,
                pass_rate_pct=99.0 - i * 0.5,
                overall_pass=True,
            )
            for i in range(1, 11)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label in (TrendLabel.GRADUAL_INCREASE, TrendLabel.SHARP_INCREASE)
        assert result.max_deviation_slope > 0

    def test_improving(self):
        """Test improving (decreasing) trend detection."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=1.5 - i * 0.1,
                mean_leaf_deviation_mm=0.8 - i * 0.05,
                rmse_mm=1.0 - i * 0.07,
                pass_rate_pct=90.0 + i * 0.8,
                overall_pass=False if i < 3 else True,
            )
            for i in range(1, 11)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.IMPROVING
        assert result.max_deviation_slope < 0

    def test_single_spike(self):
        """Test single spike detection."""
        values = [0.5] * 10
        values[5] = 2.0
        metrics = [
            FractionMetrics(
                fraction_number=i + 1,
                plan_version=1,
                max_leaf_deviation_mm=v,
                mean_leaf_deviation_mm=v * 0.4,
                rmse_mm=v * 0.6,
                pass_rate_pct=98.0 if v < 1.0 else 80.0,
                overall_pass=v < 1.0,
            )
            for i, v in enumerate(values)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.SINGLE_SPIKE
        assert len(result.anomaly_flags) > 0

    def test_erratic(self):
        """Test erratic (highly variable) trend detection."""
        values = [0.5, 1.2, 0.3, 1.5, 0.4, 1.1, 0.2, 1.3, 0.5, 1.0]
        metrics = [
            FractionMetrics(
                fraction_number=i + 1,
                plan_version=1,
                max_leaf_deviation_mm=v,
                mean_leaf_deviation_mm=v * 0.5,
                rmse_mm=v * 0.7,
                pass_rate_pct=95.0 - v * 10,
                overall_pass=v < 1.0,
            )
            for i, v in enumerate(values)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.ERRATIC
        assert len(result.anomaly_flags) > 0

    def test_insufficient_data_single_point(self):
        """Test insufficient data with single point."""
        metrics = [FractionMetrics(
            fraction_number=1,
            plan_version=1,
            max_leaf_deviation_mm=0.5,
            mean_leaf_deviation_mm=0.2,
            rmse_mm=0.3,
            pass_rate_pct=95.0,
            overall_pass=True,
        )]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.INSUFFICIENT_DATA
        assert result.total_fractions == 1

    def test_insufficient_data_empty(self):
        """Test insufficient data with empty list."""
        analyzer = TrendAnalyzer()
        result = analyzer.analyze([])
        assert result.trend_label == TrendLabel.INSUFFICIENT_DATA
        assert result.total_fractions == 0

    def test_all_pass(self):
        """Test all pass detection (all below threshold)."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=0.1 + i * 0.01,
                mean_leaf_deviation_mm=0.05,
                rmse_mm=0.08,
                pass_rate_pct=99.0 + i * 0.1,
                overall_pass=True,
            )
            for i in range(1, 11)
        ]
        analyzer = TrendAnalyzer(deviation_threshold_mm=0.5)
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.ALL_PASS

    def test_all_fail(self):
        """Test all fail detection (all above threshold)."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=2.0 + i * 0.1,
                mean_leaf_deviation_mm=1.0,
                rmse_mm=1.5,
                pass_rate_pct=70.0 + i * 0.5,
                overall_pass=False,
            )
            for i in range(1, 11)
        ]
        analyzer = TrendAnalyzer(deviation_threshold_mm=1.0, pass_rate_threshold_pct=90.0)
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.ALL_FAIL
        assert len(result.anomaly_flags) > 0

    def test_chart_data_structure(self):
        """Test chart data structure in result."""
        metrics = [
            FractionMetrics(
                fraction_number=i,
                plan_version=1,
                max_leaf_deviation_mm=0.5 + i * 0.05,
                mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3,
                pass_rate_pct=98.0,
                overall_pass=True,
            )
            for i in range(1, 6)
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert "fraction_numbers" in result.chart_data
        assert "max_deviation_mm" in result.chart_data
        assert "pass_rate_pct" in result.chart_data
        assert len(result.chart_data["fraction_numbers"]) == 5

    def test_linear_regression_slope(self):
        """Test linear regression slope calculation."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        slope, r = TrendAnalyzer._linear_regression_slope(x, y)
        assert abs(slope - 2.0) < 0.001
        assert abs(r - 1.0) < 0.001

    def test_detect_single_spike(self):
        """Test single spike detection."""
        values = np.array([1.0, 1.1, 1.0, 3.0, 1.0, 1.0])
        analyzer = TrendAnalyzer(spike_threshold_mm=2.0)
        spike_idx = analyzer._detect_single_spike(values)
        assert spike_idx == 3

    def test_detect_single_spike_none(self):
        """Test no spike detected for smooth data."""
        values = np.array([1.0, 1.1, 1.2, 1.1, 1.0, 1.1])
        analyzer = TrendAnalyzer()
        spike_idx = analyzer._detect_single_spike(values)
        assert spike_idx is None

    def test_description_not_empty(self):
        """Test that description is always a non-empty string."""
        test_cases = [
            [FractionMetrics(
                fraction_number=1, plan_version=1,
                max_leaf_deviation_mm=0.5, mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3, pass_rate_pct=95.0, overall_pass=True,
            )],
            [FractionMetrics(
                fraction_number=i, plan_version=1,
                max_leaf_deviation_mm=0.5, mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3, pass_rate_pct=95.0, overall_pass=True,
            ) for i in range(1, 6)],
        ]
        analyzer = TrendAnalyzer()
        for metrics in test_cases:
            result = analyzer.analyze(metrics)
            assert isinstance(result.overall_description, str)
            assert len(result.overall_description) > 0

    def test_plan_version_handling(self):
        """Test that plan version changes are detected."""
        metrics = [
            FractionMetrics(
                fraction_number=1, plan_version=1,
                max_leaf_deviation_mm=0.8, mean_leaf_deviation_mm=0.3,
                rmse_mm=0.5, pass_rate_pct=92.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=2, plan_version=1,
                max_leaf_deviation_mm=0.85, mean_leaf_deviation_mm=0.32,
                rmse_mm=0.52, pass_rate_pct=91.5, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=1, plan_version=2,
                max_leaf_deviation_mm=0.4, mean_leaf_deviation_mm=0.15,
                rmse_mm=0.25, pass_rate_pct=98.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=2, plan_version=2,
                max_leaf_deviation_mm=0.42, mean_leaf_deviation_mm=0.16,
                rmse_mm=0.26, pass_rate_pct=97.8, overall_pass=True,
            ),
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.total_fractions == 4
        assert len(result.plan_versions) == 2
        assert 1 in result.plan_versions
        assert 2 in result.plan_versions
        assert any("re-planned" in flag.lower() or "plan" in flag.lower()
                   for flag in result.anomaly_flags)


class TestFractionQASummaryCRUD:
    """Tests for FractionQASummary CRUD operations."""

    def test_create_summary(self, db_session):
        """Test creating a fraction QA summary."""
        from mlc_qa import crud, schemas, models

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-001")

        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-001",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        summary_in = schemas.FractionQASummaryCreate(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=1,
            num_qa_results=1,
            max_leaf_deviation_mm=0.5,
            mean_leaf_deviation_mm=0.2,
            rmse_mm=0.25,
            dose_rate_deviation_pct=0.5,
            control_point_pass_rate_pct=95.0,
            overall_pass_rate_pct=100.0,
            trend_label="STABLE_NORMAL",
            trend_confidence=0.9,
            deviation_delta_from_previous_mm=0.0,
        )
        summary = crud.fraction_qa_summary.create(db_session, summary_in)

        assert summary.id is not None
        assert summary.fraction_number == 1
        assert summary.plan_version == 1
        assert summary.max_leaf_deviation_mm == 0.5
        assert summary.trend_label == "STABLE_NORMAL"

    def test_get_by_fraction(self, db_session):
        """Test getting summary by plan+beam+fraction+version."""
        from mlc_qa import crud, schemas

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-002")
        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-002",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        summary_in = schemas.FractionQASummaryCreate(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=3,
            plan_version=2,
            num_qa_results=1,
            max_leaf_deviation_mm=0.3,
            mean_leaf_deviation_mm=0.1,
            rmse_mm=0.15,
            overall_pass_rate_pct=100.0,
        )
        crud.fraction_qa_summary.create(db_session, summary_in)

        found = crud.fraction_qa_summary.get_by_fraction(
            db_session, plan_db.id, beam_db.id, 3, plan_version=2
        )
        assert found is not None
        assert found.fraction_number == 3
        assert found.plan_version == 2

    def test_list_by_plan_beam(self, db_session):
        """Test listing summaries by plan+beam."""
        from mlc_qa import crud, schemas

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-003")
        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-003",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        for i in range(1, 6):
            summary_in = schemas.FractionQASummaryCreate(
                plan_id=plan_db.id,
                beam_id=beam_db.id,
                fraction_number=i,
                plan_version=1,
                num_qa_results=1,
                max_leaf_deviation_mm=0.3 + i * 0.05,
                rmse_mm=0.15 + i * 0.02,
                overall_pass_rate_pct=98.0,
            )
            crud.fraction_qa_summary.create(db_session, summary_in)

        summaries = crud.fraction_qa_summary.list_by_plan_beam(
            db_session, plan_db.id, beam_db.id
        )
        assert len(summaries) == 5
        assert summaries[0].fraction_number == 1
        assert summaries[-1].fraction_number == 5

    def test_update_from_qa_result_first_time(self, db_session):
        """Test update_from_qa_result creates summary on first result."""
        from mlc_qa import crud, models

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-004")
        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-004",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        qa_result = models.QAResult(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=1,
            max_leaf_deviation_mm=0.5,
            mean_leaf_deviation_mm=0.2,
            rmse_mm=0.3,
            dose_rate_deviation_pct=0.5,
            control_point_pass_rate_pct=95.0,
            num_control_points=10,
            num_failed_control_points=1,
            num_leaves=60,
            overall_pass=1,
        )
        db_session.add(qa_result)
        db_session.commit()
        db_session.refresh(qa_result)

        summary = crud.fraction_qa_summary.update_from_qa_result(db_session, qa_result)
        assert summary is not None
        assert summary.num_qa_results == 1
        assert summary.max_leaf_deviation_mm == 0.5
        assert summary.overall_pass_rate_pct == 100.0

    def test_update_from_qa_result_duplicate_upload(self, db_session):
        """Test duplicate upload (same fraction) aggregates correctly."""
        from mlc_qa import crud, models

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-005")
        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-005",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        qa_result1 = models.QAResult(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=1,
            max_leaf_deviation_mm=0.5,
            mean_leaf_deviation_mm=0.2,
            rmse_mm=0.3,
            dose_rate_deviation_pct=0.5,
            control_point_pass_rate_pct=90.0,
            num_control_points=10,
            num_failed_control_points=1,
            num_leaves=60,
            overall_pass=1,
        )
        db_session.add(qa_result1)
        db_session.commit()
        db_session.refresh(qa_result1)
        crud.fraction_qa_summary.update_from_qa_result(db_session, qa_result1)

        qa_result2 = models.QAResult(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=1,
            max_leaf_deviation_mm=0.7,
            mean_leaf_deviation_mm=0.3,
            rmse_mm=0.4,
            dose_rate_deviation_pct=0.6,
            control_point_pass_rate_pct=85.0,
            num_control_points=10,
            num_failed_control_points=2,
            num_leaves=60,
            overall_pass=0,
        )
        db_session.add(qa_result2)
        db_session.commit()
        db_session.refresh(qa_result2)
        summary = crud.fraction_qa_summary.update_from_qa_result(db_session, qa_result2)

        assert summary.num_qa_results == 2
        assert summary.max_leaf_deviation_mm == 0.7
        assert summary.overall_pass_rate_pct == 50.0
        assert summary.rmse_mm == pytest.approx(0.35)

    def test_plan_version_separation(self, db_session):
        """Test that different plan versions are stored separately."""
        from mlc_qa import crud, models

        patient = crud.patient_alias.get_or_create(db_session, "test-patient-006")
        plan_data = create_simplified_plan_json(
            plan_uid="TEST-FRAC-006",
            beam_name="Test Beam",
            num_leaves=60,
            num_control_points=10,
        )
        from mlc_qa.dicom_parser import DicomRTParser
        plan_parsed = DicomRTParser.parse_string(json.dumps(plan_data))
        plan_db = crud.plan.create_from_plan_data(db_session, patient.id, plan_parsed)
        beam_db = crud.beam.get_by_plan_and_name(db_session, plan_db.id, "Test Beam")

        qa_v1 = models.QAResult(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=1,
            max_leaf_deviation_mm=0.5,
            mean_leaf_deviation_mm=0.2,
            rmse_mm=0.3,
            dose_rate_deviation_pct=0.5,
            control_point_pass_rate_pct=95.0,
            num_control_points=10,
            num_failed_control_points=1,
            num_leaves=60,
            overall_pass=1,
        )
        db_session.add(qa_v1)
        db_session.commit()
        db_session.refresh(qa_v1)
        crud.fraction_qa_summary.update_from_qa_result(db_session, qa_v1)

        qa_v2 = models.QAResult(
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            fraction_number=1,
            plan_version=2,
            max_leaf_deviation_mm=0.3,
            mean_leaf_deviation_mm=0.1,
            rmse_mm=0.15,
            dose_rate_deviation_pct=0.3,
            control_point_pass_rate_pct=99.0,
            num_control_points=10,
            num_failed_control_points=0,
            num_leaves=60,
            overall_pass=1,
        )
        db_session.add(qa_v2)
        db_session.commit()
        db_session.refresh(qa_v2)
        crud.fraction_qa_summary.update_from_qa_result(db_session, qa_v2)

        v1_summary = crud.fraction_qa_summary.get_by_fraction(
            db_session, plan_db.id, beam_db.id, 1, plan_version=1
        )
        v2_summary = crud.fraction_qa_summary.get_by_fraction(
            db_session, plan_db.id, beam_db.id, 1, plan_version=2
        )

        assert v1_summary is not None
        assert v2_summary is not None
        assert v1_summary.max_leaf_deviation_mm == 0.5
        assert v2_summary.max_leaf_deviation_mm == 0.3
        assert v1_summary.id != v2_summary.id


class TestFractionTrendAPI:
    """Tests for fraction trend API endpoints."""

    def _submit_qa(self, client, sample_plan_json, sample_log_csv,
                   fraction_number=1, plan_version=1,
                   patient_id="test-patient"):
        """Helper to submit a QA result."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")

        response = client.post(
            "/api/qa/submit",
            params={
                "patient_anonymous_id": patient_id,
                "beam_name": "AP Field",
                "fraction_number": fraction_number,
                "plan_version": plan_version,
            },
            files={
                "plan_file": ("plan.json", plan_content, "application/json"),
                "log_file": ("log.csv", sample_log_csv.encode("utf-8"), "text/csv"),
            },
        )
        return response

    def _get_plan_and_beam(self, client, patient_id="test-patient"):
        """Helper to get first plan and beam IDs."""
        plans_response = client.get(
            f"/api/patients/{patient_id}/plans"
        )
        plans = plans_response.json()
        assert len(plans) > 0
        plan_id = plans[0]["id"]

        plan_detail_response = client.get(f"/api/plans/{plan_id}")
        plan_detail = plan_detail_response.json()
        beam_id = plan_detail["beams"][0]["id"]
        return plan_id, beam_id

    def test_submit_with_fraction_number(self, client, sample_plan_json, sample_log_csv):
        """Test QA submit with fraction_number and plan_version."""
        response = self._submit_qa(
            client, sample_plan_json, sample_log_csv,
            fraction_number=3, plan_version=2,
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["fraction_number"] == 3
        assert data["plan_version"] == 2
        assert "fraction_summary_id" in data

    def test_list_fraction_summaries(self, client, sample_plan_json, sample_log_csv):
        """Test listing fraction summaries via API."""
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=1, patient_id="patient-trend-1")
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=2, patient_id="patient-trend-1")

        plan_id, beam_id = self._get_plan_and_beam(client, "patient-trend-1")

        response = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/fractions"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 2

    def test_get_fraction_summary(self, client, sample_plan_json, sample_log_csv):
        """Test getting a specific fraction summary."""
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=5, patient_id="patient-trend-2")

        plan_id, beam_id = self._get_plan_and_beam(client, "patient-trend-2")

        response = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/fractions/5"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["fraction_number"] == 5

    def test_trend_analysis_endpoint(self, client, sample_plan_json, sample_log_csv):
        """Test trend analysis API endpoint."""
        for i in range(1, 6):
            self._submit_qa(
                client, sample_plan_json, sample_log_csv,
                fraction_number=i,
                patient_id="patient-trend-3",
            )

        plan_id, beam_id = self._get_plan_and_beam(client, "patient-trend-3")

        response = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/trend",
        )
        assert response.status_code == 200
        data = response.json()
        assert "trend_label" in data
        assert "chart_data" in data
        assert "anomaly_flags" in data
        assert "total_fractions" in data
        assert data["total_fractions"] == 5
        assert "fractions" in data
        assert len(data["fractions"]) == 5

    def test_trend_analysis_missing_data(self, client, sample_plan_json, sample_log_csv):
        """Test trend analysis with no data returns 404."""
        plan_response = client.get("/api/plans")
        assert len(plan_response.json()) == 0

        response = client.get(
            "/api/plans/99999/beams/99999/trend"
        )
        assert response.status_code == 404

    def test_plan_version_filter(self, client, sample_plan_json, sample_log_csv):
        """Test trend analysis filtered by plan version."""
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=1, plan_version=1,
                        patient_id="patient-version-test")
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=2, plan_version=2,
                        patient_id="patient-version-test")

        plan_id, beam_id = self._get_plan_and_beam(client, "patient-version-test")

        response_v1 = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/fractions",
            params={"plan_version": 1},
        )
        assert response_v1.status_code == 200
        assert len(response_v1.json()) == 1

        response_v2 = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/fractions",
            params={"plan_version": 2},
        )
        assert response_v2.status_code == 200
        assert len(response_v2.json()) == 1

        response_all = client.get(
            f"/api/plans/{plan_id}/beams/{beam_id}/fractions",
        )
        assert response_all.status_code == 200
        assert len(response_all.json()) == 2

    def test_list_plan_fractions_all_beams(self, client, sample_plan_json, sample_log_csv):
        """Test listing all fractions for a plan (all beams)."""
        self._submit_qa(client, sample_plan_json, sample_log_csv,
                        fraction_number=1, patient_id="patient-plan-frac")

        plans_response = client.get("/api/patients/patient-plan-frac/plans")
        plan_id = plans_response.json()[0]["id"]

        response = client.get(f"/api/plans/{plan_id}/fractions")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1


class TestMissingFractions:
    """Tests for handling missing fractions in trend analysis."""

    def test_gaps_in_fraction_numbers(self):
        """Test trend analysis with gaps (missing fractions)."""
        metrics = [
            FractionMetrics(
                fraction_number=1, plan_version=1,
                max_leaf_deviation_mm=0.3, mean_leaf_deviation_mm=0.1,
                rmse_mm=0.2, pass_rate_pct=99.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=3, plan_version=1,
                max_leaf_deviation_mm=0.5, mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3, pass_rate_pct=97.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=5, plan_version=1,
                max_leaf_deviation_mm=0.7, mean_leaf_deviation_mm=0.3,
                rmse_mm=0.4, pass_rate_pct=93.0, overall_pass=False,
            ),
            FractionMetrics(
                fraction_number=10, plan_version=1,
                max_leaf_deviation_mm=0.4, mean_leaf_deviation_mm=0.15,
                rmse_mm=0.25, pass_rate_pct=98.0, overall_pass=True,
            ),
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.total_fractions == 4
        assert len(result.chart_data["fraction_numbers"]) == 4
        assert result.trend_label in (
            TrendLabel.GRADUAL_INCREASE,
            TrendLabel.SHARP_INCREASE,
            TrendLabel.ERRATIC,
            TrendLabel.STABLE_NORMAL,
            TrendLabel.SINGLE_SPIKE,
        )

    def test_non_sequential_fractions(self):
        """Test with non-sequential fraction numbers (e.g. 1, 5, 10)."""
        metrics = [
            FractionMetrics(
                fraction_number=1, plan_version=1,
                max_leaf_deviation_mm=0.5, mean_leaf_deviation_mm=0.2,
                rmse_mm=0.3, pass_rate_pct=96.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=5, plan_version=1,
                max_leaf_deviation_mm=0.6, mean_leaf_deviation_mm=0.25,
                rmse_mm=0.35, pass_rate_pct=95.5, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=10, plan_version=1,
                max_leaf_deviation_mm=0.7, mean_leaf_deviation_mm=0.3,
                rmse_mm=0.4, pass_rate_pct=95.0, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=15, plan_version=1,
                max_leaf_deviation_mm=0.8, mean_leaf_deviation_mm=0.35,
                rmse_mm=0.45, pass_rate_pct=94.5, overall_pass=True,
            ),
            FractionMetrics(
                fraction_number=20, plan_version=1,
                max_leaf_deviation_mm=0.9, mean_leaf_deviation_mm=0.4,
                rmse_mm=0.5, pass_rate_pct=94.0, overall_pass=True,
            ),
        ]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.total_fractions == 5
        assert result.max_deviation_slope > 0

    def test_single_fraction_no_trend(self):
        """Test single fraction gives INSUFFICIENT_DATA."""
        metrics = [FractionMetrics(
            fraction_number=10, plan_version=1,
            max_leaf_deviation_mm=0.5, mean_leaf_deviation_mm=0.2,
            rmse_mm=0.3, pass_rate_pct=95.0, overall_pass=True,
        )]
        analyzer = TrendAnalyzer()
        result = analyzer.analyze(metrics)
        assert result.trend_label == TrendLabel.INSUFFICIENT_DATA
        assert "Insufficient" in result.overall_description


class TestReportWithTrend:
    """Tests for PDF report with trend data."""

    def test_pdf_export_with_trend(self, client, sample_plan_json, sample_log_csv):
        """Test PDF export includes trend data when available."""
        for i in range(1, 4):
            self._submit_qa_for_report(
                client, sample_plan_json, sample_log_csv,
                fraction_number=i,
                patient_id="patient-pdf-trend",
            )

        qa_response = client.get("/api/qa/results")
        qa_results = qa_response.json()
        assert len(qa_results) > 0
        qa_id = qa_results[0]["id"]

        response = client.get(
            f"/api/qa/results/{qa_id}/pdf",
            params={"include_trend": True},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert len(response.content) > 0

    def test_pdf_export_without_trend(self, client, sample_plan_json, sample_log_csv):
        """Test PDF export without trend data."""
        self._submit_qa_for_report(
            client, sample_plan_json, sample_log_csv,
            fraction_number=1,
            patient_id="patient-pdf-notrend",
        )

        qa_response = client.get("/api/qa/results")
        qa_results = qa_response.json()
        qa_id = qa_results[0]["id"]

        response = client.get(
            f"/api/qa/results/{qa_id}/pdf",
            params={"include_trend": False},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"

    @staticmethod
    def _submit_qa_for_report(client, sample_plan_json, sample_log_csv,
                              fraction_number=1, patient_id="test"):
        """Helper to submit QA for report tests."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        return client.post(
            "/api/qa/submit",
            params={
                "patient_anonymous_id": patient_id,
                "beam_name": "AP Field",
                "fraction_number": fraction_number,
            },
            files={
                "plan_file": ("plan.json", plan_content, "application/json"),
                "log_file": ("log.csv", log_content, "text/csv"),
            },
        )
