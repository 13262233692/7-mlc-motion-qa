"""Tests for DICOM-RT Plan parser."""
import json
import tempfile
import os

import pytest
import numpy as np

from mlc_qa.dicom_parser import (
    DicomRTParser,
    DICOMParserError,
    MissingControlPointError,
    LeafCountMismatchError,
    create_simplified_plan_json,
)


class TestDicomParserBasic:
    """Basic DICOM parser tests."""

    def test_parse_simplified_json(self, sample_plan_json):
        """Test parsing simplified JSON plan format."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(sample_plan_json, f)
            temp_path = f.name

        try:
            plan = DicomRTParser.parse(temp_path)
            assert plan.plan_uid == "TEST-PLAN-001"
            assert plan.modality == "RTPLAN"
            assert len(plan.beams) == 1
            assert plan.beams[0].beam_name == "AP Field"
            assert plan.beams[0].num_leaves == 60
            assert plan.beams[0].num_control_points == 10
        finally:
            os.unlink(temp_path)

    def test_parse_string(self, sample_plan_json):
        """Test parsing from string content."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)
        assert plan.plan_uid == "TEST-PLAN-001"
        assert len(plan.beams) == 1

    def test_plan_to_dict_roundtrip(self, sample_plan_json):
        """Test plan to dict and back conversion."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)

        plan_dict = plan.to_dict()
        assert plan_dict["plan_uid"] == "TEST-PLAN-001"
        assert len(plan_dict["beams"]) == 1

        plan2 = DicomRTParser._build_plan_from_dict(plan_dict)
        assert plan2.plan_uid == plan.plan_uid
        assert plan2.num_beams == plan.num_beams

    def test_get_beam_by_name(self, sample_plan_json):
        """Test getting beam by name."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)
        beam = plan.get_beam_by_name("AP Field")
        assert beam is not None
        assert beam.beam_name == "AP Field"

        missing_beam = plan.get_beam_by_name("Non-existent")
        assert missing_beam is None

    def test_get_beam_by_number(self, sample_plan_json):
        """Test getting beam by number."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)
        beam = plan.get_beam_by_number(1)
        assert beam is not None
        assert beam.beam_number == 1

    def test_leaf_position_arrays(self, sample_plan_json):
        """Test leaf position array extraction."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)
        beam = plan.beams[0]

        bank_a = beam.get_leaf_positions_bank_a()
        bank_b = beam.get_leaf_positions_bank_b()

        assert bank_a.shape == (10, 60)
        assert bank_b.shape == (10, 60)
        assert np.all(bank_a <= 0)
        assert np.all(bank_b >= 0)

    def test_cumulative_weights(self, sample_plan_json):
        """Test cumulative weights extraction."""
        json_str = json.dumps(sample_plan_json)
        plan = DicomRTParser.parse_string(json_str)
        beam = plan.beams[0]

        weights = beam.get_cumulative_weights()
        assert len(weights) == 10
        assert weights[0] == 0.0
        assert np.isclose(weights[-1], 1.0)
        assert np.all(np.diff(weights) >= 0)


class TestMissingControlPoints:
    """Test case 1: Missing/incomplete control points."""

    def test_empty_control_points(self):
        """Test plan with no control points raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-001",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"] = []

        json_str = json.dumps(bad_plan)
        with pytest.raises(MissingControlPointError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "No control points" in str(exc_info.value)

    def test_single_control_point(self):
        """Test plan with only 1 control point raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-002",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"] = bad_plan["beams"][0]["control_points"][:1]

        json_str = json.dumps(bad_plan)
        with pytest.raises(MissingControlPointError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "at least 2 control points" in str(exc_info.value)

    def test_first_cp_weight_not_zero(self):
        """Test first control point weight not zero raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-003",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][0]["cumulative_meterset_weight"] = 0.1

        json_str = json.dumps(bad_plan)
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "First control point must have cumulative weight 0.0" in str(exc_info.value)

    def test_last_cp_weight_not_one(self):
        """Test last control point weight not ~1.0 raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-004",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][-1]["cumulative_meterset_weight"] = 0.5

        json_str = json.dumps(bad_plan)
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "Last control point must have cumulative weight ~1.0" in str(exc_info.value)

    def test_non_monotonic_weights(self):
        """Test non-monotonic weights raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-005",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][5]["cumulative_meterset_weight"] = 0.3
        bad_plan["beams"][0]["control_points"][6]["cumulative_meterset_weight"] = 0.2

        json_str = json.dumps(bad_plan)
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "non-decreasing" in str(exc_info.value)

    def test_missing_required_field(self):
        """Test missing required field raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-006",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        del bad_plan["beams"][0]["control_points"][0]["cumulative_meterset_weight"]

        json_str = json.dumps(bad_plan)
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "Missing required control point field" in str(exc_info.value)


class TestLeafCountMismatch:
    """Test case 2: Inconsistent leaf count across control points."""

    def test_bank_ab_mismatch_within_cp(self):
        """Test bank A and B have different leaf counts in same CP."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-010",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][0]["leaf_positions_bank_b"] = (
            bad_plan["beams"][0]["control_points"][0]["leaf_positions_bank_b"][:-1]
        )

        json_str = json.dumps(bad_plan)
        with pytest.raises(LeafCountMismatchError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "Bank A and B leaf count mismatch" in str(exc_info.value)

    def test_leaf_count_mismatch_across_cp(self):
        """Test different leaf counts across control points."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-011",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][5]["leaf_positions_bank_a"] = (
            bad_plan["beams"][0]["control_points"][5]["leaf_positions_bank_a"][:-5]
        )
        bad_plan["beams"][0]["control_points"][5]["leaf_positions_bank_b"] = (
            bad_plan["beams"][0]["control_points"][5]["leaf_positions_bank_b"][:-5]
        )

        json_str = json.dumps(bad_plan)
        with pytest.raises(LeafCountMismatchError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "Leaf count mismatch at control point 5" in str(exc_info.value)

    def test_zero_leaves(self):
        """Test plan with zero leaves raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-012",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        for cp in bad_plan["beams"][0]["control_points"]:
            cp["leaf_positions_bank_a"] = []
            cp["leaf_positions_bank_b"] = []

        json_str = json.dumps(bad_plan)
        with pytest.raises(LeafCountMismatchError) as exc_info:
            DicomRTParser.parse_string(json_str)

    def test_missing_required_beam_field(self):
        """Test missing required beam field raises error."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN-013",
            beam_name="Bad Beam",
            num_leaves=60,
            num_control_points=10,
        )
        del bad_plan["beams"][0]["beam_number"]

        json_str = json.dumps(bad_plan)
        with pytest.raises(DICOMParserError) as exc_info:
            DicomRTParser.parse_string(json_str)
        assert "Missing required beam field" in str(exc_info.value)


class TestCreateSimplifiedPlan:
    """Test plan creation helper function."""

    def test_create_plan_default(self):
        """Test creating a plan with default parameters."""
        plan = create_simplified_plan_json(
            plan_uid="TEST-001",
            beam_name="Test Beam",
        )
        assert plan["plan_uid"] == "TEST-001"
        assert plan["beams"][0]["beam_name"] == "Test Beam"
        assert len(plan["beams"][0]["control_points"]) == 10
        assert len(plan["beams"][0]["control_points"][0]["leaf_positions_bank_a"]) == 60

    def test_create_plan_custom(self):
        """Test creating a plan with custom parameters."""
        plan = create_simplified_plan_json(
            plan_uid="TEST-002",
            beam_name="Custom Beam",
            num_leaves=40,
            num_control_points=20,
            max_leaf_position=150.0,
        )
        assert len(plan["beams"][0]["control_points"]) == 20
        assert len(plan["beams"][0]["control_points"][0]["leaf_positions_bank_a"]) == 40
        assert plan["beams"][0]["control_points"][-1]["leaf_positions_bank_b"][0] == 150.0
