"""Tests for FastAPI endpoints."""
import io
import json

import pytest

from mlc_qa.dicom_parser import create_simplified_plan_json
from mlc_qa.log_parser import create_sample_log_csv


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self, client):
        """Test health check endpoint returns 200."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data
        assert "timestamp" in data


class TestPlanUpload:
    """Test plan upload endpoints."""

    def test_upload_valid_plan(self, client, sample_plan_json):
        """Test uploading a valid plan file."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        files = {"file": ("plan.json", plan_content, "application/json")}

        response = client.post("/api/plan/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["plan_uid"] == "TEST-PLAN-001"
        assert data["num_beams"] == 1
        assert "AP Field" in data["beam_names"]

    def test_upload_invalid_json(self, client):
        """Test uploading invalid JSON returns 400."""
        files = {"file": ("plan.json", b"invalid json", "application/json")}
        response = client.post("/api/plan/upload", files=files)
        assert response.status_code == 400

    def test_upload_plan_missing_control_points(self, client):
        """Test uploading plan with missing control points returns 400."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN",
            beam_name="Bad",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"] = []

        content = json.dumps(bad_plan).encode("utf-8")
        files = {"file": ("plan.json", content, "application/json")}

        response = client.post("/api/plan/upload", files=files)
        assert response.status_code == 400

    def test_upload_plan_leaf_mismatch(self, client):
        """Test uploading plan with leaf count mismatch returns 400."""
        bad_plan = create_simplified_plan_json(
            plan_uid="BAD-PLAN",
            beam_name="Bad",
            num_leaves=60,
            num_control_points=10,
        )
        bad_plan["beams"][0]["control_points"][0]["leaf_positions_bank_b"] = (
            bad_plan["beams"][0]["control_points"][0]["leaf_positions_bank_b"][:-1]
        )

        content = json.dumps(bad_plan).encode("utf-8")
        files = {"file": ("plan.json", content, "application/json")}

        response = client.post("/api/plan/upload", files=files)
        assert response.status_code == 400


class TestLogUpload:
    """Test log upload endpoints."""

    def test_upload_valid_log(self, client, sample_log_csv):
        """Test uploading a valid log file."""
        log_content = sample_log_csv.encode("utf-8")
        files = {"file": ("log.csv", log_content, "text/csv")}

        response = client.post("/api/log/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["num_samples"] == 100
        assert data["num_leaves"] == 60

    def test_upload_log_with_num_leaves(self, client, sample_log_csv):
        """Test uploading log with explicit leaf count."""
        log_content = sample_log_csv.encode("utf-8")
        files = {"file": ("log.csv", log_content, "text/csv")}

        response = client.post(
            "/api/log/upload",
            files=files,
            params={"num_leaves": 60}
        )
        assert response.status_code == 200

    def test_upload_log_with_wrong_leaf_count(self, client, sample_log_csv):
        """Test uploading log with wrong expected leaf count returns 400."""
        log_content = sample_log_csv.encode("utf-8")
        files = {"file": ("log.csv", log_content, "text/csv")}

        response = client.post(
            "/api/log/upload",
            files=files,
            params={"num_leaves": 30}
        )
        assert response.status_code == 400

    def test_upload_malformed_csv(self, client):
        """Test uploading malformed CSV returns 400."""
        files = {"file": ("log.csv", b"bad,csv\n1,2,3", "text/csv")}
        response = client.post("/api/log/upload", files=files)
        assert response.status_code == 400

    def test_upload_log_missing_columns(self, client):
        """Test uploading log with missing columns returns 400."""
        csv_content = "bad_column,another\n1,2\n"
        files = {"file": ("log.csv", csv_content.encode("utf-8"), "text/csv")}
        response = client.post("/api/log/upload", files=files)
        assert response.status_code == 400


class TestQASubmit:
    """Test QA submission endpoint."""

    def test_submit_qa_success(self, client, sample_plan_json, sample_log_csv):
        """Test successful QA submission."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
            "notes": "Test QA submission",
        }

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["qa_result_id"] > 0
        assert "max_deviation_mm" in data
        assert "pass_rate_pct" in data
        assert "overall_pass" in data

    def test_submit_qa_wrong_beam_name(self, client, sample_plan_json, sample_log_csv):
        """Test QA submission with wrong beam name returns 404."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "Wrong Beam",
        }

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 404

    def test_submit_qa_leaf_count_mismatch(self, client, sample_plan_json):
        """Test QA submission with leaf count mismatch returns 400."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")

        log_csv = create_sample_log_csv(
            num_samples=100,
            num_leaves=30,
            noise_std=0.1,
        )
        log_content = log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 400

    def test_submit_qa_privacy(self, client, sample_plan_json, sample_log_csv):
        """Test that only anonymous ID is stored, no PHI."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "ANON-12345",
            "beam_name": "AP Field",
        }

        response = client.post("/api/qa/submit", files=files, params=params)
        assert response.status_code == 200

        patients_response = client.get("/api/patients")
        assert patients_response.status_code == 200
        patients = patients_response.json()

        assert len(patients) == 1
        assert patients[0]["anonymous_id"] == "ANON-12345"
        assert "name" not in patients[0]
        assert "mrn" not in patients[0]
        assert "dob" not in patients[0]


class TestQAResults:
    """Test QA results endpoints."""

    def test_list_qa_results_empty(self, client):
        """Test listing QA results when empty."""
        response = client.get("/api/qa/results")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_qa_results(self, client, sample_plan_json, sample_log_csv):
        """Test listing QA results after submission."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        submit_response = client.post("/api/qa/submit", files=files, params=params)
        qa_id = submit_response.json()["qa_result_id"]

        list_response = client.get("/api/qa/results")
        assert list_response.status_code == 200
        results = list_response.json()
        assert len(results) == 1
        assert results[0]["id"] == qa_id

    def test_get_qa_result_detail(self, client, sample_plan_json, sample_log_csv):
        """Test getting QA result detail."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        submit_response = client.post("/api/qa/submit", files=files, params=params)
        qa_id = submit_response.json()["qa_result_id"]

        detail_response = client.get(f"/api/qa/results/{qa_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["id"] == qa_id
        assert "max_leaf_deviation_mm" in detail
        assert "leaf_error_samples" in detail

    def test_get_nonexistent_qa_result(self, client):
        """Test getting non-existent QA result returns 404."""
        response = client.get("/api/qa/results/99999")
        assert response.status_code == 404

    def test_delete_qa_result(self, client, sample_plan_json, sample_log_csv):
        """Test deleting a QA result."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        submit_response = client.post("/api/qa/submit", files=files, params=params)
        qa_id = submit_response.json()["qa_result_id"]

        delete_response = client.delete(f"/api/qa/results/{qa_id}")
        assert delete_response.status_code == 200

        detail_response = client.get(f"/api/qa/results/{qa_id}")
        assert detail_response.status_code == 404

    def test_filter_qa_results_by_pass(self, client, sample_plan_json):
        """Test filtering QA results by pass/fail status."""
        good_log = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=0.0,
        )
        bad_log = create_sample_log_csv(
            num_samples=100,
            num_leaves=60,
            noise_std=10.0,
        )

        plan_content = json.dumps(sample_plan_json).encode("utf-8")

        for log_content in [good_log, bad_log]:
            files = {
                "plan_file": ("plan.json", plan_content, "application/json"),
                "log_file": ("log.csv", log_content.encode("utf-8"), "text/csv"),
            }
            params = {
                "patient_anonymous_id": "PATIENT-001",
                "beam_name": "AP Field",
            }
            client.post("/api/qa/submit", files=files, params=params)

        pass_response = client.get("/api/qa/results", params={"pass_filter": True})
        assert pass_response.status_code == 200
        pass_results = pass_response.json()
        assert len(pass_results) >= 1

        fail_response = client.get("/api/qa/results", params={"pass_filter": False})
        assert fail_response.status_code == 200
        fail_results = fail_response.json()
        assert len(fail_results) >= 1


class TestPDFExport:
    """Test PDF export endpoint."""

    def test_export_pdf(self, client, sample_plan_json, sample_log_csv):
        """Test exporting QA result as PDF."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        submit_response = client.post("/api/qa/submit", files=files, params=params)
        qa_id = submit_response.json()["qa_result_id"]

        pdf_response = client.get(f"/api/qa/results/{qa_id}/pdf")
        assert pdf_response.status_code == 200
        assert pdf_response.headers["content-type"] == "application/pdf"
        assert "attachment" in pdf_response.headers["content-disposition"]

        content = pdf_response.content
        assert content.startswith(b"%PDF")

    def test_export_nonexistent_pdf(self, client):
        """Test exporting non-existent QA result as PDF returns 404."""
        response = client.get("/api/qa/results/99999/pdf")
        assert response.status_code == 404


class TestPlanEndpoints:
    """Test plan management endpoints."""

    def test_list_plans_empty(self, client):
        """Test listing plans when empty."""
        response = client.get("/api/plans")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_plans(self, client, sample_plan_json, sample_log_csv):
        """Test listing plans after QA submission."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        client.post("/api/qa/submit", files=files, params=params)

        list_response = client.get("/api/plans")
        assert list_response.status_code == 200
        plans = list_response.json()
        assert len(plans) == 1
        assert plans[0]["plan_uid"] == "TEST-PLAN-001"

    def test_get_plan_detail(self, client, sample_plan_json, sample_log_csv):
        """Test getting plan detail."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        client.post("/api/qa/submit", files=files, params=params)

        plans = client.get("/api/plans").json()
        plan_id = plans[0]["id"]

        detail_response = client.get(f"/api/plans/{plan_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert "beams" in detail
        assert "qa_results" in detail

    def test_get_nonexistent_plan(self, client):
        """Test getting non-existent plan returns 404."""
        response = client.get("/api/plans/99999")
        assert response.status_code == 404


class TestPatientEndpoints:
    """Test patient alias endpoints."""

    def test_list_patients_empty(self, client):
        """Test listing patients when empty."""
        response = client.get("/api/patients")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_patients(self, client, sample_plan_json, sample_log_csv):
        """Test listing patients after QA submission."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        client.post("/api/qa/submit", files=files, params=params)

        list_response = client.get("/api/patients")
        assert list_response.status_code == 200
        patients = list_response.json()
        assert len(patients) == 1
        assert patients[0]["anonymous_id"] == "PATIENT-001"

    def test_get_patient_plans(self, client, sample_plan_json, sample_log_csv):
        """Test getting plans for a patient."""
        plan_content = json.dumps(sample_plan_json).encode("utf-8")
        log_content = sample_log_csv.encode("utf-8")

        files = {
            "plan_file": ("plan.json", plan_content, "application/json"),
            "log_file": ("log.csv", log_content, "text/csv"),
        }
        params = {
            "patient_anonymous_id": "PATIENT-001",
            "beam_name": "AP Field",
        }
        client.post("/api/qa/submit", files=files, params=params)

        plans_response = client.get("/api/patients/PATIENT-001/plans")
        assert plans_response.status_code == 200
        plans = plans_response.json()
        assert len(plans) == 1

    def test_get_nonexistent_patient_plans(self, client):
        """Test getting plans for non-existent patient returns 404."""
        response = client.get("/api/patients/NONEXISTENT/plans")
        assert response.status_code == 404
