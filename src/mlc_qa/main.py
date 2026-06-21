"""
FastAPI REST API for MLC Motion QA System.

Endpoints:
- POST   /api/plan/upload          - Upload DICOM-RT plan file
- POST   /api/log/upload           - Upload treatment log CSV
- POST   /api/qa/submit            - Submit QA analysis
- GET    /api/qa/results           - List QA results
- GET    /api/qa/results/{id}      - Get QA result detail
- GET    /api/qa/results/{id}/pdf  - Export QA result as PDF
- GET    /api/plans                - List plans
- GET    /api/plans/{id}           - Get plan detail
- GET    /api/patients             - List patient aliases
- GET    /api/health               - Health check
"""
import io
import json
from typing import Optional, List
from datetime import datetime

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from mlc_qa.database import get_db, init_db
from mlc_qa import schemas, crud
from mlc_qa.dicom_parser import (
    DicomRTParser,
    PlanData,
    DICOMParserError,
    MissingControlPointError,
    LeafCountMismatchError,
)
from mlc_qa.log_parser import (
    TreatmentLogParser,
    TreatmentLog,
    LogParserError,
    MissingDataError,
)
from mlc_qa.calculations import (
    MLCQACalculator,
    CalculationError,
    QAAnalysisResult,
    TrendAnalyzer,
    FractionMetrics,
)
from mlc_qa.report_generator import generate_qa_report_pdf

app = FastAPI(
    title="MLC Motion QA API",
    description="Multi-Leaf Collimator motion log verification system for radiation therapy physics",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database on startup."""
    init_db()


@app.get("/api/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/api/plan/upload", response_model=schemas.PlanParseResponse, tags=["Plan"])
async def upload_plan(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """
    Upload and parse a DICOM-RT Plan file (or simplified JSON format).

    The file is parsed but not stored in the database until QA submission.
    Only anonymous patient ID is stored.
    """
    try:
        content = await file.read()
        plan_data = None

        try:
            content_str = content.decode("utf-8")
            plan_data = DicomRTParser.parse_string(content_str)
        except (UnicodeDecodeError, DICOMParserError, json.JSONDecodeError):
            import tempfile
            import os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                plan_data = DicomRTParser.parse(tmp_path)
            except (DICOMParserError, FileNotFoundError):
                raise HTTPException(
                    status_code=400,
                    detail="Unable to parse file as DICOM or JSON plan format"
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        return schemas.PlanParseResponse(
            success=True,
            plan_uid=plan_data.plan_uid,
            modality=plan_data.modality,
            num_beams=len(plan_data.beams),
            beam_names=[beam.beam_name for beam in plan_data.beams],
            message=f"Successfully parsed plan with {len(plan_data.beams)} beam(s)",
        )

    except HTTPException:
        raise
    except MissingControlPointError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {str(e)}")
    except LeafCountMismatchError as e:
        raise HTTPException(status_code=400, detail=f"Leaf count error: {str(e)}")
    except DICOMParserError as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/log/upload", response_model=schemas.LogParseResponse, tags=["Log"])
async def upload_log(
    file: UploadFile = File(...),
    num_leaves: Optional[int] = Query(None, description="Expected number of leaves per bank"),
):
    """
    Upload and parse a treatment log CSV file.

    The file is parsed but not stored until QA submission.
    """
    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        log = TreatmentLogParser.parse_string(
            content_str,
            filename=file.filename or "log.csv",
            num_leaves=num_leaves,
        )

        return schemas.LogParseResponse(
            success=True,
            num_samples=log.num_samples,
            time_range_sec=log.duration_sec,
            num_leaves=log.num_leaves,
            message=(
                f"Successfully parsed {log.num_samples} samples, "
                f"{log.num_leaves} leaves, duration {log.duration_sec:.2f}s"
            ),
        )

    except MissingDataError as e:
        raise HTTPException(status_code=400, detail=f"Missing data: {str(e)}")
    except LogParserError as e:
        raise HTTPException(status_code=400, detail=f"Parse error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/api/qa/submit", response_model=schemas.QASubmitResponse, tags=["QA"])
async def submit_qa(
    patient_anonymous_id: str = Query(..., description="Anonymous patient identifier"),
    plan_file: UploadFile = File(..., description="DICOM-RT Plan file (JSON or DICOM)"),
    log_file: UploadFile = File(..., description="Treatment log CSV file"),
    beam_name: str = Query(..., description="Beam name to analyze"),
    notes: Optional[str] = Query(None, description="Optional notes"),
    fraction_number: int = Query(0, description="Fraction number (treatment session number)"),
    plan_version: int = Query(1, description="Plan version number for re-planning"),
    db: Session = Depends(get_db),
):
    """
    Submit a complete QA analysis.

    Uploads plan and log, parses them, performs QA analysis, and stores results in database.
    Only the anonymous patient ID is stored - no PHI is saved.
    """
    try:
        plan_content = await plan_file.read()
        log_content = await log_file.read()
        plan_data = None

        try:
            plan_str = plan_content.decode("utf-8")
            plan_data = DicomRTParser.parse_string(plan_str)
        except (UnicodeDecodeError, DICOMParserError, json.JSONDecodeError):
            import tempfile
            import os
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                    tmp.write(plan_content)
                    tmp_path = tmp.name
                plan_data = DicomRTParser.parse(tmp_path)
            except (DICOMParserError, FileNotFoundError):
                raise HTTPException(
                    status_code=400,
                    detail="Unable to parse plan file as DICOM or JSON plan format"
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

        log_str = log_content.decode("utf-8")
        log_data = TreatmentLogParser.parse_string(
            log_str,
            filename=log_file.filename or "log.csv",
        )

        beam_data = plan_data.get_beam_by_name(beam_name)
        if beam_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Beam '{beam_name}' not found in plan. "
                       f"Available beams: {[b.beam_name for b in plan_data.beams]}"
            )

        if beam_data.num_leaves != log_data.num_leaves:
            raise HTTPException(
                status_code=400,
                detail=f"Leaf count mismatch: plan has {beam_data.num_leaves}, "
                       f"log has {log_data.num_leaves}"
            )

        patient = crud.patient_alias.get_or_create(db, patient_anonymous_id)

        existing_plan = crud.plan.get_by_plan_uid(db, plan_data.plan_uid)
        if existing_plan:
            plan_db = existing_plan
        else:
            plan_db = crud.plan.create_from_plan_data(db, patient.id, plan_data)

        beam_db = crud.beam.get_by_plan_and_name(db, plan_db.id, beam_name)
        if beam_db is None:
            raise HTTPException(
                status_code=500,
                detail=f"Beam '{beam_name}' not found in database after plan creation"
            )

        calculator = MLCQACalculator()
        analysis_result = calculator.analyze(beam_data, log_data)

        qa_result_db = crud.qa_result.create_from_analysis(
            db,
            plan_id=plan_db.id,
            beam_id=beam_db.id,
            analysis_result=analysis_result,
            log_filename=log_file.filename,
            notes=notes,
            fraction_number=fraction_number,
            plan_version=plan_version,
        )

        fraction_summary = crud.fraction_qa_summary.update_from_qa_result(
            db, qa_result_db
        )

        return schemas.QASubmitResponse(
            success=True,
            qa_result_id=qa_result_db.id,
            fraction_summary_id=fraction_summary.id,
            message=(
                f"QA analysis completed. "
                f"Pass rate: {analysis_result.control_point_pass_rate_pct:.2f}%"
            ),
            max_deviation_mm=analysis_result.max_leaf_deviation_mm,
            pass_rate_pct=analysis_result.control_point_pass_rate_pct,
            overall_pass=analysis_result.overall_pass,
            fraction_number=fraction_number,
            plan_version=plan_version,
        )

    except MissingControlPointError as e:
        raise HTTPException(status_code=400, detail=f"Invalid plan: {str(e)}")
    except LeafCountMismatchError as e:
        raise HTTPException(status_code=400, detail=f"Leaf count error: {str(e)}")
    except DICOMParserError as e:
        raise HTTPException(status_code=400, detail=f"Plan parse error: {str(e)}")
    except LogParserError as e:
        raise HTTPException(status_code=400, detail=f"Log parse error: {str(e)}")
    except MissingDataError as e:
        raise HTTPException(status_code=400, detail=f"Missing data: {str(e)}")
    except CalculationError as e:
        raise HTTPException(status_code=400, detail=f"Calculation error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/api/qa/results", response_model=List[schemas.QAResultResponse], tags=["QA"])
async def list_qa_results(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    pass_filter: Optional[bool] = Query(None, description="Filter by pass/fail status"),
    db: Session = Depends(get_db),
):
    """List QA results with pagination and optional filtering."""
    results = crud.qa_result.list(db, skip=skip, limit=limit, pass_filter=pass_filter)
    return results


@app.get("/api/qa/results/{qa_result_id}", response_model=schemas.QAResultDetail, tags=["QA"])
async def get_qa_result(
    qa_result_id: int,
    db: Session = Depends(get_db),
):
    """Get detailed QA result including leaf error samples."""
    result = crud.qa_result.get(db, qa_result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="QA result not found")

    samples = crud.leaf_error_sample.get_by_qa_result_id(db, qa_result_id)
    result.leaf_error_samples = samples
    return result


@app.get("/api/qa/results/{qa_result_id}/pdf", tags=["QA"])
async def export_qa_pdf(
    qa_result_id: int,
    include_trend: bool = Query(True, description="Include fraction trend chart in report"),
    db: Session = Depends(get_db),
):
    """Export QA result as PDF report."""
    result = crud.qa_result.get(db, qa_result_id)
    if result is None:
        raise HTTPException(status_code=404, detail="QA result not found")

    samples = crud.leaf_error_sample.get_by_qa_result_id(db, qa_result_id)

    fraction_summaries = None
    if include_trend:
        fraction_summaries = crud.fraction_qa_summary.list_by_plan_beam(
            db, result.plan_id, result.beam_id, plan_version=result.plan_version
        )

    try:
        pdf_content = generate_qa_report_pdf(result, samples, fraction_summaries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")

    filename = f"mlc_qa_report_{qa_result_id}.pdf"

    return StreamingResponse(
        io.BytesIO(pdf_content),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}",
            "Content-Length": str(len(pdf_content)),
        },
    )


@app.delete("/api/qa/results/{qa_result_id}", tags=["QA"])
async def delete_qa_result(
    qa_result_id: int,
    db: Session = Depends(get_db),
):
    """Delete a QA result and its associated leaf error samples."""
    deleted = crud.qa_result.delete(db, qa_result_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="QA result not found")
    return {"success": True, "message": f"QA result {qa_result_id} deleted"}


@app.get("/api/plans", response_model=List[schemas.PlanResponse], tags=["Plan"])
async def list_plans(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """List all plans."""
    return crud.plan.list(db, skip=skip, limit=limit)


@app.get("/api/plans/{plan_id}", response_model=schemas.PlanWithDetail, tags=["Plan"])
async def get_plan(
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Get plan details including beams and QA results."""
    plan = crud.plan.get(db, plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    return plan


@app.delete("/api/plans/{plan_id}", tags=["Plan"])
async def delete_plan(
    plan_id: int,
    db: Session = Depends(get_db),
):
    """Delete a plan and all associated beams and QA results."""
    deleted = crud.plan.delete(db, plan_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Plan not found")
    return {"success": True, "message": f"Plan {plan_id} deleted"}


@app.get("/api/patients", response_model=List[schemas.PatientAliasResponse], tags=["Patient"])
async def list_patients(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """List all patient aliases (anonymous IDs only - no PHI)."""
    return crud.patient_alias.list(db, skip=skip, limit=limit)


@app.get("/api/patients/{anonymous_id}/plans", response_model=List[schemas.PlanResponse], tags=["Patient"])
async def get_patient_plans(
    anonymous_id: str,
    db: Session = Depends(get_db),
):
    """Get all plans for a patient (by anonymous ID)."""
    patient = crud.patient_alias.get_by_anonymous_id(db, anonymous_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    return crud.plan.get_by_patient_id(db, patient.id)


@app.get("/api/plans/{plan_id}/beams/{beam_id}/fractions",
         response_model=List[schemas.FractionQASummaryResponse],
         tags=["Fraction"])
async def list_fraction_summaries(
    plan_id: int,
    beam_id: int,
    plan_version: Optional[int] = Query(None, description="Filter by plan version"),
    db: Session = Depends(get_db),
):
    """List all fraction QA summaries for a plan+beam."""
    summaries = crud.fraction_qa_summary.list_by_plan_beam(
        db, plan_id, beam_id, plan_version=plan_version
    )
    return summaries


@app.get("/api/plans/{plan_id}/beams/{beam_id}/fractions/{fraction_number}",
         response_model=schemas.FractionQASummaryResponse,
         tags=["Fraction"])
async def get_fraction_summary(
    plan_id: int,
    beam_id: int,
    fraction_number: int,
    plan_version: int = Query(1, description="Plan version number"),
    db: Session = Depends(get_db),
):
    """Get a specific fraction QA summary."""
    summary = crud.fraction_qa_summary.get_by_fraction(
        db, plan_id, beam_id, fraction_number, plan_version
    )
    if summary is None:
        raise HTTPException(status_code=404, detail="Fraction summary not found")
    return summary


@app.get("/api/plans/{plan_id}/beams/{beam_id}/trend",
         response_model=schemas.TrendAnalysisResult,
         tags=["Fraction"])
async def get_fraction_trend(
    plan_id: int,
    beam_id: int,
    plan_version: Optional[int] = Query(None, description="Filter by plan version (None for all)"),
    db: Session = Depends(get_db),
):
    """
    Perform trend analysis on fraction QA results.

    Analyzes leaf deviation and pass rate trends across fractions and returns
    trend labels (STABLE_NORMAL, GRADUAL_INCREASE, SHARP_INCREASE,
    SINGLE_SPIKE, IMPROVING, ERRATIC, INSUFFICIENT_DATA, ALL_PASS, ALL_FAIL),
    chart data, anomaly flags, and per-fraction details.
    """
    summaries = crud.fraction_qa_summary.list_by_plan_beam(
        db, plan_id, beam_id, plan_version=plan_version
    )

    if not summaries:
        raise HTTPException(status_code=404, detail="No fraction data found for trend analysis")

    fraction_metrics = []
    total_qa_results = 0
    for s in summaries:
        if (s.max_leaf_deviation_mm is None and s.rmse_mm is None
                and s.control_point_pass_rate_pct is None):
            continue
        fm = FractionMetrics(
            fraction_number=s.fraction_number,
            plan_version=s.plan_version,
            max_leaf_deviation_mm=s.max_leaf_deviation_mm or 0.0,
            mean_leaf_deviation_mm=s.mean_leaf_deviation_mm or 0.0,
            rmse_mm=s.rmse_mm or 0.0,
            pass_rate_pct=s.control_point_pass_rate_pct or 0.0,
            overall_pass=(s.overall_pass_rate_pct or 0.0) >= 95.0,
            qa_date=s.qa_date,
        )
        fraction_metrics.append(fm)
        total_qa_results += s.num_qa_results

    if len(fraction_metrics) < 2:
        return schemas.TrendAnalysisResult(
            plan_id=plan_id,
            beam_id=beam_id,
            total_fractions=len(fraction_metrics),
            total_qa_results=total_qa_results,
            plan_versions=sorted({s.plan_version for s in summaries}),
            trend_label="INSUFFICIENT_DATA",
            trend_confidence=0.0,
            overall_trend_description="Insufficient data for trend analysis (need at least 2 fractions)",
            latest_fraction=fraction_metrics[-1].fraction_number if fraction_metrics else 0,
            latest_max_deviation_mm=(fraction_metrics[-1].max_leaf_deviation_mm
                                      if fraction_metrics else None),
            latest_pass_rate_pct=(fraction_metrics[-1].pass_rate_pct
                                   if fraction_metrics else None),
            fractions=[schemas.FractionQASummaryResponse.model_validate(s) for s in summaries],
            chart_data={},
            anomaly_flags=[],
        )

    analyzer = TrendAnalyzer()
    result = analyzer.analyze(fraction_metrics)

    return schemas.TrendAnalysisResult(
        plan_id=plan_id,
        beam_id=beam_id,
        total_fractions=result.total_fractions,
        total_qa_results=total_qa_results,
        plan_versions=result.plan_versions,
        trend_label=result.trend_label.value,
        trend_confidence=result.trend_confidence,
        overall_trend_description=result.overall_description,
        max_deviation_trend_slope_mm_per_fraction=result.max_deviation_slope,
        pass_rate_trend_slope_pct_per_fraction=result.pass_rate_slope,
        latest_fraction=result.latest_fraction,
        latest_max_deviation_mm=result.latest_max_deviation,
        latest_pass_rate_pct=result.latest_pass_rate,
        fractions=[schemas.FractionQASummaryResponse.model_validate(s) for s in summaries],
        chart_data=result.chart_data,
        anomaly_flags=result.anomaly_flags,
    )


@app.get("/api/plans/{plan_id}/fractions",
         response_model=List[schemas.FractionQASummaryResponse],
         tags=["Fraction"])
async def list_plan_fractions(
    plan_id: int,
    db: Session = Depends(get_db),
):
    """List all fraction QA summaries for a plan (all beams and versions)."""
    summaries = crud.fraction_qa_summary.list_by_plan(db, plan_id)
    return summaries


if __name__ == "__main__":
    import uvicorn
    from mlc_qa.config import APP_HOST, APP_PORT

    uvicorn.run(
        "mlc_qa.main:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True,
    )
