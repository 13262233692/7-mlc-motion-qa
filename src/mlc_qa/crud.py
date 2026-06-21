"""CRUD database operations layer."""
from typing import Optional, List, Dict, Any
from datetime import datetime

import numpy as np
from sqlalchemy.orm import Session

from mlc_qa import models, schemas
from mlc_qa.dicom_parser import PlanData, BeamData
from mlc_qa.log_parser import TreatmentLog
from mlc_qa.calculations import QAAnalysisResult


class CRUDPatientAlias:
    """CRUD operations for PatientAlias."""

    @staticmethod
    def create(db: Session, obj_in: schemas.PatientAliasCreate) -> models.PatientAlias:
        """Create a new patient alias."""
        db_obj = models.PatientAlias(
            anonymous_id=obj_in.anonymous_id
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get(db: Session, patient_id: int) -> Optional[models.PatientAlias]:
        """Get patient alias by ID."""
        return db.query(models.PatientAlias).filter(
            models.PatientAlias.id == patient_id
        ).first()

    @staticmethod
    def get_by_anonymous_id(
        db: Session, anonymous_id: str
    ) -> Optional[models.PatientAlias]:
        """Get patient alias by anonymous ID."""
        return db.query(models.PatientAlias).filter(
            models.PatientAlias.anonymous_id == anonymous_id
        ).first()

    @staticmethod
    def get_or_create(
        db: Session, anonymous_id: str
    ) -> models.PatientAlias:
        """Get patient alias or create if it doesn't exist."""
        obj = CRUDPatientAlias.get_by_anonymous_id(db, anonymous_id)
        if obj:
            return obj
        return CRUDPatientAlias.create(
            db, schemas.PatientAliasCreate(anonymous_id=anonymous_id)
        )

    @staticmethod
    def list(db: Session, skip: int = 0, limit: int = 100) -> List[models.PatientAlias]:
        """List all patient aliases."""
        return db.query(models.PatientAlias).offset(skip).limit(limit).all()


class CRUDPlan:
    """CRUD operations for Plan."""

    @staticmethod
    def create(
        db: Session,
        obj_in: schemas.PlanCreate,
    ) -> models.Plan:
        """Create a new plan."""
        db_obj = models.Plan(
            patient_id=obj_in.patient_id,
            plan_uid=obj_in.plan_uid,
            plan_name=obj_in.plan_name,
            modality=obj_in.modality,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def create_from_plan_data(
        db: Session,
        patient_id: int,
        plan_data: PlanData,
    ) -> models.Plan:
        """Create plan and associated beams from parsed PlanData."""
        plan = CRUDPlan.create(
            db,
            schemas.PlanCreate(
                patient_id=patient_id,
                plan_uid=plan_data.plan_uid,
                plan_name=plan_data.plan_name,
                modality=plan_data.modality,
            ),
        )

        for beam_data in plan_data.beams:
            CRUDBeam.create_from_beam_data(db, plan.id, beam_data)

        return plan

    @staticmethod
    def get(db: Session, plan_id: int) -> Optional[models.Plan]:
        """Get plan by ID."""
        return db.query(models.Plan).filter(models.Plan.id == plan_id).first()

    @staticmethod
    def get_by_plan_uid(db: Session, plan_uid: str) -> Optional[models.Plan]:
        """Get plan by UID."""
        return db.query(models.Plan).filter(models.Plan.plan_uid == plan_uid).first()

    @staticmethod
    def get_by_patient_id(
        db: Session, patient_id: int
    ) -> List[models.Plan]:
        """Get all plans for a patient."""
        return db.query(models.Plan).filter(
            models.Plan.patient_id == patient_id
        ).all()

    @staticmethod
    def list(db: Session, skip: int = 0, limit: int = 100) -> List[models.Plan]:
        """List all plans."""
        return db.query(models.Plan).offset(skip).limit(limit).all()

    @staticmethod
    def delete(db: Session, plan_id: int) -> bool:
        """Delete a plan and its associated beams and QA results."""
        plan = CRUDPlan.get(db, plan_id)
        if plan:
            db.delete(plan)
            db.commit()
            return True
        return False


class CRUDBeam:
    """CRUD operations for Beam."""

    @staticmethod
    def create(
        db: Session,
        plan_id: int,
        beam_name: str,
        beam_number: int,
        beam_type: str,
        energy: str,
        control_points_data: List[Dict[str, Any]],
        leaf_positions: Dict[str, Any],
        dose_rates: Optional[List[float]] = None,
        gantry_angles: Optional[List[float]] = None,
    ) -> models.Beam:
        """Create a new beam."""
        db_obj = models.Beam(
            plan_id=plan_id,
            beam_name=beam_name,
            beam_number=beam_number,
            beam_type=beam_type,
            energy=energy,
            control_points_data=control_points_data,
            leaf_positions=leaf_positions,
            dose_rates=dose_rates,
            gantry_angles=gantry_angles,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def create_from_beam_data(
        db: Session,
        plan_id: int,
        beam_data: BeamData,
    ) -> models.Beam:
        """Create beam from parsed BeamData."""
        control_points_data = [
            {
                "index": cp.index,
                "cumulative_meterset_weight": float(cp.cumulative_meterset_weight),
                "dose_rate": float(cp.dose_rate),
                "gantry_angle": float(cp.gantry_angle),
            }
            for cp in beam_data.control_points
        ]

        leaf_positions = {
            "bank_a": beam_data.get_leaf_positions_bank_a().tolist(),
            "bank_b": beam_data.get_leaf_positions_bank_b().tolist(),
            "num_leaves": beam_data.num_leaves,
        }

        return CRUDBeam.create(
            db,
            plan_id=plan_id,
            beam_name=beam_data.beam_name,
            beam_number=beam_data.beam_number,
            beam_type=beam_data.beam_type,
            energy=beam_data.energy,
            control_points_data=control_points_data,
            leaf_positions=leaf_positions,
            dose_rates=beam_data.get_dose_rates().tolist(),
            gantry_angles=beam_data.get_gantry_angles().tolist(),
        )

    @staticmethod
    def get(db: Session, beam_id: int) -> Optional[models.Beam]:
        """Get beam by ID."""
        return db.query(models.Beam).filter(models.Beam.id == beam_id).first()

    @staticmethod
    def get_by_plan_and_name(
        db: Session, plan_id: int, beam_name: str
    ) -> Optional[models.Beam]:
        """Get beam by plan ID and beam name."""
        return db.query(models.Beam).filter(
            models.Beam.plan_id == plan_id,
            models.Beam.beam_name == beam_name,
        ).first()

    @staticmethod
    def get_by_plan_id(db: Session, plan_id: int) -> List[models.Beam]:
        """Get all beams for a plan."""
        return db.query(models.Beam).filter(models.Beam.plan_id == plan_id).all()


class CRUDQAResult:
    """CRUD operations for QAResult."""

    @staticmethod
    def create(
        db: Session,
        obj_in: schemas.QAResultCreate,
    ) -> models.QAResult:
        """Create a new QA result."""
        db_obj = models.QAResult(
            plan_id=obj_in.plan_id,
            beam_id=obj_in.beam_id,
            fraction_number=obj_in.fraction_number,
            plan_version=obj_in.plan_version,
            log_filename=obj_in.log_filename,
            max_leaf_deviation_mm=obj_in.max_leaf_deviation_mm,
            mean_leaf_deviation_mm=obj_in.mean_leaf_deviation_mm,
            rmse_mm=obj_in.rmse_mm,
            dose_rate_deviation_pct=obj_in.dose_rate_deviation_pct,
            control_point_pass_rate_pct=obj_in.control_point_pass_rate_pct,
            num_control_points=obj_in.num_control_points,
            num_failed_control_points=obj_in.num_failed_control_points,
            num_leaves=obj_in.num_leaves,
            gantry_angle_range=obj_in.gantry_angle_range,
            overall_pass=obj_in.overall_pass,
            notes=obj_in.notes,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def create_from_analysis(
        db: Session,
        plan_id: int,
        beam_id: int,
        analysis_result: QAAnalysisResult,
        log_filename: Optional[str] = None,
        notes: Optional[str] = None,
        fraction_number: int = 0,
        plan_version: int = 1,
    ) -> models.QAResult:
        """Create QA result from analysis result."""
        gantry_range = (
            f"{analysis_result.gantry_angle_start:.1f}° - "
            f"{analysis_result.gantry_angle_end:.1f}°"
        )

        qa_result = CRUDQAResult.create(
            db,
            schemas.QAResultCreate(
                plan_id=plan_id,
                beam_id=beam_id,
                log_filename=log_filename,
                max_leaf_deviation_mm=analysis_result.max_leaf_deviation_mm,
                mean_leaf_deviation_mm=analysis_result.mean_leaf_deviation_mm,
                rmse_mm=analysis_result.rmse_mm,
                dose_rate_deviation_pct=analysis_result.dose_rate_deviation_pct,
                control_point_pass_rate_pct=analysis_result.control_point_pass_rate_pct,
                num_control_points=analysis_result.num_control_points,
                num_failed_control_points=analysis_result.num_failed_control_points,
                num_leaves=analysis_result.num_leaves,
                gantry_angle_range=gantry_range,
                overall_pass=1 if analysis_result.overall_pass else 0,
                notes=notes,
                fraction_number=fraction_number,
                plan_version=plan_version,
            ),
        )

        for leaf_dev in analysis_result.leaf_deviations:
            CRUDLeafErrorSample.create(
                db,
                qa_result_id=qa_result.id,
                control_point_index=leaf_dev.control_point_index,
                leaf_index=leaf_dev.leaf_index,
                bank=leaf_dev.bank,
                planned_position_mm=leaf_dev.planned_position_mm,
                actual_position_mm=leaf_dev.actual_position_mm,
                deviation_mm=leaf_dev.deviation_mm,
                timestamp_sec=leaf_dev.timestamp_sec,
            )

        return qa_result

    @staticmethod
    def get(db: Session, qa_result_id: int) -> Optional[models.QAResult]:
        """Get QA result by ID."""
        return db.query(models.QAResult).filter(
            models.QAResult.id == qa_result_id
        ).first()

    @staticmethod
    def get_by_plan_id(
        db: Session, plan_id: int
    ) -> List[models.QAResult]:
        """Get all QA results for a plan."""
        return db.query(models.QAResult).filter(
            models.QAResult.plan_id == plan_id
        ).order_by(models.QAResult.qa_date.desc()).all()

    @staticmethod
    def get_by_beam_id(
        db: Session, beam_id: int
    ) -> List[models.QAResult]:
        """Get all QA results for a beam."""
        return db.query(models.QAResult).filter(
            models.QAResult.beam_id == beam_id
        ).order_by(models.QAResult.qa_date.desc()).all()

    @staticmethod
    def list(
        db: Session,
        skip: int = 0,
        limit: int = 100,
        pass_filter: Optional[bool] = None,
    ) -> List[models.QAResult]:
        """List QA results with optional filtering."""
        query = db.query(models.QAResult)
        if pass_filter is not None:
            query = query.filter(models.QAResult.overall_pass == (1 if pass_filter else 0))
        return query.order_by(models.QAResult.qa_date.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def delete(db: Session, qa_result_id: int) -> bool:
        """Delete a QA result and its associated leaf error samples."""
        qa_result = CRUDQAResult.get(db, qa_result_id)
        if qa_result:
            db.delete(qa_result)
            db.commit()
            return True
        return False


class CRUDLeafErrorSample:
    """CRUD operations for LeafErrorSample."""

    @staticmethod
    def create(
        db: Session,
        qa_result_id: int,
        control_point_index: int,
        leaf_index: int,
        bank: str,
        planned_position_mm: float,
        actual_position_mm: float,
        deviation_mm: float,
        timestamp_sec: Optional[float] = None,
    ) -> models.LeafErrorSample:
        """Create a new leaf error sample."""
        db_obj = models.LeafErrorSample(
            qa_result_id=qa_result_id,
            control_point_index=control_point_index,
            leaf_index=leaf_index,
            bank=bank,
            planned_position_mm=planned_position_mm,
            actual_position_mm=actual_position_mm,
            deviation_mm=deviation_mm,
            timestamp_sec=timestamp_sec,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get(db: Session, sample_id: int) -> Optional[models.LeafErrorSample]:
        """Get leaf error sample by ID."""
        return db.query(models.LeafErrorSample).filter(
            models.LeafErrorSample.id == sample_id
        ).first()

    @staticmethod
    def get_by_qa_result_id(
        db: Session, qa_result_id: int, limit: int = 1000
    ) -> List[models.LeafErrorSample]:
        """Get all leaf error samples for a QA result."""
        return db.query(models.LeafErrorSample).filter(
            models.LeafErrorSample.qa_result_id == qa_result_id
        ).order_by(
            models.LeafErrorSample.deviation_mm.desc()
        ).limit(limit).all()


class CRUDFractionQASummary:
    """CRUD operations for FractionQASummary."""

    @staticmethod
    def create(
        db: Session,
        obj_in: schemas.FractionQASummaryCreate,
    ) -> models.FractionQASummary:
        """Create a new fraction QA summary."""
        db_obj = models.FractionQASummary(
            plan_id=obj_in.plan_id,
            beam_id=obj_in.beam_id,
            fraction_number=obj_in.fraction_number,
            plan_version=obj_in.plan_version,
            num_qa_results=obj_in.num_qa_results,
            latest_qa_result_id=obj_in.latest_qa_result_id,
            max_leaf_deviation_mm=obj_in.max_leaf_deviation_mm,
            mean_leaf_deviation_mm=obj_in.mean_leaf_deviation_mm,
            rmse_mm=obj_in.rmse_mm,
            dose_rate_deviation_pct=obj_in.dose_rate_deviation_pct,
            control_point_pass_rate_pct=obj_in.control_point_pass_rate_pct,
            overall_pass_rate_pct=obj_in.overall_pass_rate_pct,
            trend_label=obj_in.trend_label,
            trend_confidence=obj_in.trend_confidence,
            deviation_delta_from_previous_mm=obj_in.deviation_delta_from_previous_mm,
        )
        db.add(db_obj)
        db.commit()
        db.refresh(db_obj)
        return db_obj

    @staticmethod
    def get(db: Session, summary_id: int) -> Optional[models.FractionQASummary]:
        """Get fraction QA summary by ID."""
        return db.query(models.FractionQASummary).filter(
            models.FractionQASummary.id == summary_id
        ).first()

    @staticmethod
    def get_by_fraction(
        db: Session,
        plan_id: int,
        beam_id: int,
        fraction_number: int,
        plan_version: int = 1,
    ) -> Optional[models.FractionQASummary]:
        """Get fraction summary by plan+beam+fraction+version."""
        return db.query(models.FractionQASummary).filter(
            models.FractionQASummary.plan_id == plan_id,
            models.FractionQASummary.beam_id == beam_id,
            models.FractionQASummary.fraction_number == fraction_number,
            models.FractionQASummary.plan_version == plan_version,
        ).first()

    @staticmethod
    def list_by_plan_beam(
        db: Session,
        plan_id: int,
        beam_id: int,
        plan_version: Optional[int] = None,
    ) -> List[models.FractionQASummary]:
        """List all fraction summaries for a plan+beam."""
        query = db.query(models.FractionQASummary).filter(
            models.FractionQASummary.plan_id == plan_id,
            models.FractionQASummary.beam_id == beam_id,
        )
        if plan_version is not None:
            query = query.filter(
                models.FractionQASummary.plan_version == plan_version
            )
        return query.order_by(
            models.FractionQASummary.fraction_number.asc()
        ).all()

    @staticmethod
    def list_by_plan(
        db: Session,
        plan_id: int,
    ) -> List[models.FractionQASummary]:
        """List all fraction summaries for a plan."""
        return db.query(models.FractionQASummary).filter(
            models.FractionQASummary.plan_id == plan_id
        ).order_by(
            models.FractionQASummary.plan_version.asc(),
            models.FractionQASummary.fraction_number.asc(),
        ).all()

    @staticmethod
    def update(
        db: Session,
        summary_id: int,
        update_data: Dict[str, Any],
    ) -> Optional[models.FractionQASummary]:
        """Update an existing fraction QA summary."""
        summary = CRUDFractionQASummary.get(db, summary_id)
        if not summary:
            return None
        for key, value in update_data.items():
            if hasattr(summary, key) and value is not None:
                setattr(summary, key, value)
        db.commit()
        db.refresh(summary)
        return summary

    @staticmethod
    def update_from_qa_result(
        db: Session,
        qa_result: models.QAResult,
    ) -> models.FractionQASummary:
        """Update or create fraction summary from a new QA result.

        Aggregates metrics:
        - If this is the first result for this fraction, create a new summary
        - If a summary exists, update with latest values
        """
        existing = CRUDFractionQASummary.get_by_fraction(
            db,
            plan_id=qa_result.plan_id,
            beam_id=qa_result.beam_id,
            fraction_number=qa_result.fraction_number,
            plan_version=qa_result.plan_version,
        )

        all_results = db.query(models.QAResult).filter(
            models.QAResult.plan_id == qa_result.plan_id,
            models.QAResult.beam_id == qa_result.beam_id,
            models.QAResult.fraction_number == qa_result.fraction_number,
            models.QAResult.plan_version == qa_result.plan_version,
        ).all()

        num_results = len(all_results)
        pass_count = sum(1 for r in all_results if r.overall_pass)
        pass_rate_pct = (pass_count / num_results * 100.0) if num_results > 0 else 0.0

        max_devs = [r.max_leaf_deviation_mm for r in all_results if r.max_leaf_deviation_mm is not None]
        mean_devs = [r.mean_leaf_deviation_mm for r in all_results if r.mean_leaf_deviation_mm is not None]
        rmse_vals = [r.rmse_mm for r in all_results if r.rmse_mm is not None]
        dose_devs = [r.dose_rate_deviation_pct for r in all_results if r.dose_rate_deviation_pct is not None]
        cp_rates = [r.control_point_pass_rate_pct for r in all_results if r.control_point_pass_rate_pct is not None]

        latest = qa_result

        update_dict = {
            "num_qa_results": num_results,
            "latest_qa_result_id": qa_result.id,
            "max_leaf_deviation_mm": max(max_devs) if max_devs else None,
            "mean_leaf_deviation_mm": float(np.mean(mean_devs)) if mean_devs else None,
            "rmse_mm": float(np.mean(rmse_vals)) if rmse_vals else None,
            "dose_rate_deviation_pct": float(np.mean(dose_devs)) if dose_devs else None,
            "control_point_pass_rate_pct": float(np.mean(cp_rates)) if cp_rates else None,
            "overall_pass_rate_pct": pass_rate_pct,
            "qa_date": qa_result.qa_date,
        }

        if existing:
            updated = CRUDFractionQASummary.update(db, existing.id, update_dict)
            return updated
        else:
            summary_in = schemas.FractionQASummaryCreate(
                plan_id=qa_result.plan_id,
                beam_id=qa_result.beam_id,
                fraction_number=qa_result.fraction_number,
                plan_version=qa_result.plan_version,
                **{k: v for k, v in update_dict.items() if k != "qa_date"}
            )
            return CRUDFractionQASummary.create(db, summary_in)

    @staticmethod
    def delete(db: Session, summary_id: int) -> bool:
        """Delete a fraction QA summary."""
        summary = CRUDFractionQASummary.get(db, summary_id)
        if summary:
            db.delete(summary)
            db.commit()
            return True
        return False


patient_alias = CRUDPatientAlias()
plan = CRUDPlan()
beam = CRUDBeam()
qa_result = CRUDQAResult()
leaf_error_sample = CRUDLeafErrorSample()
fraction_qa_summary = CRUDFractionQASummary()
