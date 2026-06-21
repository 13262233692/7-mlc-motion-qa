"""SQLAlchemy database models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship

from mlc_qa.database import Base


class PatientAlias(Base):
    """Anonymous patient record - no PHI stored."""
    __tablename__ = "patient_alias"

    id = Column(Integer, primary_key=True, index=True)
    anonymous_id = Column(String(64), unique=True, index=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    plans = relationship("Plan", back_populates="patient")


class Plan(Base):
    """Treatment plan metadata."""
    __tablename__ = "plan"

    id = Column(Integer, primary_key=True, index=True)
    patient_id = Column(Integer, ForeignKey("patient_alias.id"), nullable=False)
    plan_uid = Column(String(128), unique=True, index=True, nullable=False)
    plan_name = Column(String(256))
    modality = Column(String(32))
    created_at = Column(DateTime, default=datetime.utcnow)

    patient = relationship("PatientAlias", back_populates="plans")
    beams = relationship("Beam", back_populates="plan", cascade="all, delete-orphan")
    qa_results = relationship("QAResult", back_populates="plan", cascade="all, delete-orphan")


class Beam(Base):
    """Treatment beam definition from DICOM-RT plan."""
    __tablename__ = "beam"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("plan.id"), nullable=False)
    beam_name = Column(String(128), nullable=False)
    beam_number = Column(Integer, nullable=False)
    beam_type = Column(String(64))
    energy = Column(String(32))
    control_points_data = Column(JSON, nullable=False)
    leaf_positions = Column(JSON, nullable=False)
    dose_rates = Column(JSON)
    gantry_angles = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    plan = relationship("Plan", back_populates="beams")
    qa_results = relationship("QAResult", back_populates="beam", cascade="all, delete-orphan")


class QAResult(Base):
    """QA analysis result."""
    __tablename__ = "qa_result"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("plan.id"), nullable=False)
    beam_id = Column(Integer, ForeignKey("beam.id"), nullable=False)
    fraction_number = Column(Integer, default=0, index=True)
    plan_version = Column(Integer, default=1, index=True)
    log_filename = Column(String(256))
    max_leaf_deviation_mm = Column(Float)
    mean_leaf_deviation_mm = Column(Float)
    rmse_mm = Column(Float)
    dose_rate_deviation_pct = Column(Float)
    control_point_pass_rate_pct = Column(Float)
    num_control_points = Column(Integer)
    num_failed_control_points = Column(Integer)
    num_leaves = Column(Integer)
    gantry_angle_range = Column(String(64))
    qa_date = Column(DateTime, default=datetime.utcnow)
    overall_pass = Column(Integer)
    notes = Column(Text)

    plan = relationship("Plan", back_populates="qa_results")
    beam = relationship("Beam", back_populates="qa_results")
    leaf_error_samples = relationship(
        "LeafErrorSample",
        back_populates="qa_result",
        cascade="all, delete-orphan"
    )


class FractionQASummary(Base):
    """Per-fraction QA summary for trend analysis.

    Aggregates QA results by (plan, beam, fraction_number, plan_version).
    Used for trend analysis of leaf deviation patterns across treatment sessions.
    """
    __tablename__ = "fraction_qa_summary"

    id = Column(Integer, primary_key=True, index=True)
    plan_id = Column(Integer, ForeignKey("plan.id"), nullable=False, index=True)
    beam_id = Column(Integer, ForeignKey("beam.id"), nullable=False, index=True)
    fraction_number = Column(Integer, nullable=False, index=True)
    plan_version = Column(Integer, default=1, index=True)

    num_qa_results = Column(Integer, default=0)
    latest_qa_result_id = Column(Integer, ForeignKey("qa_result.id"))

    max_leaf_deviation_mm = Column(Float)
    mean_leaf_deviation_mm = Column(Float)
    rmse_mm = Column(Float)
    dose_rate_deviation_pct = Column(Float)
    control_point_pass_rate_pct = Column(Float)
    overall_pass_rate_pct = Column(Float)

    trend_label = Column(String(32))
    trend_confidence = Column(Float)
    deviation_delta_from_previous_mm = Column(Float)

    qa_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class LeafErrorSample(Base):
    """Sample leaf error data for detailed analysis."""
    __tablename__ = "leaf_error_sample"

    id = Column(Integer, primary_key=True, index=True)
    qa_result_id = Column(Integer, ForeignKey("qa_result.id"), nullable=False)
    control_point_index = Column(Integer)
    leaf_index = Column(Integer)
    bank = Column(String(16))
    planned_position_mm = Column(Float)
    actual_position_mm = Column(Float)
    deviation_mm = Column(Float)
    timestamp_sec = Column(Float)
    log_time = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

    qa_result = relationship("QAResult", back_populates="leaf_error_samples")
