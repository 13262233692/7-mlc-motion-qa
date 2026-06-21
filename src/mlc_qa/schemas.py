"""Pydantic schemas for API data validation and serialization."""
from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class PatientAliasBase(BaseModel):
    anonymous_id: str = Field(..., max_length=64, description="Anonymous patient identifier")


class PatientAliasCreate(PatientAliasBase):
    pass


class PatientAliasResponse(PatientAliasBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True


class PlanBase(BaseModel):
    plan_uid: str = Field(..., max_length=128, description="Plan unique identifier")
    plan_name: Optional[str] = Field(None, max_length=256)
    modality: Optional[str] = Field(None, max_length=32)


class PlanCreate(PlanBase):
    patient_id: int


class PlanResponse(PlanBase):
    id: int
    patient_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class BeamBase(BaseModel):
    beam_name: str
    beam_number: int
    beam_type: Optional[str] = None
    energy: Optional[str] = None


class BeamResponse(BeamBase):
    id: int
    plan_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class LeafErrorSampleBase(BaseModel):
    control_point_index: int
    leaf_index: int
    bank: str
    planned_position_mm: float
    actual_position_mm: float
    deviation_mm: float
    timestamp_sec: Optional[float] = None


class LeafErrorSampleResponse(LeafErrorSampleBase):
    id: int
    qa_result_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class QAResultBase(BaseModel):
    max_leaf_deviation_mm: float
    mean_leaf_deviation_mm: float
    rmse_mm: float
    dose_rate_deviation_pct: float
    control_point_pass_rate_pct: float
    num_control_points: int
    num_failed_control_points: int
    num_leaves: int
    gantry_angle_range: Optional[str] = None
    overall_pass: int
    notes: Optional[str] = None


class QAResultCreate(QAResultBase):
    plan_id: int
    beam_id: int
    log_filename: Optional[str] = None


class QAResultResponse(QAResultBase):
    id: int
    plan_id: int
    beam_id: int
    log_filename: Optional[str]
    qa_date: datetime

    class Config:
        from_attributes = True


class QAResultDetail(QAResultResponse):
    leaf_error_samples: List[LeafErrorSampleResponse] = []


class PlanWithDetail(PlanResponse):
    beams: List[BeamResponse] = []
    qa_results: List[QAResultResponse] = []


class QAUploadResponse(BaseModel):
    success: bool
    qa_result_id: int
    message: str
    max_deviation_mm: float
    pass_rate_pct: float
    overall_pass: bool


class PlanParseResponse(BaseModel):
    success: bool
    plan_uid: str
    modality: Optional[str]
    num_beams: int
    beam_names: List[str]
    message: str


class LogParseResponse(BaseModel):
    success: bool
    num_samples: int
    time_range_sec: float
    num_leaves: int
    message: str


class QASubmitRequest(BaseModel):
    patient_anonymous_id: str
    plan_uid: str
    beam_name: str
    plan_file_content: Optional[str] = None
    log_file_content: Optional[str] = None
    notes: Optional[str] = None
