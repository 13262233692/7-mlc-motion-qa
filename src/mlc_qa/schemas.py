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
    fraction_number: int = 0
    plan_version: int = 1
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
    fraction_number: Optional[int] = None
    plan_version: Optional[int] = None


class QASubmitResponse(BaseModel):
    success: bool
    qa_result_id: int
    fraction_summary_id: Optional[int] = None
    message: str
    max_deviation_mm: float
    pass_rate_pct: float
    overall_pass: bool
    fraction_number: int
    plan_version: int


class FractionQASummaryBase(BaseModel):
    plan_id: int
    beam_id: int
    fraction_number: int
    plan_version: int = 1
    num_qa_results: int = 0
    max_leaf_deviation_mm: Optional[float] = None
    mean_leaf_deviation_mm: Optional[float] = None
    rmse_mm: Optional[float] = None
    dose_rate_deviation_pct: Optional[float] = None
    control_point_pass_rate_pct: Optional[float] = None
    overall_pass_rate_pct: Optional[float] = None
    trend_label: Optional[str] = None
    trend_confidence: Optional[float] = None
    deviation_delta_from_previous_mm: Optional[float] = None


class FractionQASummaryCreate(FractionQASummaryBase):
    latest_qa_result_id: Optional[int] = None


class FractionQASummaryResponse(FractionQASummaryBase):
    id: int
    latest_qa_result_id: Optional[int] = None
    qa_date: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TrendAnalysisResult(BaseModel):
    """Complete trend analysis for a plan + beam combination."""
    plan_id: int
    beam_id: int
    total_fractions: int
    total_qa_results: int
    plan_versions: List[int]

    trend_label: str
    trend_confidence: float
    overall_trend_description: str

    max_deviation_trend_slope_mm_per_fraction: Optional[float] = None
    pass_rate_trend_slope_pct_per_fraction: Optional[float] = None

    latest_fraction: int
    latest_max_deviation_mm: Optional[float] = None
    latest_pass_rate_pct: Optional[float] = None

    fractions: List[FractionQASummaryResponse] = []

    chart_data: Dict[str, Any] = {}

    anomaly_flags: List[str] = []
    notes: Optional[str] = None
