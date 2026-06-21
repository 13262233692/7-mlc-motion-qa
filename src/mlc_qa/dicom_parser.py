"""
Simplified DICOM-RT Plan Parser.

Parses both standard DICOM-RT Plan files (using pydicom) and simplified
JSON representation for testing and data exchange.

Extracts: beams, control points, leaf positions, dose rates, gantry angles.
"""
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import numpy as np

try:
    import pydicom
    PYDICOM_AVAILABLE = True
except ImportError:
    PYDICOM_AVAILABLE = False


@dataclass
class ControlPoint:
    """Control point data structure."""
    index: int
    cumulative_meterset_weight: float
    dose_rate: float
    gantry_angle: float
    leaf_positions_bank_a: np.ndarray  # shape: (num_leaves,)
    leaf_positions_bank_b: np.ndarray  # shape: (num_leaves,)


@dataclass
class BeamData:
    """Beam data structure."""
    beam_number: int
    beam_name: str
    beam_type: str
    energy: str
    control_points: List[ControlPoint] = field(default_factory=list)

    @property
    def num_control_points(self) -> int:
        return len(self.control_points)

    @property
    def num_leaves(self) -> int:
        if self.control_points:
            return len(self.control_points[0].leaf_positions_bank_a)
        return 0

    def get_cumulative_weights(self) -> np.ndarray:
        """Get cumulative meterset weights array."""
        return np.array([cp.cumulative_meterset_weight for cp in self.control_points])

    def get_dose_rates(self) -> np.ndarray:
        """Get dose rates array."""
        return np.array([cp.dose_rate for cp in self.control_points])

    def get_gantry_angles(self, unwrap: bool = False) -> np.ndarray:
        """Get gantry angles array.

        Args:
            unwrap: If True, unwrap angles to avoid 0/360 discontinuity.
        """
        angles = np.array([cp.gantry_angle for cp in self.control_points], dtype=np.float64)
        if unwrap:
            if len(angles) >= 2:
                unwrapped = np.unwrap(np.deg2rad(angles))
                angles = np.rad2deg(unwrapped)
        return angles

    def get_leaf_positions_bank_a(self) -> np.ndarray:
        """Get bank A leaf positions array (num_control_points x num_leaves)."""
        if not self.control_points:
            return np.array([])
        return np.vstack([cp.leaf_positions_bank_a for cp in self.control_points])

    def get_leaf_positions_bank_b(self) -> np.ndarray:
        """Get bank B leaf positions array (num_control_points x num_leaves)."""
        if not self.control_points:
            return np.array([])
        return np.vstack([cp.leaf_positions_bank_b for cp in self.control_points])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "beam_number": self.beam_number,
            "beam_name": self.beam_name,
            "beam_type": self.beam_type,
            "energy": self.energy,
            "control_points": [
                {
                    "index": cp.index,
                    "cumulative_meterset_weight": float(cp.cumulative_meterset_weight),
                    "dose_rate": float(cp.dose_rate),
                    "gantry_angle": float(cp.gantry_angle),
                    "leaf_positions_bank_a": cp.leaf_positions_bank_a.tolist(),
                    "leaf_positions_bank_b": cp.leaf_positions_bank_b.tolist(),
                }
                for cp in self.control_points
            ],
        }


@dataclass
class PlanData:
    """Complete DICOM-RT plan data."""
    plan_uid: str
    plan_name: str
    modality: str
    beams: List[BeamData] = field(default_factory=list)

    @property
    def num_beams(self) -> int:
        return len(self.beams)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "plan_uid": self.plan_uid,
            "plan_name": self.plan_name,
            "modality": self.modality,
            "beams": [beam.to_dict() for beam in self.beams],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def get_beam_by_name(self, beam_name: str) -> Optional[BeamData]:
        """Get beam by name."""
        for beam in self.beams:
            if beam.beam_name == beam_name:
                return beam
        return None

    def get_beam_by_number(self, beam_number: int) -> Optional[BeamData]:
        """Get beam by number."""
        for beam in self.beams:
            if beam.beam_number == beam_number:
                return beam
        return None


class DICOMParserError(Exception):
    """Custom exception for DICOM parsing errors."""
    pass


class MissingControlPointError(DICOMParserError):
    """Raised when control points are missing or incomplete."""
    pass


class LeafCountMismatchError(DICOMParserError):
    """Raised when leaf count is inconsistent across control points."""
    pass


class DicomRTParser:
    """Parser for DICOM-RT Plan files and simplified JSON plans."""

    @classmethod
    def parse(cls, file_path: str) -> PlanData:
        """
        Parse a DICOM-RT Plan file or simplified JSON plan.

        Args:
            file_path: Path to the DICOM file or JSON file.

        Returns:
            PlanData object containing parsed plan information.

        Raises:
            DICOMParserError: If parsing fails.
            FileNotFoundError: If the file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()

        if suffix in (".json", ".txt"):
            return cls._parse_json(path)
        elif suffix in (".dcm", ".dicom", ""):
            return cls._parse_dicom(path)
        else:
            try:
                return cls._parse_json(path)
            except (json.JSONDecodeError, KeyError):
                try:
                    return cls._parse_dicom(path)
                except Exception as e:
                    raise DICOMParserError(f"Unable to parse file as JSON or DICOM: {e}")

    @classmethod
    def parse_string(cls, content: str) -> PlanData:
        """Parse plan data from string content (JSON or DICOM bytes)."""
        try:
            data = json.loads(content)
            return cls._build_plan_from_dict(data)
        except (json.JSONDecodeError, KeyError):
            raise DICOMParserError("Unable to parse string as JSON plan")

    @classmethod
    def _parse_json(cls, path: Path) -> PlanData:
        """Parse simplified JSON plan format."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls._build_plan_from_dict(data)

    @classmethod
    def _build_plan_from_dict(cls, data: Dict[str, Any]) -> PlanData:
        """Build PlanData from dictionary."""
        required_fields = ["plan_uid", "modality", "beams"]
        for field in required_fields:
            if field not in data:
                raise DICOMParserError(f"Missing required field: {field}")

        plan = PlanData(
            plan_uid=data["plan_uid"],
            plan_name=data.get("plan_name", data["plan_uid"]),
            modality=data["modality"],
        )

        for beam_data in data.get("beams", []):
            beam = cls._build_beam_from_dict(beam_data)
            plan.beams.append(beam)

        return plan

    @classmethod
    def _build_beam_from_dict(cls, beam_data: Dict[str, Any]) -> BeamData:
        """Build BeamData from dictionary."""
        required_fields = ["beam_number", "beam_name", "control_points"]
        for field in required_fields:
            if field not in beam_data:
                raise DICOMParserError(f"Missing required beam field: {field}")

        beam = BeamData(
            beam_number=beam_data["beam_number"],
            beam_name=beam_data["beam_name"],
            beam_type=beam_data.get("beam_type", "STATIC"),
            energy=beam_data.get("energy", "6MV"),
        )

        control_points_data = beam_data.get("control_points", [])
        if not control_points_data:
            raise MissingControlPointError(
                f"No control points found for beam {beam_data['beam_name']}"
            )

        num_leaves = None
        for i, cp_data in enumerate(control_points_data):
            cp = cls._build_control_point_from_dict(cp_data, i)
            if num_leaves is None:
                num_leaves = len(cp.leaf_positions_bank_a)
            elif len(cp.leaf_positions_bank_a) != num_leaves:
                raise LeafCountMismatchError(
                    f"Leaf count mismatch at control point {i}: "
                    f"expected {num_leaves}, got {len(cp.leaf_positions_bank_a)}"
                )
            beam.control_points.append(cp)

        if num_leaves == 0:
            raise LeafCountMismatchError(
                f"No leaf positions found in beam {beam_data['beam_name']}"
            )

        cls._validate_control_points(beam.control_points)
        return beam

    @classmethod
    def _build_control_point_from_dict(
        cls, cp_data: Dict[str, Any], index: int
    ) -> ControlPoint:
        """Build ControlPoint from dictionary."""
        required_fields = [
            "cumulative_meterset_weight",
            "leaf_positions_bank_a",
            "leaf_positions_bank_b",
        ]
        for field in required_fields:
            if field not in cp_data:
                raise DICOMParserError(
                    f"Missing required control point field: {field} at index {index}"
                )

        bank_a = np.array(cp_data["leaf_positions_bank_a"], dtype=np.float64)
        bank_b = np.array(cp_data["leaf_positions_bank_b"], dtype=np.float64)

        if len(bank_a) != len(bank_b):
            raise LeafCountMismatchError(
                f"Bank A and B leaf count mismatch at control point {index}: "
                f"bank A={len(bank_a)}, bank B={len(bank_b)}"
            )

        return ControlPoint(
            index=index,
            cumulative_meterset_weight=float(cp_data["cumulative_meterset_weight"]),
            dose_rate=float(cp_data.get("dose_rate", 0.0)),
            gantry_angle=float(cp_data.get("gantry_angle", 0.0)),
            leaf_positions_bank_a=bank_a,
            leaf_positions_bank_b=bank_b,
        )

    @staticmethod
    def _validate_control_points(control_points: List[ControlPoint]) -> None:
        """Validate control points for proper ordering and completeness."""
        if len(control_points) < 2:
            raise MissingControlPointError(
                f"Plan must have at least 2 control points, got {len(control_points)}"
            )

        weights = [cp.cumulative_meterset_weight for cp in control_points]

        if weights[0] != 0.0:
            raise DICOMParserError(
                f"First control point must have cumulative weight 0.0, got {weights[0]}"
            )

        if not (np.isclose(weights[-1], 1.0, atol=1e-6) or weights[-1] > 0.9):
            raise DICOMParserError(
                f"Last control point must have cumulative weight ~1.0, got {weights[-1]}"
            )

        for i in range(1, len(weights)):
            if weights[i] < weights[i - 1]:
                raise DICOMParserError(
                    f"Cumulative weights must be non-decreasing: "
                    f"cp[{i}]={weights[i]} < cp[{i-1}]={weights[i-1]}"
                )

    @classmethod
    def _parse_dicom(cls, path: Path) -> PlanData:
        """Parse standard DICOM-RT Plan file using pydicom."""
        if not PYDICOM_AVAILABLE:
            raise DICOMParserError(
                "pydicom is not installed. Install with: pip install pydicom"
            )

        try:
            ds = pydicom.dcmread(str(path), force=True)
        except Exception as e:
            raise DICOMParserError(f"Failed to read DICOM file: {e}")

        if not hasattr(ds, "SOPClassUID"):
            raise DICOMParserError("Not a valid DICOM file: missing SOPClassUID")

        if "RT Plan Module" not in ds.SOPClassUID.name:
            raise DICOMParserError(
                f"Not a DICOM-RT Plan file. SOP Class: {ds.SOPClassUID.name}"
            )

        if not hasattr(ds, "SOPInstanceUID"):
            raise DICOMParserError("Not a valid DICOM file: missing SOPInstanceUID")

        if not hasattr(ds, "Modality"):
            raise DICOMParserError("Not a valid DICOM file: missing Modality")

        plan = PlanData(
            plan_uid=ds.SOPInstanceUID,
            plan_name=getattr(ds, "RTPlanName", ds.SOPInstanceUID),
            modality=ds.Modality,
        )

        if not hasattr(ds, "BeamSequence") or len(ds.BeamSequence) == 0:
            raise DICOMParserError("No beams found in DICOM-RT Plan")

        for beam_idx, beam_seq in enumerate(ds.BeamSequence):
            beam = cls._parse_dicom_beam(beam_seq, beam_idx)
            plan.beams.append(beam)

        return plan

    @classmethod
    def _parse_dicom_beam(cls, beam_seq: Any, beam_idx: int) -> BeamData:
        """Parse a single beam from DICOM BeamSequence."""
        beam = BeamData(
            beam_number=getattr(beam_seq, "BeamNumber", beam_idx + 1),
            beam_name=getattr(beam_seq, "BeamName", f"Beam_{beam_idx + 1}"),
            beam_type=getattr(beam_seq, "BeamType", "STATIC"),
            energy=cls._extract_energy(beam_seq),
        )

        if not hasattr(beam_seq, "ControlPointSequence"):
            raise MissingControlPointError(
                f"No ControlPointSequence found for beam {beam.beam_name}"
            )

        num_leaves = None
        for cp_idx, cp_seq in enumerate(beam_seq.ControlPointSequence):
            cp = cls._parse_dicom_control_point(cp_seq, cp_idx, num_leaves)
            if num_leaves is None and len(cp.leaf_positions_bank_a) > 0:
                num_leaves = len(cp.leaf_positions_bank_a)
            beam.control_points.append(cp)

        if num_leaves is None or num_leaves == 0:
            raise DICOMParserError(
                f"No leaf positions found in beam {beam.beam_name}"
            )

        cls._validate_control_points(beam.control_points)
        return beam

    @staticmethod
    def _extract_energy(beam_seq: Any) -> str:
        """Extract beam energy from DICOM."""
        if hasattr(beam_seq, "ControlPointSequence") and len(beam_seq.ControlPointSequence) > 0:
            cp = beam_seq.ControlPointSequence[0]
            if hasattr(cp, "NominalBeamEnergy"):
                return f"{cp.NominalBeamEnergy}MV"
        return "6MV"

    @classmethod
    def _parse_dicom_control_point(
        cls, cp_seq: Any, index: int, expected_num_leaves: Optional[int]
    ) -> ControlPoint:
        """Parse a single control point from DICOM."""
        cumulative_weight = getattr(cp_seq, "CumulativeMetersetWeight", 0.0)
        dose_rate = getattr(cp_seq, "DoseRateSet", 0.0)
        gantry_angle = getattr(cp_seq, "GantryAngle", 0.0)

        bank_a, bank_b = cls._extract_leaf_positions(cp_seq)

        if expected_num_leaves is not None and len(bank_a) > 0:
            if len(bank_a) != expected_num_leaves:
                raise LeafCountMismatchError(
                    f"Leaf count mismatch at control point {index}: "
                    f"expected {expected_num_leaves}, got {len(bank_a)}"
                )

        return ControlPoint(
            index=index,
            cumulative_meterset_weight=float(cumulative_weight),
            dose_rate=float(dose_rate),
            gantry_angle=float(gantry_angle),
            leaf_positions_bank_a=bank_a,
            leaf_positions_bank_b=bank_b,
        )

    @staticmethod
    def _extract_leaf_positions(cp_seq: Any) -> Tuple[np.ndarray, np.ndarray]:
        """Extract leaf positions from DICOM control point."""
        bank_a = np.array([], dtype=np.float64)
        bank_b = np.array([], dtype=np.float64)

        if hasattr(cp_seq, "BeamLimitingDevicePositionSequence"):
            for bldp in cp_seq.BeamLimitingDevicePositionSequence:
                rt_dev_type = getattr(bldp, "RTBeamLimitingDeviceType", "")
                if "MLCX" in rt_dev_type or "ASYMX" in rt_dev_type:
                    positions = getattr(bldp, "LeafJawPositions", [])
                    pos_array = np.array(positions, dtype=np.float64)
                    mid = len(pos_array) // 2
                    if len(pos_array) > 0:
                        bank_a = pos_array[:mid]
                        bank_b = pos_array[mid:]

        return bank_a, bank_b


def create_simplified_plan_json(
    plan_uid: str,
    beam_name: str,
    num_leaves: int = 60,
    num_control_points: int = 10,
    max_leaf_position: float = 100.0,
) -> Dict[str, Any]:
    """
    Create a simplified plan JSON structure for testing.

    Args:
        plan_uid: Plan unique identifier.
        beam_name: Beam name.
        num_leaves: Number of MLC leaves per bank.
        num_control_points: Number of control points.
        max_leaf_position: Maximum leaf opening in mm.

    Returns:
        Dictionary in simplified plan format.
    """
    control_points = []
    for i in range(num_control_points):
        weight = i / (num_control_points - 1) if num_control_points > 1 else 0.0
        leaf_pos = max_leaf_position * weight
        control_points.append({
            "cumulative_meterset_weight": weight,
            "dose_rate": 600.0,
            "gantry_angle": 0.0 + 180.0 * weight,
            "leaf_positions_bank_a": [-leaf_pos] * num_leaves,
            "leaf_positions_bank_b": [leaf_pos] * num_leaves,
        })

    return {
        "plan_uid": plan_uid,
        "plan_name": plan_uid,
        "modality": "RTPLAN",
        "beams": [
            {
                "beam_number": 1,
                "beam_name": beam_name,
                "beam_type": "DYNAMIC",
                "energy": "6MV",
                "control_points": control_points,
            }
        ],
    }
