"""Application configuration management."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./mlc_qa.db")
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))

MAX_LEAF_DEVIATION_THRESHOLD = float(os.getenv("MAX_LEAF_DEVIATION_THRESHOLD", "1.0"))
CONTROL_POINT_PASS_THRESHOLD = float(os.getenv("CONTROL_POINT_PASS_THRESHOLD", "95.0"))

NUM_LEAVES_DEFAULT = 60
TOLERANCE_MM = 1.0
