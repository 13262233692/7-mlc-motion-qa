"""Test fixtures and configuration."""
import os
import sys
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import mlc_qa.database as db_module
from mlc_qa.database import Base, get_db
from mlc_qa.main import app
from mlc_qa.dicom_parser import create_simplified_plan_json
from mlc_qa.log_parser import create_sample_log_csv


TEST_DATABASE_URL = "sqlite:///:memory:"


@pytest.fixture(scope="function")
def db_engine():
    """Create a clean database engine for each test."""
    original_engine = db_module.engine
    original_session_local = db_module.SessionLocal

    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)

    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    db_module.engine = engine
    db_module.SessionLocal = TestingSessionLocal

    yield engine

    Base.metadata.drop_all(bind=engine)
    engine.dispose()

    db_module.engine = original_engine
    db_module.SessionLocal = original_session_local


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Create a clean database session for each test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db_session, db_engine):
    """Create a test client with database override."""

    def override_get_db():
        yield db_session

    app.dependency_overrides = {}
    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_plan_json():
    """Create a sample simplified plan JSON."""
    return create_simplified_plan_json(
        plan_uid="TEST-PLAN-001",
        beam_name="AP Field",
        num_leaves=60,
        num_control_points=10,
    )


@pytest.fixture
def sample_log_csv():
    """Create a sample treatment log CSV."""
    return create_sample_log_csv(
        num_samples=100,
        num_leaves=60,
        duration_sec=30.0,
        noise_std=0.0,
    )


@pytest.fixture
def temp_plan_file(sample_plan_json):
    """Create a temporary plan file."""
    import json
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_plan_json, f)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_log_file(sample_log_csv):
    """Create a temporary log file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(sample_log_csv)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)
