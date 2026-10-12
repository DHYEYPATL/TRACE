"""
Security and Authorization Tests for TRACE API.
Verifies network security, authentication, RBAC authorization,
polygon input validation, and demo-fault protection.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from trace.api.routes import create_app
from trace.config import AppConfig, SecurityConfig
from trace.pipeline import TracePipeline
from trace.platform.db import DatabaseManager


@pytest.fixture
def test_app_and_client(tmp_path):
    cfg = AppConfig()
    cfg.security.demo_mode = False # Default disabled
    cfg.security.require_auth = True # Enforce auth
    cfg.security.admin_token = "admin-secret-token"
    cfg.security.operator_token = "operator-secret-token"
    cfg.security.viewer_token = "viewer-secret-token"

    db_path = tmp_path / "test.db"
    db_mgr = DatabaseManager(str(db_path))

    pipeline = TracePipeline(cfg, db_mgr)
    app = create_app(pipeline)
    client = TestClient(app)
    return client, cfg


def test_auth_required_when_enabled(test_app_and_client):
    client, cfg = test_app_and_client

    # Request without token should return 401
    resp = client.post("/api/settings", json={"timers": {}})
    assert resp.status_code == 401
    assert "Authentication required" in resp.json()["detail"]


def test_invalid_token_rejected(test_app_and_client):
    client, cfg = test_app_and_client

    resp = client.post(
        "/api/settings",
        json={"timers": {}},
        headers={"Authorization": "Bearer totally-fake-token"}
    )
    assert resp.status_code == 401
    assert "Invalid authentication token" in resp.json()["detail"]


def test_rbac_viewer_forbidden_from_admin_action(test_app_and_client):
    client, cfg = test_app_and_client

    # Viewer token trying to update admin settings -> 403 Forbidden
    resp = client.post(
        "/api/settings",
        json={"timers": {}},
        headers={"Authorization": f"Bearer {cfg.security.viewer_token}"}
    )
    assert resp.status_code == 403
    assert "Forbidden" in resp.json()["detail"]


def test_rbac_operator_forbidden_from_admin_action(test_app_and_client):
    client, cfg = test_app_and_client

    # Operator token trying to update settings -> 403 Forbidden
    resp = client.post(
        "/api/settings",
        json={"timers": {}},
        headers={"Authorization": f"Bearer {cfg.security.operator_token}"}
    )
    assert resp.status_code == 403


def test_admin_permitted_on_settings(test_app_and_client):
    client, cfg = test_app_and_client

    resp = client.post(
        "/api/settings",
        json={"timers": {"moved_votes": 6}},
        headers={"Authorization": f"Bearer {cfg.security.admin_token}"}
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_demo_fault_forbidden_when_demo_mode_disabled(test_app_and_client):
    client, cfg = test_app_and_client

    # Even with admin token, fault injection is blocked if demo_mode is False!
    resp = client.post(
        "/api/demo/inject",
        json={"fault_type": "cover_lens", "duration_seconds": 3.0},
        headers={"Authorization": f"Bearer {cfg.security.admin_token}"}
    )
    assert resp.status_code == 403
    assert "Demo fault injection is disabled" in resp.json()["detail"]


def test_polygon_validation_rejects_malformed_geometry(test_app_and_client):
    client, cfg = test_app_and_client

    # Polygon with only 2 vertices (degenerate line)
    bad_req = {
        "id": "asset-bad",
        "name": "Bad Asset",
        "class": "safe",
        "region": [[10, 10], [20, 20]],
    }
    resp = client.post(
        "/api/assets",
        json=bad_req,
        headers={"Authorization": f"Bearer {cfg.security.admin_token}"}
    )
    assert resp.status_code == 400
    assert "at least 3 vertices" in resp.json()["detail"]
