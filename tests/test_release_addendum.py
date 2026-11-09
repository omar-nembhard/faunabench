"""Research export regression tests: no provider execution or credentials."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("release_addendum", ROOT / "scripts/prepare_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture
def packet(tmp_path):
    source = ROOT / "results/addendum-2026-11-08"
    target = tmp_path / source.name
    shutil.copytree(source, target)
    return target


def rehash(packet):
    manifest_path = packet / "manifest.json"
    audit_path = packet / "audit-receipt.json"
    manifest = json.loads(manifest_path.read_text())
    audit = json.loads(audit_path.read_text())
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    for name in audit["files_sha256"]:
        audit["files_sha256"][name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    audit_path.write_text(json.dumps(audit))


def test_reviewed_addendum_passes_and_stays_byte_identical(packet):
    before = {p.name: p.read_bytes() for p in packet.iterdir()}
    release.validate_addendum(packet)
    for path in packet.iterdir():
        actual = release.publication_bytes(path, Path("results") / packet.name / path.name)
        assert actual == before[path.name]


@pytest.mark.parametrize("change", ["missing", "extra", "nested", "symlink"])
def test_allowlist_rejects_unreviewed_members(packet, change):
    if change == "missing":
        (packet / "README.md").unlink()
    elif change == "extra":
        (packet / "other.txt").write_text("unreviewed")
    elif change == "nested":
        (packet / "nested").mkdir()
    else:
        (packet / "README.md").unlink()
        (packet / "README.md").symlink_to(packet / "summary.json")
    with pytest.raises(ValueError, match="allowlist"):
        release.validate_addendum(packet)


def test_tamper_after_review_is_rejected(packet):
    (packet / "README.md").write_text("Changed after review")
    with pytest.raises(ValueError, match="hash mismatch"):
        release.validate_addendum(packet)


@pytest.mark.parametrize("value", ["/Users/example/local", "grant application", "account balance", "api_key", "secret"])
def test_even_rehashed_private_text_is_rejected(packet, value):
    if value == "secret":
        value = "sk-" + "ant-" + "x" * 24
    with (packet / "README.md").open("a") as stream:
        stream.write(value)
    rehash(packet)
    with pytest.raises(ValueError, match="private text"):
        release.validate_addendum(packet)


def test_rehashed_raw_quote_is_rejected(packet):
    path = packet / "summary.json"
    data = json.loads(path.read_text())
    data["metadata"] = {"quote": "raw assistant response"}
    path.write_text(json.dumps(data))
    rehash(packet)
    with pytest.raises(ValueError, match="raw field"):
        release.validate_addendum(packet)


def test_account_controls_are_removed_from_csv(tmp_path):
    path = tmp_path / "episodes.csv"
    path.write_text("model,record_type,attempt_status,account_caps,cost_basis,cost\nm,episode,error,private,console_workspace_delta_owner_authorized,0.069796\n")
    text = release.publication_bytes(path, Path("results/smoke-date/episodes.csv")).decode()
    assert "account_caps" not in text and "private" not in text and "owner_authorized" not in text
    assert "estimated_cent_rounded_console_reconciliation" in text and "0.069796" in text


def test_account_observations_block_summary(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"schema": "faunabench.multi_smoke.v1", "evidence_type": "model_run",
        "models": [], "ledger": {"observed_spend_usd": 1.66}}))
    with pytest.raises(ValueError, match="raw provider payload"):
        release.publication_bytes(path, Path("results/smoke-date/summary.json"))


def test_complete_combined_repository_exports_without_history(tmp_path):
    target = tmp_path / "release"
    result = release.prepare_release(ROOT, target)
    assert not (target / ".git").exists()
    assert not (target / ".env").exists()
    for path in (ROOT / "results/addendum-2026-11-08").iterdir():
        assert (target / "results/addendum-2026-11-08" / path.name).read_bytes() == path.read_bytes()
    assert result["history_copied"] is False

