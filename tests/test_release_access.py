"""Release checks for separately audited exploratory access follow-ups."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("access_release", ROOT / "scripts/prepare_release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)

@pytest.fixture
def packet(tmp_path):
    source = ROOT / "results/access-checks-2026-11-08"
    target = tmp_path / source.name
    shutil.copytree(source, target)
    return target

def rehash(packet):
    manifest = json.loads((packet / "manifest.json").read_text())
    for name in manifest["files"]:
        manifest["files"][name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    (packet / "manifest.json").write_text(json.dumps(manifest))
    audit = json.loads((packet / "audit-receipt.json").read_text())
    for name in audit["files_sha256"]:
        audit["files_sha256"][name] = hashlib.sha256((packet / name).read_bytes()).hexdigest()
    (packet / "audit-receipt.json").write_text(json.dumps(audit))

def test_reviewed_access_packet_is_byte_preserved(packet):
    release.validate_access_checks(packet)
    for path in packet.iterdir():
        assert release.publication_bytes(path, Path("results") / packet.name / path.name) == path.read_bytes()

def test_extra_access_record_is_rejected(packet):
    (packet / "raw.txt").write_text("Unreviewed")
    with pytest.raises(ValueError, match="allowlist"):
        release.validate_access_checks(packet)

def test_access_tamper_invalidates_review(packet):
    (packet / "README.md").write_text("Changed after audit")
    with pytest.raises(ValueError, match="hash mismatch"):
        release.validate_access_checks(packet)

@pytest.mark.parametrize("key,value", [("quote", "raw text"), ("request_headers", {}),
                                      ("owner_authority", "not public")])
def test_rehashed_access_raw_fields_are_rejected(packet, key, value):
    path = packet / "summary.json"
    data = json.loads(path.read_text())
    data["extra"] = {key: value}
    path.write_text(json.dumps(data))
    rehash(packet)
    with pytest.raises(ValueError, match="raw field|private text"):
        release.validate_access_checks(packet)

def test_combined_reexport_preserves_both_followups(tmp_path):
    target = tmp_path / "release"
    result = release.prepare_release(ROOT, target)
    for name in ("addendum-2026-11-08", "access-checks-2026-11-08"):
        for path in (ROOT / "results" / name).iterdir():
            assert (target / "results" / name / path.name).read_bytes() == path.read_bytes()
    assert result["history_copied"] is False and not (target / ".git").exists()

