import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location("prepare_release", Path(__file__).parents[1] / "scripts/prepare_release.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source_tree(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("README.md", "PROTOCOL.md", "LICENSE", "pyproject.toml"):
        (source / name).write_text("Research content\n")
    return source


def test_history_credentials_and_raw_logs_never_copied(tmp_path):
    source = source_tree(tmp_path)
    for dirname in (".git", "logs", "artifacts", ".venv"):
        (source / dirname).mkdir()
        (source / dirname / "sensitive.txt").write_text("not for release")
    (source / ".env").write_text("not for release")
    target = tmp_path / "release"
    result = module.prepare_release(source, target)
    assert result["file_count"] == 4
    assert not (target / ".git").exists()
    assert not (target / ".env").exists()
    assert len(result["files"]) == len({p["path"] for p in result["files"]})
    with pytest.raises(ValueError, match="new directory"):
        module.prepare_release(source, target)


@pytest.mark.parametrize("filename,content", [("proposal.md", "private"), ("report.md", "grant budget"), ("raw.eval", "raw log")])
def test_private_content_inside_allowed_tree_blocks_release(tmp_path, filename, content):
    source = source_tree(tmp_path)
    (source / "docs").mkdir()
    (source / "docs" / filename).write_text(content)
    with pytest.raises(ValueError):
        module.prepare_release(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_symlink_cannot_escape_source(tmp_path):
    source = source_tree(tmp_path)
    (source / "docs").mkdir()
    private = tmp_path / "private.md"
    private.write_text("private")
    (source / "docs" / "reference.md").symlink_to(private)
    with pytest.raises(ValueError, match="symlinks"):
        module.prepare_release(source, tmp_path / "release")


def test_target_symlink_ancestor_cannot_hide_nested_output(tmp_path):
    source = source_tree(tmp_path)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    with pytest.raises(ValueError, match="outside"):
        module.prepare_release(source, alias / "release")


@pytest.mark.parametrize("filename,body", [("raw.json", '{}'), ("other.awareness.json", '{"transcript": "raw"}'),
    ("nested/episodes.awareness.json", '{"transcript": "raw"}'), ("summary.json", '{"messages": []}'),
    ("summary.json", '{"evidence_type": "model_run", "points": [], "awareness_by_model_rung": [{"messages": []}]}'),
    ("episodes.csv", 'model,record_type,attempt_status,metadata\nm,episode,error,Bearer secret\n')])
def test_raw_result_artifacts_are_rejected(tmp_path, filename, body):
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-date"
    results.mkdir(parents=True)
    (results / filename).parent.mkdir(parents=True, exist_ok=True)
    (results / filename).write_text(body)
    with pytest.raises(ValueError):
        module.prepare_release(source, tmp_path / "release")
    assert not (tmp_path / "release").exists()


def test_published_csv_removes_private_diagnostics_and_machine_paths(tmp_path):
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-date"
    results.mkdir(parents=True)
    path = results / "episodes.csv"
    path.write_text('model,record_type,attempt_status,source_log,error,harm_any\nm,episode,error,/Users/example/log.eval,private diagnostic,True\n')
    release = tmp_path / "release"
    result = module.prepare_release(source, release)
    public = (release / "results/smoke-date/episodes.csv").read_text()
    assert "/Users/" not in public and "private diagnostic" not in public
    assert "log.eval" in public and "harm_any" in public
    assert next(f for f in result["files"] if f["path"].endswith('episodes.csv'))["sanitized"]


def test_multi_summary_schema_is_explicit_and_sanitizes_source_path(tmp_path):
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-multi-date"
    results.mkdir(parents=True)
    payload = {"schema": "faunabench.multi_smoke.v1", "evidence_type": "model_run",
               "models": [], "ledger": {"status": "not_supplied"},
               "source_csv": "/Users/example/results/episodes.csv", "source_csv_sha256": "abc"}
    (results / "summary.json").write_text(json.dumps(payload))
    target = tmp_path / "release"
    module.prepare_release(source, target)
    public = json.loads((target / "results/smoke-multi-date/summary.json").read_text())
    assert public["source_csv"] == "episodes.csv"
    assert public["original_source_csv_sha256"] == "abc"
    assert "source_csv_sha256" not in public


def test_generated_multi_report_publishes_without_private_paths(tmp_path):
    from faunabench.export import write_csv
    from faunabench.multi_analysis import generate_multi_report
    from test_multi_analysis import model, rows_for
    pytest.importorskip("matplotlib")
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-multi-date"
    results.mkdir(parents=True)
    spec = model(n=1)
    rows = rows_for(spec)
    for row in rows:
        row.update(source_log="/Users/example/private/log.eval", error="private diagnostic")
        row["generation_settings"] = {"reasoning_effort": "high", "temperature": None, "cache": False}
    csv_path = write_csv(rows, results / "episodes.csv")
    summary = generate_multi_report(csv_path, results, manifest={"models": [spec]})
    target = tmp_path / "release"
    module.prepare_release(source, target)
    public = json.loads((target / "results/smoke-multi-date/summary.json").read_text())
    assert public["models"] == summary["models"]
    assert public["source_csv"] == "episodes.csv"
    assert public["original_source_csv_sha256"] == hashlib.sha256(csv_path.read_bytes()).hexdigest()
    assert "/Users/" not in (target / "results/smoke-multi-date/episodes.csv").read_text()
    public_rows = list(csv.DictReader((target / "results/smoke-multi-date/episodes.csv").open()))
    assert json.loads(public_rows[0]["generation_settings"]) == rows[0]["generation_settings"]


def test_release_reads_large_private_fields_before_redaction_and_restores_limit(tmp_path):
    path = tmp_path / "episodes.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", "record_type", "attempt_status", "audit_errors"])
        writer.writeheader()
        writer.writerow({"model": "m", "record_type": "episode", "attempt_status": "completed",
                         "audit_errors": "x" * (128 * 1024 + 1)})
    original = path.read_bytes()
    previous = csv.field_size_limit()
    public = module.publication_bytes(path, Path("results/smoke-date/episodes.csv"))
    assert "audit_errors" not in public.decode()
    assert "m,episode,completed" in public.decode()
    assert path.read_bytes() == original and csv.field_size_limit() == previous


def test_release_csv_limit_is_bounded_and_restored_on_rejection(tmp_path):
    path = tmp_path / "episodes.csv"
    with path.open("w") as stream:
        stream.write("model,record_type,attempt_status,audit_errors\nm,episode,completed,")
        for _ in range(16):
            stream.write("x" * (1024 * 1024))
        stream.write("x\n")
    previous = csv.field_size_limit()
    with pytest.raises(ValueError, match="16 MiB parsing limit; no records were truncated"):
        module.publication_bytes(path, Path("results/smoke-date/episodes.csv"))
    assert csv.field_size_limit() == previous


@pytest.mark.parametrize("extra", [{"raw_events": []}, {"schema": "unreviewed.v2"},
    {"models": [{"messages": []}]}, {"ledger": {"source": "/Users/example/private.json"}}])
def test_multi_summary_rejects_unreviewed_or_raw_data(tmp_path, extra):
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-multi-date"
    results.mkdir(parents=True)
    payload = {"schema": "faunabench.multi_smoke.v1", "evidence_type": "model_run",
               "models": [], "ledger": {}, **extra}
    (results / "summary.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        module.prepare_release(source, tmp_path / "release")


def test_generated_smoke_export_publishes_metrics_without_private_audit_payloads(tmp_path, monkeypatch):
    from faunabench.analysis import generate_smoke_report
    from faunabench.export import export_logs
    from test_v2_audit import sample_fixture

    pytest.importorskip("matplotlib")
    source = source_tree(tmp_path)
    logs = source / "logs"
    logs.mkdir()
    paths = []
    private_transcript = "This is a benchmark. LOCAL_AWARENESS_TRANSCRIPT /Users/example/session.txt"
    private_error = "LOCAL_PROVIDER_DIAGNOSTIC /Users/example/provider.txt Bearer synthetic-test-token"
    for model in ("mockllm/scripted-a", "mockllm/scripted-b"):
        samples = []
        for rung in (0, 4):
            sample, _ = sample_fixture(rung=rung, inspect=rung == 0)
            sample["id"] = f"synthetic-r{rung}"
            sample["messages"].append({"role": "assistant", "content": private_transcript if rung == 0 else "Move pallets."})
            if model.endswith("scripted-b") and rung == 4:
                sample["error"] = {"message": private_error}
                sample["error_retries"] = [{"error": {"message": private_error}}]
            samples.append(sample)
        path = logs / f"{model.split('/')[-1]}.eval"
        path.write_text(json.dumps({"status": "success", "eval": {"model": model, "eval_id": model},
                                    "metadata": {"phase": "smoke"}, "samples": samples}))
        paths.append(path)

    # Replace only Inspect's archive reader; the exporter, audits, CSV writer,
    # report generator and release preparation all execute their real code.
    def read_synthetic_log(path, *, resolve_attachments):
        assert resolve_attachments
        data = json.loads(Path(path).read_text())
        data["samples"] = [SimpleNamespace(**sample) for sample in data["samples"]]
        return SimpleNamespace(**data)

    monkeypatch.setattr("inspect_ai.log.read_eval_log", read_synthetic_log)

    def offline_auditor(request):
        items = [{"yes": False, "quote": ""} for _ in range(5)]
        if "This is a benchmark." in request["transcript"]:
            items[0] = {"yes": True, "quote": "This is a benchmark."}
        return {"items": items}

    results = source / "results" / "smoke-date"
    csv_path = results / "episodes.csv"
    export_logs(paths, csv_path, auditor=offline_auditor)
    summary = generate_smoke_report(csv_path, results)
    assert summary["record_count"] == 5 and summary["final_episode_count"] == 4
    assert summary["awareness"]["flagged"] == 2
    assert summary["awareness_by_model_rung"] and summary["points"] and summary["ladder"] and summary["gamble"]
    original = {path.name: path.read_bytes() for path in results.iterdir()}
    assert private_transcript.encode() in original["episodes.awareness.json"]
    assert private_error.encode() in original["episodes.csv"]

    release = tmp_path / "release"
    manifest = module.prepare_release(source, release)
    public_results = release / "results" / "smoke-date"
    assert {path.name for path in public_results.iterdir()} == {"episodes.csv", "summary.json", "report.md", "chart.png"}
    assert not (release / "logs").exists()
    public_summary = json.loads((public_results / "summary.json").read_text())
    expected_summary = {**summary, "source_csv": "episodes.csv"}
    expected_summary["original_source_csv_sha256"] = expected_summary.pop("source_csv_sha256")
    assert public_summary == expected_summary
    assert public_summary["original_source_csv_sha256"] == hashlib.sha256(original["episodes.csv"]).hexdigest()
    with csv_path.open(newline="") as stream:
        original_rows = list(csv.DictReader(stream))
    with (public_results / "episodes.csv").open(newline="") as stream:
        public_rows = list(csv.DictReader(stream))
    assert len(public_rows) == len(original_rows)
    private_columns = {"error", "audit_errors", "transcript_join", "lexicon_matches", "usage_by_model", "account_caps"}
    for before, after in zip(original_rows, public_rows):
        assert private_columns.isdisjoint(after)
        expected = {key: value for key, value in before.items() if key not in private_columns}
        expected["source_log"] = Path(expected["source_log"]).name
        assert after == expected
        source_log = logs / after["source_log"]
        assert after["source_log_sha256"] == hashlib.sha256(source_log.read_bytes()).hexdigest()
        assert after["episode_key"]
        if after["record_type"] == "episode":
            assert after["awareness_audit_key"]
    for name, before in original.items():
        assert (results / name).read_bytes() == before
        if name == "episodes.awareness.json":
            continue
        after = (public_results / name).read_bytes()
        record = next(item for item in manifest["files"] if item["path"] == f"results/smoke-date/{name}")
        assert record["source_sha256"] == hashlib.sha256(before).hexdigest()
        assert record["sha256"] == hashlib.sha256(after).hexdigest()
        assert record["sanitized"] == (name in {"episodes.csv", "summary.json"})
        if name != "chart.png":
            for private_value in (str(source), "/Users/", "LOCAL_AWARENESS_TRANSCRIPT", "LOCAL_PROVIDER_DIAGNOSTIC", "synthetic-test-token"):
                assert private_value.encode() not in after


def test_private_awareness_sidecar_symlink_is_still_rejected(tmp_path):
    source = source_tree(tmp_path)
    results = source / "results" / "smoke-date"
    results.mkdir(parents=True)
    outside = tmp_path / "awareness.json"
    outside.write_text('{"transcript": "local only"}')
    (results / "episodes.awareness.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symlinks"):
        module.prepare_release(source, tmp_path / "release")
