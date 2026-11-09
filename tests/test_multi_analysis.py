"""Multi-model denominators, historical references and cross-phase accounting."""
from copy import deepcopy
import csv
import io
import json
from pathlib import Path
import runpy

import pytest

from faunabench.export import write_csv
from faunabench.multi_analysis import enrich_rows, generate_multi_report, safe_generation_settings, summarize_multi


def model(name="m", *, n=10, phase="smoke_multi", r0_only=False, score=None):
    result = {"id": name, "model": name, "label": name, "track": "A", "phases": [phase],
              "planned": {"0": n, "4": 0 if r0_only else n}}
    if score is not None:
        result["compassionbench"] = {"score": score, "source_url": "https://example.org/overall", "match_status": "contextual_family"}
    return result


def rows_for(spec, *, inspect=True, detour=True):
    return [{"record_type": "episode", "model": spec["model"], "phase": spec["phases"][0],
             "episode_key": f"{spec['model']}:{rung}:{seed}", "billing_key": f"bill:{spec['model']}:{rung}:{seed}",
             "attempt": 1, "attempt_status": "completed", "rung": int(rung), "seed": seed,
             "deadline": 5, "layout": "LA", "analysis_eligible": True, "world_eligible": True,
             "join_valid": True, "brief_hash_match": True, "banned_word_leak": False,
             "partial_execution": False,
             "inspect_first": inspect, "detour_occupied": detour, "harm_any": not detour,
             "harm_count": 0 if detour else 3, "success_on_time": True, "gamble": False,
             "blind_direct_count": 0, "usd": .02, "evidence_type": "model_run", "provider_retries": 0,
             "served_models": [spec["model"] + "-served"], "served_model_status": "consistent"}
            for rung, n in spec["planned"].items() for seed in range(n)]


def test_includes_declared_haiku_excludes_off_preflights_and_google_footnote():
    haiku = model("haiku", phase="smoke")
    other = model("other")
    rows = rows_for(haiku) + rows_for(other)
    rows += [{**r, "phase": "reasoning_off"} for r in rows_for(haiku)]
    rows += [{**r, "phase": "preflight_multi"} for r in rows_for(other)]
    rows += rows_for(model("free-google", n=1, phase="smoke"))
    result = summarize_multi(rows, {"models": [haiku, other], "footnotes": ["Free Google is separate."]})
    assert result["primary_attempt_count"] == 40
    assert result["complete_planned_design"]
    assert all(m["usable_episodes"] == 20 for m in result["models"])
    assert result["footnotes"] == ["Free Google is separate."]


def test_failures_retries_and_limits_do_not_become_behavioral_zeroes():
    spec = model(n=2)
    rows = rows_for(spec)
    failure = {**rows[0], "billing_key": "failed-bill", "attempt": 2, "attempt_status": "error", "usd": .01}
    rows[1].update(attempt_status="limited", usd=.04)
    result = summarize_multi(rows + [failure, deepcopy(failure)], {"models": [spec]})["models"][0]
    assert result["attempts"] == 5
    assert result["usable_episodes"] == 3
    assert result["failed_attempts"] == 1 and result["limited_attempts"] == 1
    assert result["points"][0]["n"] == 1 and result["points"][0]["k"] == 1
    assert result["cost"]["known_usd"] == pytest.approx(.11)
    assert result["cost_per_usable_episode_usd"] == pytest.approx(.11 / 3)


def test_no_log_infrastructure_failure_counts_attempt_and_cost_but_no_behavior(tmp_path):
    spec = model()
    rows = rows_for(spec)
    missing = rows.pop(0)
    operational = {k: missing[k] for k in ("model", "phase", "rung", "seed", "deadline", "layout")}
    operational.update(record_type="infrastructure_attempt", attempt_status="error", attempt=1,
                       billing_key="journal:missing:1", analysis_eligible=False, world_eligible=False,
                       source_log=None, source_log_sha256=None, source_journal_sha256="journal-snapshot",
                       usd="0.00441485", cost_basis="reconciled_provider_key_usage_delta")
    rows.append(operational)
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["attempts"] == result["attempted_episodes"] == 20
    assert result["completed_episodes"] == result["usable_episodes"] == 19
    assert result["failed_attempts"] == result["infrastructure_attempts"] == 1
    assert result["pending_attempts"] == 0 and result["excluded_episodes"] == 1
    assert result["points"][0]["n"] == 9 and result["points"][1]["n"] == 10
    assert result["points"][0]["total"] == 10 and result["points"][0]["failed_or_voided"] == 1
    assert result["points"][0]["excluding_flagged"]["failed_or_voided"] == 1
    assert result["harm_episodes"]["n"] == 19 and not result["complete_planned_design"]
    assert result["cost"]["known_usd"] == pytest.approx(.38441485)
    assert "reconciled_provider_key_usage_delta" in result["cost_bases"]
    path = write_csv(rows, tmp_path / "episodes.csv")
    generate_multi_report(path, tmp_path, manifest={"models": [spec]})
    assert "| m | 20 / 20 / 19 | 0 | 0 known; 1 attempts unknown | 1/0/0/0 |" in (tmp_path / "report.md").read_text()


def test_reconciled_infrastructure_failure_and_successful_rerun_keep_costs_and_slot(tmp_path):
    spec = model()
    rows = rows_for(spec)
    success = next(r for r in rows if r["rung"] == 0 and r["seed"] == 5)
    success.update(attempt=2, episode_lineage="lost-usage-slot", billing_key="rerun-bill")
    failure = {k: success[k] for k in ("model", "phase", "rung", "seed", "deadline", "layout", "episode_lineage")}
    failure.update(record_type="infrastructure_attempt", attempt=1, attempt_status="error",
                   billing_key="journal:lost-usage-slot:1", analysis_eligible=False, world_eligible=False,
                   usd="0.069796", billing_uncertain=False,
                   cost_basis="console_workspace_delta_owner_authorized")
    rows.append(failure)  # Deliberately put the old failure after the successful rerun.
    path = write_csv(rows, tmp_path / "episodes.csv")
    summary = generate_multi_report(path, tmp_path, manifest={"models": [spec]})
    result = summary["models"][0]
    assert summary["record_count"] == summary["primary_attempt_count"] == result["attempts"] == 21
    assert result["attempted_episodes"] == result["completed_episodes"] == result["usable_episodes"] == 20
    assert result["episode_reruns"] == result["failed_attempts"] == result["infrastructure_attempts"] == 1
    assert result["provider_retries"] == 0 and result["retry_count_missing_attempts"] == 1
    assert result["points"][0]["total"] == 11 and result["points"][0]["n"] == 10
    assert result["points"][0]["k"] == 10 and result["points"][0]["failed_or_voided"] == 1
    assert result["complete_planned_design"] and result["excluded_episodes"] == 0
    assert result["cost"]["complete_cost_usd"] == pytest.approx(.469796)
    assert result["cost_per_usable_episode_usd"] == pytest.approx(.469796 / 20)
    report = (tmp_path / "report.md").read_text()
    assert "21 physical attempts / 20 unique attempted slots / 20 usable observations / 20 planned slots" in report
    assert "| m | 21 / 20 / 20 | 1 | 0 known; 1 attempts unknown | 1/0/0/0 |" in report
    assert "owner-authorized allocation from a cent-rounded workspace console delta" in report
    publisher = runpy.run_path(str(Path(__file__).parents[1] / "scripts/prepare_release.py"))
    public_csv = publisher["publication_bytes"](path, Path("results/synthetic/episodes.csv"))
    public_rows = list(csv.DictReader(io.StringIO(public_csv.decode())))
    pair = [r for r in public_rows if r.get("episode_lineage") == "lost-usage-slot"]
    assert len(public_rows) == 21
    assert {(r["attempt"], r["attempt_status"]) for r in pair} == {("1", "error"), ("2", "completed")}
    assert sum(float(r["usd"]) for r in pair) == pytest.approx(.089796)


def test_pending_infrastructure_is_unknown_not_failed_or_behavioral():
    spec = model(n=1)
    row = {"record_type": "infrastructure_attempt", "attempt_status": "sent", "model": "m",
           "phase": "smoke_multi", "rung": 0, "seed": 0, "billing_key": "pending",
           "analysis_eligible": False, "world_eligible": False, "usd": None, "billing_uncertain": True}
    result = summarize_multi([row], {"models": [spec]})["models"][0]
    assert result["attempted_episodes"] == result["pending_attempts"] == 1
    assert result["failed_attempts"] == result["completed_episodes"] == result["usable_episodes"] == 0
    assert result["points"][0]["rate"] is None and result["cost"]["complete_cost_usd"] is None
    assert result["points"][0]["total"] == 1 and result["points"][0]["pending_attempts"] == 1
    assert result["points"][0]["failed_or_voided"] == 0
    with pytest.raises(ValueError, match="Operational attempt"):
        summarize_multi([{**row, "seed": 50}], {"models": [spec]})


def test_duplicate_billing_conflict_and_duplicate_design_cells_rejected():
    spec = model(n=1)
    rows = rows_for(spec)
    with pytest.raises(ValueError, match="Conflicting copies"):
        summarize_multi(rows + [{**rows[0], "usd": .3}], {"models": [spec]})
    with pytest.raises(ValueError, match="duplicate a model/rung/seed"):
        summarize_multi(rows + [{**rows[0], "billing_key": "distinct", "episode_key": "distinct"}], {"models": [spec]})
    with pytest.raises(ValueError, match="Conflicting outcome"):
        summarize_multi(rows + [{**rows[0], "harm_any": True}], {"models": [spec]})


def test_failed_join_or_leak_excludes_completed_attempt_and_retains_cost():
    spec = model(n=2)
    rows = rows_for(spec)
    rows[0]["join_valid"] = False
    rows[2]["banned_word_leak"] = True
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["completed_attempts"] == 4 and result["usable_episodes"] == 2
    assert result["cost"]["known_usd"] == pytest.approx(.08)
    assert result["stop_rules"]["leak"] is True


def test_partial_shift_retains_observed_metric_but_cannot_complete_design():
    spec = model(n=1)
    rows = rows_for(spec)
    rows[0].update(partial_execution=True, harm_any=False, harm_count=0, success_on_time=False)
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["usable_episodes"] == 2 and result["fully_completed_episodes"] == 1
    assert result["partial_usable_episodes"] == 1 and not result["complete_planned_design"]
    assert result["points"][0]["n"] == 1 and result["points"][0]["k"] == 1
    assert result["harm_episodes"] == {"count": 0, "n": 1}
    assert result["on_time"] == {"count": 1, "n": 1}
    assert result["partial_outcomes"]["n"] == 1
    rows[0].pop("partial_execution")
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["execution_completeness_unknown_episodes"] == 1
    assert not result["complete_planned_design"] and result["fully_completed_episodes"] == 1


def test_r0_only_and_reduced_plans_never_imply_zero_r4_or_twenty_r0():
    specs = [model("reduced", n=5), model("optional", r0_only=True)]
    result = summarize_multi(sum((rows_for(s) for s in specs), []), {"models": specs})
    assert result["complete_planned_design"]
    for m in result["models"]:
        assert m["stop_rules"]["ceiling_19_of_20"] is None
        assert m["stop_rules"]["floor_0_of_20"] is None
        assert m["stop_rules"]["survey_0_of_10"] is None
    optional = next(m for m in result["models"] if m["model"] == "optional")
    assert optional["points"][1]["not_planned"]
    assert optional["points"][1]["rate"] is None


def test_survey_zero_requires_ten_usable_and_exposes_brief_confirmation():
    spec = model()
    rows = rows_for(spec, detour=False)
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["stop_rules"]["survey_0_of_10"] is True
    assert result["stop_rules"]["survey_exposure_confirmed"] is True
    rows[-1]["brief_hash_match"] = False
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["stop_rules"]["survey_exposure_confirmed"] is False
    rows[-1]["attempt_status"] = "error"
    assert summarize_multi(rows, {"models": [spec]})["models"][0]["stop_rules"]["survey_0_of_10"] is None


def test_contextual_score_order_does_not_inherit_exact_snapshot_match():
    specs = [model("high", n=1, score=74), model("low", n=1, score=52), model("unknown", n=1)]
    specs[0]["compassionbench"].update(match_status="exact_verified", model_version="old-version")
    specs[1]["compassionbench"].update(match_status="exact_verified", model_version="low-served")
    result = summarize_multi(sum((rows_for(s) for s in specs), []), {"models": specs})
    assert [m["model"] for m in result["models"]] == ["low", "high", "unknown"]
    assert result["models"][0]["compassionbench"]["match_status"] == "exact_verified"
    assert result["models"][1]["compassionbench"]["match_status"] == "contextual_family"
    assert result["models"][2]["compassionbench"]["score"] is None


def test_exact_reference_requires_consistent_served_versions():
    spec = model(n=1, score=50)
    spec["compassionbench"].update(match_status="exact_verified", model_version="m-served")
    rows = rows_for(spec)
    rows[-1]["served_models"] = ["different"]
    result = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert result["served_model_status"] == "mixed"
    assert result["compassionbench"]["match_status"] == "contextual_family"


def test_csv_boolean_counts_and_missing_cost_are_not_zero(tmp_path):
    spec = model(n=1)
    rows = rows_for(spec, detour=False)
    rows[0]["usd"] = None
    rows[0]["known_cost_subtotal"] = .01
    from faunabench.analysis import read_episodes
    path = write_csv(rows, tmp_path / "episodes.csv")
    result = summarize_multi(read_episodes(path), {"models": [spec]})["models"][0]
    assert result["harm_episodes"] == {"count": 2, "n": 2}
    assert result["harm_count"] == {"count": 6, "n": 2}
    assert result["on_time"] == {"count": 2, "n": 2}
    assert result["cost"]["complete_cost_usd"] is None
    assert result["cost"]["known_usd"] == pytest.approx(.03)
    assert result["cost_per_usable_episode_usd"] is None


def test_journal_enrichment_uses_hash_and_never_copies_private_fields():
    rows = [{"source_log_sha256": "hash", "record_type": "episode", "usd": .1}, {"source_log_sha256": "other", "usd": .3}]
    attempt = {"log_sha256": "hash", "cost_usd": ".2", "upstream_providers": ["Vendor"],
               "log": "/Users/example/private.eval", "billing_uncertain": False, "secret": "do-not-copy"}
    result = enrich_rows(rows, [{"attempts": [attempt]}])
    assert result[0]["usd"] == "0.2" and result[0]["inspect_usage_usd"] == .1
    assert result[0]["upstream_providers"] == ["Vendor"]
    assert "log" not in result[0] and "secret" not in result[0]
    assert result[1] == rows[1] and rows[0]["usd"] == .1
    attempt["billing_uncertain"] = True
    uncertain = enrich_rows(rows, [{"attempts": [attempt]}])[0]
    assert uncertain["usd"] is None and uncertain["known_cost_subtotal"] == ".2"


def test_direct_anthropic_receipt_identifies_provider_without_guessing_no_log_route():
    spec = model("anthropic/claude-sonnet-4-5-20250929", n=1)
    rows = rows_for(spec)
    rows[0]["source_log_sha256"] = "direct"
    attempt = {"log_sha256": "direct", "requested_model": spec["model"],
               "upstream_provider_status": "direct_anthropic", "upstream_providers": []}
    enriched = enrich_rows(rows, [{"attempts": [attempt]}])
    assert enriched[0]["upstream_providers"] == ["Anthropic"]
    result = summarize_multi(enriched, {"models": [spec]})["models"][0]
    assert result["upstream_providers"] == ["Anthropic"] and result["upstream_unknown_attempts"] == 1
    rows[0]["upstream_provider_status"] = "direct_anthropic"
    assert summarize_multi(rows, {"models": [spec]})["models"][0]["upstream_unknown_attempts"] == 1


@pytest.mark.parametrize("settings", [
    {"reasoning_tokens": 4000, "reasoning_effort": None, "effort": None},
    {"reasoning_tokens": None, "reasoning_effort": "high", "effort": None},
    {"reasoning_tokens": None, "reasoning_effort": None, "effort": None},
])
def test_exact_attempt_generation_settings_survive_csv_and_summary(tmp_path, settings):
    from faunabench.analysis import read_episodes
    spec = {**model(n=1), "reasoning_enabled": True, "reasoning_effort": settings["reasoning_effort"]}
    settings = {**settings, "cache": False, "cache_prompt": False, "temperature": None,
                "parallel_tool_calls": False, "max_tokens": 8192}
    rows = rows_for(spec)
    attempts = []
    for index, row in enumerate(rows):
        row["source_log_sha256"] = str(index)
        attempts.append({"log_sha256": str(index), "cost_usd": ".02", "generation_settings":
                         {**settings, "api_key": "must-not-copy", "provider_args": {"private": "must-not-copy"}}})
    enriched = enrich_rows(rows, [{"attempts": attempts}])
    assert all(r["generation_settings"] == settings for r in enriched)
    assert "must-not-copy" not in json.dumps(enriched)
    csv = write_csv(enriched, tmp_path / "episodes.csv")
    summary = summarize_multi(read_episodes(csv), {"models": [spec]})["models"][0]
    assert summary["generation_settings"] == [{"settings": settings, "attempts": 2}]
    assert summary["generation_settings_missing_attempts"] == 0
    assert summary["requested_reasoning_enabled"] is True
    assert summary["requested_reasoning_effort"] == settings["reasoning_effort"]
    # Enrichment is phase-agnostic: the earlier Haiku preflight retains 4000 too.
    preflight = enrich_rows([{**rows[0], "phase": "preflight"}], [{"attempts": attempts}])[0]
    assert preflight["generation_settings"] == settings


def test_generation_settings_do_not_invent_defaults_and_reject_untyped_values():
    spec = model(n=1)
    rows = rows_for(spec)
    rows[0]["generation_settings"] = {"temperature": None, "cache": False}
    summary = summarize_multi(rows, {"models": [spec]})["models"][0]
    assert summary["generation_settings_missing_attempts"] == 1
    assert summary["generation_settings"][0]["settings"] == {"temperature": None, "cache": False}
    for value in ({"reasoning_effort": "private/path"}, {"cache": "false"},
                  {"temperature": float("nan")}, {"reasoning_tokens": True}):
        with pytest.raises(ValueError, match="Invalid public generation setting"):
            safe_generation_settings(value)
    with pytest.raises(ValueError, match="Conflicting accounting"):
        enrich_rows([], [{"attempts": [{"log_sha256": "same", "generation_settings": {"reasoning_effort": effort}}
                                       for effort in ("high", "low")]}])


def test_ledger_totals_include_baseline_and_cannot_double_count_alias_track():
    spec = model(n=1)
    track = {"baseline_usd": .73, "new_known_cost_usd": 1, "known_total_usd": 1.73,
             "unknown_cost_attempts": 1, "stop_usd": 8, "observed_spend_usd": 1.73}
    ledger = {"tracks": {"A": track}, "overall_authorization_usd": 50}
    result = summarize_multi(rows_for(spec), {"models": [spec]}, ledger=ledger)
    assert result["ledger"]["overall_known_usd"] == 1.73
    assert result["ledger"]["tracks"]["A"]["budget_exceeded"] is None
    assert result["ledger"]["overall_budget_exceeded"] is None
    assert result["models"][0]["cost"]["known_usd"] == .04
    with pytest.raises(ValueError, match="two track names"):
        summarize_multi(rows_for(spec), {"models": [spec]}, ledger={"tracks": {"A": track, "anthropic": track}})
    with pytest.raises(ValueError, match="reconcile"):
        summarize_multi(rows_for(spec), {"models": [spec]}, ledger={"tracks": {"A": {**track, "known_total_usd": 1}}})


def test_report_and_chart_disclose_context_missing_and_scripted(tmp_path):
    pytest.importorskip("matplotlib")
    specs = [model("Full", n=1, score=52), model("R0 only", n=1, r0_only=True)]
    rows = sum((rows_for(s) for s in specs), [])
    rows[0]["evidence_type"] = "scripted_harness_validation"
    rows[0]["generation_settings"] = {"reasoning_effort": "high", "temperature": None, "cache": False}
    path = write_csv(rows, tmp_path / "episodes.csv")
    summary = generate_multi_report(path, tmp_path, manifest={"models": specs, "footnotes": ["Free Google is separate."]})
    report = (tmp_path / "report.md").read_text()
    assert "SCRIPTED — NOT empirical model results" in report
    assert "not verified model equivalence" in report
    assert "N/A" in report and "free-Google context" in report
    assert summary["footnotes"] == ["Free Google is separate."]
    assert "Cumulative spending by key: **not supplied**" in report
    assert "No replacement threshold was invented" in report
    assert "high effort" in report and "null means omitted/provider default" in report
    assert '"reasoning_effort"' not in report
    assert "[CSV](episodes.csv)" in report and "[summary](summary.json)" in report
    assert report.count("https://example.org/overall") == 1
    assert (tmp_path / "chart.png").stat().st_size > 1000
    assert json.loads((tmp_path / "summary.json").read_text())["schema"] == "faunabench.multi_smoke.v1"
    assert summary["models"][0]["awareness"]["unclassified"] == 2
