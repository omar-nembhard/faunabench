"""Offline injected executors test admission and recovery, never real providers."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from faunabench.budget import BudgetExceeded
from faunabench.planning import EpisodePlan, smoke_plan
from faunabench.runner import (DispatchUncertain, ExecutionResult, ManifestError, RequestLimits,
                              SerialRunner, build_manifest, reservation_microusd)

MODELS = ["mockllm/model-a", "mockllm/model-b"]
PRICES = {model: {"input_usd_per_million": "0.30", "output_usd_per_million": "2.00",
                  "source": "local-test-fixture-not-real-prices", "verified_at": "2026-11-07"} for model in MODELS}
APPROVAL = {"approved": True, "confirmed_by": "test fixture", "confirmation_reference": "offline-unit-test-only"}
OK = ExecutionResult(1000, 100, 10, 1, 4)


def fixture_manifest(**kwargs):
    return build_manifest(smoke_plan(MODELS, phase="preflight") + smoke_plan(MODELS), PRICES,
                          approval=APPROVAL, **kwargs)


def first(manifest, phase="preflight", model=MODELS[0]):
    return next(record["episode_id"] for record in manifest["episodes"] if record["phase"] == phase and record["model"] == model)


def test_plan_has_separate_two_preflight_and_forty_smoke():
    manifest = fixture_manifest()
    assert len(manifest["episodes"]) == 42
    assert sum(record["phase"] == "preflight" for record in manifest["episodes"]) == 2
    assert len({record["episode_id"] for record in manifest["episodes"]}) == 42


def test_unapproved_manifest_and_unverified_prices_fail(tmp_path):
    manifest = build_manifest(smoke_plan(MODELS), PRICES)
    with pytest.raises(ManifestError, match="confirmation"):
        SerialRunner(manifest, tmp_path / "budget.json")
    bad_prices = deepcopy(PRICES)
    bad_prices[MODELS[0]]["verified_at"] = ""
    with pytest.raises(ManifestError, match="dated"):
        build_manifest(smoke_plan(MODELS), bad_prices)


def test_reservation_uses_hard_bounds_and_all_retries_not_soft_cost_cap():
    limits = RequestLimits()
    bound = reservation_microusd(PRICES[MODELS[0]], limits)
    assert bound == 2005402
    assert bound > 500000
    assert reservation_microusd(PRICES[MODELS[0]], replace(limits, sample_cost_limit_usd="0.01")) == bound
    assert reservation_microusd(PRICES[MODELS[0]], replace(limits, max_retries=0)) == 668468


def test_reserve_and_sent_are_durable_before_executor_and_idempotent(tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / "budget.json"
    runner = SerialRunner(manifest, path)
    calls = []

    def execute(plan, attempt_id, limits):
        persisted = json.loads(path.read_text())
        assert persisted["budget"]["requests"][attempt_id]["state"] == "sent"
        assert persisted["budget"]["reserved_microusd"] > 0
        calls.append(attempt_id)
        return OK

    episode = first(manifest)
    result = runner.run_one(episode, execute)
    assert result["status"] == "completed"
    assert SerialRunner(manifest, path).run_one(episode, execute) == result
    assert len(calls) == 1
    assert runner.snapshot()["budget"]["spent_microusd"] == 1000
    assert runner.snapshot()["budget"]["reserved_microusd"] == 0


def test_smoke_requires_its_models_preflight(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    with pytest.raises(ValueError, match="preflight"):
        runner.run_one(first(manifest, "smoke"), lambda *_: OK)
    runner.run_one(first(manifest), lambda *_: OK)
    assert runner.run_one(first(manifest, "smoke"), lambda *_: OK)["status"] == "completed"
    with pytest.raises(ValueError, match="preflight"):
        runner.run_one(first(manifest, "smoke", MODELS[1]), lambda *_: OK)


@pytest.mark.parametrize("raises", [False, True])
def test_unknown_usage_holds_reservation_and_blocks_all_dispatch(tmp_path, raises):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")

    def fail(*_):
        if raises:
            raise RuntimeError("simulated provider disconnect")
        return replace(OK, actual_microusd=None)

    with pytest.raises(DispatchUncertain):
        runner.run_one(first(manifest), fail)
    snapshot = SerialRunner(manifest, runner.journal).snapshot()
    assert snapshot["budget"]["spent_microusd"] == 0
    assert snapshot["budget"]["reserved_microusd"] > 0
    for episode, retry in [(first(manifest), False), (first(manifest), True), (first(manifest, model=MODELS[1]), False)]:
        with pytest.raises(DispatchUncertain):
            runner.run_one(episode, lambda *_: pytest.fail("must not dispatch"), retry=retry)


def test_reconciliation_is_idempotent_and_known_failure_retry_is_explicit(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    episode = first(manifest)
    with pytest.raises(DispatchUncertain):
        runner.run_one(episode, lambda *_: replace(OK, input_tokens=None))
    attempt_id = f"{episode}:attempt-1"
    failed = replace(OK, status="error")
    result = runner.reconcile(attempt_id, failed)
    assert runner.reconcile(attempt_id, failed) == result
    assert runner.snapshot()["budget"]["spent_microusd"] == 1000
    with pytest.raises(ValueError, match="conflicting"):
        runner.reconcile(attempt_id, replace(failed, actual_microusd=2000))
    assert runner.run_one(episode, lambda *_: pytest.fail("not an automatic retry"))["status"] == "error"
    second = runner.run_one(episode, lambda *_: OK, retry=True)
    assert second["attempt_id"].endswith("attempt-2")
    assert runner.snapshot()["budget"]["spent_microusd"] == 2000


def test_resume_reserved_never_sent_is_safe(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    episode = first(manifest)
    attempt_id = f"{episode}:attempt-1"
    # Crash between the first durable save and mark_sent: this reservation is unsent.
    with runner._locked() as (data, ledger):
        bound = reservation_microusd(PRICES[MODELS[0]], runner.limits)
        ledger.reserve(attempt_id, bound)
        data["attempts"][attempt_id] = {"episode_id": episode, "attempt_id": attempt_id,
                                       "phase": "preflight", "model": MODELS[0], "status": "reserved"}
        runner._save(data, ledger)
    assert SerialRunner(manifest, runner.journal).run_one(episode, lambda *_: OK)["attempt_id"] == attempt_id
    assert runner.snapshot()["budget"]["spent_microusd"] == 1000


def test_any_manifest_or_price_change_rejects_existing_journal(tmp_path):
    manifest = fixture_manifest()
    path = tmp_path / "budget.json"
    SerialRunner(manifest, path).run_one(first(manifest), lambda *_: OK)
    changed = deepcopy(manifest)
    changed["prices"][MODELS[0]]["output_usd_per_million"] = "3.00"
    with pytest.raises(ManifestError, match="different manifest"):
        SerialRunner(changed, path).run_one(first(manifest), lambda *_: pytest.fail("must not dispatch"))


def test_overbudget_rejects_before_call_and_actual_bound_breach_freezes(tmp_path):
    manifest = fixture_manifest()
    manifest["budget_limit_microusd"] = 500000
    runner = SerialRunner(manifest, tmp_path / "tiny.json")
    with pytest.raises(BudgetExceeded):
        runner.run_one(first(manifest), lambda *_: pytest.fail("must not dispatch"))
    normal = fixture_manifest()
    runner = SerialRunner(normal, tmp_path / "breach.json")
    bound = reservation_microusd(PRICES[MODELS[0]], runner.limits)
    result = runner.run_one(first(normal), lambda *_: replace(OK, actual_microusd=bound + 1))
    assert result["status"] == "bound_breach"
    assert runner.snapshot()["budget"]["frozen"]
    with pytest.raises(BudgetExceeded):
        runner.run_one(first(normal, model=MODELS[1]), lambda *_: pytest.fail("must not dispatch"))


def test_no_actions_is_execution_failure_not_behavior(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    result = runner.run_one(first(manifest), lambda *_: replace(OK, action_count=0))
    assert result["status"] == "failed_no_actions"
    with pytest.raises(ValueError, match="preflight"):
        runner.run_one(first(manifest, "smoke"), lambda *_: OK)


def test_auditor_and_retry_share_same_total_ledger(tmp_path):
    plans = smoke_plan(MODELS, phase="preflight") + [EpisodePlan(MODELS[0], 0, 99, phase="auditor")]
    manifest = build_manifest(plans, PRICES, approval=APPROVAL)
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    runner.run_one(plans[0].episode_id, lambda *_: replace(OK, status="error"))
    runner.run_one(plans[0].episode_id, lambda *_: OK, retry=True)
    runner.run_one(plans[-1].episode_id, lambda *_: replace(OK, action_count=0))
    assert runner.snapshot()["budget"]["spent_microusd"] == 3000


def test_unfrozen_pilot_and_duplicate_ids_cannot_dispatch():
    with pytest.raises(ManifestError, match="pilot"):
        build_manifest([EpisodePlan(MODELS[0], 0, 0, phase="pilot")], PRICES)
    repeated = smoke_plan(MODELS, phase="preflight")[:1] * 2
    with pytest.raises(ManifestError, match="duplicate"):
        build_manifest(repeated, PRICES)


def test_concurrent_invocations_dispatch_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import time
    manifest = fixture_manifest()
    path = tmp_path / "budget.json"
    calls = []

    def execute(*_):
        calls.append(1)
        time.sleep(0.01)
        return OK

    def invoke(_):
        return SerialRunner(manifest, path).run_one(first(manifest), execute)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert len(calls) == 1
    assert all(result == results[0] for result in results)
    assert SerialRunner(manifest, path).snapshot()["budget"]["spent_microusd"] == 1000


def test_usage_bound_violation_freezes_even_when_cost_is_known(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    too_many = runner.limits.max_output_tokens_per_call * runner.limits.billed_request_bound + 1
    result = runner.run_one(first(manifest), lambda *_: replace(OK, output_tokens=too_many))
    assert result["status"] == "bound_breach"
    assert runner.snapshot()["budget"]["frozen"]


def test_post_initialization_manifest_mutation_cannot_change_reservation(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    runner.manifest["prices"][MODELS[0]]["input_usd_per_million"] = "0.000001"
    with pytest.raises(ManifestError, match="changed after"):
        runner.run_one(first(manifest), lambda *_: pytest.fail("must not dispatch"))
    assert not runner.journal.exists()


def test_provider_retries_do_not_expand_once_only_episode_rerun(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    episode = first(manifest)
    failed = replace(OK, status="error")
    runner.run_one(episode, lambda *_: failed)
    runner.run_one(episode, lambda *_: failed, retry=True)
    with pytest.raises(ValueError, match="retry limit"):
        runner.run_one(episode, lambda *_: pytest.fail("must not dispatch"), retry=True)
    assert runner.snapshot()["budget"]["spent_microusd"] == 2000


def test_append_only_approved_extension_preserves_ledger_and_rejects_stale_runner(tmp_path):
    from faunabench.runner import extend_manifest
    manifest = fixture_manifest()
    path = tmp_path / "budget.json"
    runner = SerialRunner(manifest, path)
    stale = SerialRunner(manifest, path)
    runner.run_one(first(manifest), lambda *_: OK)
    added = smoke_plan(MODELS, phase="floor_extension") + [EpisodePlan(MODELS[0], 0, 99, phase="auditor")]
    expanded = extend_manifest(manifest, added, approval={**APPROVAL, "confirmation_reference": "explicit-extension-fixture"}, reason="Observed floor rule and planned audit")
    before = runner.snapshot()
    runner.approve_expansion(expanded)
    after = runner.snapshot()
    assert after["ledger_events"] == before["ledger_events"]
    assert after["attempts"] == before["attempts"]
    assert after["budget"]["spent_microusd"] == 1000
    assert after["manifest_history"][0]["parent_hash"] == before["manifest_hash"]
    with pytest.raises(ManifestError, match="different manifest"):
        stale.run_one(first(manifest), lambda *_: pytest.fail("stale instance must not dispatch"))
    resumed = SerialRunner(expanded, path)
    resumed.run_one(added[0].episode_id, lambda *_: OK)
    resumed.run_one(added[-1].episode_id, lambda *_: replace(OK, action_count=0))
    assert resumed.snapshot()["budget"]["spent_microusd"] == 3000


@pytest.mark.parametrize("change", ["prices", "limits", "budget", "old_episode", "missing_reason"])
def test_expansion_rejects_changed_approved_baseline(tmp_path, change):
    from faunabench.runner import extend_manifest
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    expanded = extend_manifest(manifest, smoke_plan(MODELS, phase="floor_extension"),
                               approval={**APPROVAL, "confirmation_reference": "fresh-extension"}, reason="Conditional floor-rule follow-up")
    if change == "prices":
        expanded["prices"][MODELS[0]]["output_usd_per_million"] = "3.00"
    elif change == "limits":
        expanded["limits"]["max_output_tokens_per_call"] += 1
    elif change == "budget":
        expanded["budget_limit_microusd"] -= 1
    elif change == "old_episode":
        expanded["episodes"][0] = EpisodePlan(MODELS[0], 0, 7, phase="preflight").record()
    else:
        expanded["expansion_reason"] = ""
    with pytest.raises(ManifestError):
        runner.approve_expansion(expanded)
    assert not runner.journal.exists()


def test_limited_retry_can_raise_soft_cap_without_lowering_reservation(tmp_path):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    episode = first(manifest)
    first_attempt = runner.run_one(episode, lambda *_: replace(OK, status="limited"))
    seen = []

    def retry_executor(plan, attempt_id, limits):
        seen.append(limits.sample_cost_limit_usd)
        assert limits.max_output_tokens_per_call == 4096
        return OK

    retry_attempt = runner.run_one(episode, retry_executor, retry=True, cost_limit_override_usd="1.00")
    assert seen == ["1.00"]
    assert retry_attempt["upper_bound_microusd"] == first_attempt["upper_bound_microusd"]
    assert retry_attempt["limits"]["sample_cost_limit_usd"] == "1.00"
    assert runner.snapshot()["budget"]["spent_microusd"] == 2000


@pytest.mark.parametrize("status,override", [("error", "1.00"), ("limited", "1.01"), ("limited", "0.50")])
def test_cost_cap_override_requires_limited_known_retry_within_approved_max(tmp_path, status, override):
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    episode = first(manifest)
    with pytest.raises(ValueError, match="reconciled limited"):
        runner.run_one(episode, lambda *_: pytest.fail("no initial override"), cost_limit_override_usd="1.00")
    runner.run_one(episode, lambda *_: replace(OK, status=status))
    with pytest.raises(ValueError):
        runner.run_one(episode, lambda *_: pytest.fail("invalid override"), retry=True, cost_limit_override_usd=override)
    assert runner.snapshot()["budget"]["spent_microusd"] == 1000


def test_authorized_floor_extension_reuses_original_authority_without_microapproval(tmp_path):
    from faunabench.runner import extend_manifest
    manifest = fixture_manifest()
    runner = SerialRunner(manifest, tmp_path / "budget.json")
    expanded = extend_manifest(manifest, smoke_plan(MODELS, phase="floor_extension"),
                               approval=APPROVAL, reason="Original run authority covers triggered R3 extension")
    runner.approve_expansion(expanded)
    history = runner.snapshot()["manifest_history"][0]
    assert history["approval"]["confirmation_reference"] == APPROVAL["confirmation_reference"]
    assert history["reason"]


def test_cache_writes_need_verified_maximum_billable_input_rate():
    limits = replace(RequestLimits(), cache_write_enabled=True)
    with pytest.raises(ManifestError, match="cache-write"):
        reservation_microusd(PRICES[MODELS[0]], limits)
    prices = {**PRICES[MODELS[0]], "max_input_usd_per_million": "1.25"}
    assert reservation_microusd(prices, limits) == 6488064
    assert RequestLimits().cache_write_enabled is False
