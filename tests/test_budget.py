import concurrent.futures

import pytest

from faunabench.budget import BudgetExceeded, CostLedger


def test_concurrent_reservations_never_over_admit():
    ledger = CostLedger(10_000)
    def reserve(index):
        try:
            ledger.reserve(str(index), 2_000)
            return True
        except BudgetExceeded:
            return False
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        accepted = list(pool.map(reserve, range(10)))
    assert sum(accepted) == 5
    assert ledger.snapshot()["remaining_microusd"] == 0


def test_retry_and_reconciliation_are_idempotent():
    ledger = CostLedger(100)
    ledger.reserve("a", 60)
    ledger.reserve("a", 60)
    ledger.mark_sent("a")
    ledger.mark_sent("a")
    ledger.settle("a", 40)
    ledger.settle("a", 40)
    assert ledger.snapshot()["spent_microusd"] == 40
    assert ledger.snapshot()["reserved_microusd"] == 0
    assert len(ledger.events) == 3
    with pytest.raises(ValueError):
        ledger.settle("a", 41)


def test_unknown_usage_after_dispatch_remains_reserved():
    ledger = CostLedger(100)
    ledger.reserve("a", 80)
    ledger.mark_sent("a")
    with pytest.raises(ValueError, match="never-sent"):
        ledger.release_unsent("a")
    with pytest.raises(BudgetExceeded):
        ledger.reserve("retry", 80)
    assert ledger.snapshot()["reserved_microusd"] == 80


def test_unsent_cancel_releases_only_that_reservation():
    ledger = CostLedger(100)
    ledger.reserve("a", 80)
    ledger.release_unsent("a")
    ledger.release_unsent("a")
    ledger.reserve("b", 100)
    assert ledger.snapshot()["reserved_microusd"] == 100
    with pytest.raises(ValueError, match="conflict"):
        ledger.reserve("a", 80)


def test_underestimated_bill_is_recorded_and_freezes_admission():
    ledger = CostLedger(100)
    ledger.reserve("a", 80)
    ledger.mark_sent("a")
    ledger.settle("a", 110)
    assert ledger.snapshot()["spent_microusd"] == 110
    assert ledger.snapshot()["remaining_microusd"] == -10
    assert ledger.events[-1]["bound_breached"]
    with pytest.raises(BudgetExceeded):
        ledger.reserve("b", 1)


@pytest.mark.parametrize("amount", [True, -1, 1.2])
def test_invalid_amounts_cannot_change_state(amount):
    ledger = CostLedger(100)
    with pytest.raises(ValueError):
        ledger.reserve("bad", amount)
    assert not ledger.events
