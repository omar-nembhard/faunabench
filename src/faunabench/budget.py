"""Deterministic local admission accounting, not a provider billing guarantee.

Amounts use integer micro-USD. A future provider runner must obtain a genuine
upper bound, reserve before dispatch, then reconcile actual billed usage. This
prototype has no provider client and does not enforce an external account limit.
"""

from __future__ import annotations

from copy import deepcopy
import threading


class BudgetExceeded(ValueError):
    pass


class CostLedger:
    def __init__(self, limit_microusd: int):
        self._amount(limit_microusd, positive=True)
        self.limit = limit_microusd
        self.spent = 0
        self.frozen = False
        self.requests: dict[str, dict] = {}
        self.events: list[dict] = []
        self._lock = threading.RLock()

    @staticmethod
    def _amount(value: int, positive: bool = False) -> None:
        if type(value) is not int or value < (1 if positive else 0):
            raise ValueError("amount must be integer micro-USD within supported range")

    def snapshot(self) -> dict:
        with self._lock:
            held = sum(r["upper_bound"] for r in self.requests.values() if r["state"] in ("reserved", "sent"))
            return {"limit_microusd": self.limit, "spent_microusd": self.spent,
                    "reserved_microusd": held, "remaining_microusd": self.limit - self.spent - held,
                    "frozen": self.frozen, "requests": deepcopy(self.requests)}

    def reserve(self, request_id: str, upper_bound_microusd: int) -> None:
        with self._lock:
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a nonempty string")
            self._amount(upper_bound_microusd, positive=True)
            previous = self.requests.get(request_id)
            if previous:
                if previous["upper_bound"] != upper_bound_microusd or previous["state"] == "cancelled":
                    raise ValueError("request_id conflict")
                return
            if self.frozen or upper_bound_microusd > self.snapshot()["remaining_microusd"]:
                raise BudgetExceeded("request would exceed available budget or ledger is frozen")
            self.requests[request_id] = {"upper_bound": upper_bound_microusd, "state": "reserved"}
            self.events.append({"action": "reserve", "request_id": request_id, "amount": upper_bound_microusd})

    def mark_sent(self, request_id: str) -> None:
        with self._lock:
            request = self.requests[request_id]
            if request["state"] == "sent":
                return
            if request["state"] != "reserved" or self.frozen:
                raise ValueError("request cannot be dispatched")
            request["state"] = "sent"
            self.events.append({"action": "sent", "request_id": request_id})

    def release_unsent(self, request_id: str) -> None:
        with self._lock:
            request = self.requests[request_id]
            if request["state"] == "cancelled":
                return
            if request["state"] != "reserved":
                raise ValueError("only a never-sent request can release its reservation")
            request["state"] = "cancelled"
            self.events.append({"action": "release_unsent", "request_id": request_id})

    def settle(self, request_id: str, actual_microusd: int) -> None:
        with self._lock:
            self._amount(actual_microusd)
            request = self.requests[request_id]
            if request["state"] == "settled":
                if request["actual"] != actual_microusd:
                    raise ValueError("conflicting reconciliation")
                return
            if request["state"] != "sent":
                raise ValueError("only sent requests can be reconciled")
            self.spent += actual_microusd
            breached_bound = actual_microusd > request["upper_bound"]
            self.frozen = self.frozen or breached_bound or self.spent > self.limit
            request.update(state="settled", actual=actual_microusd)
            self.events.append({"action": "settle", "request_id": request_id,
                                "actual": actual_microusd, "bound_breached": breached_bound})
