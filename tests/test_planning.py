from collections import Counter

import pytest

from faunabench.planning import pilot_plan, smoke_plan


def test_smoke_balance_and_phase_identity():
    models = ["mockllm/a", "mockllm/b"]
    episodes = smoke_plan(models)
    assert len(episodes) == len({e.episode_id for e in episodes}) == 40
    for model in models:
        for rung in (0, 4):
            assert Counter(e.seed % 3 for e in episodes if e.model == model and e.rung == rung) == {0: 4, 1: 3, 2: 3}
    assert not ({e.episode_id for e in episodes} & {e.episode_id for e in smoke_plan(models, phase="preflight")})
    assert len(smoke_plan(models, phase="floor_extension")) == 20


def test_pilot_baseline_and_paraphrase_split():
    plan = pilot_plan([f"provider/model{i}" for i in range(6)])
    assert plan["core_episodes"] == 1080
    assert plan["lb_episodes"] == 240
    es = plan["episodes"]
    assert len(es) == len({e["episode_id"] for e in es}) == 1320
    assert Counter(e["paraphrase"] for e in es if e["model"] == "provider/model0" and e["phase"] == "pilot_core" and e["rung"] == 0) == {"original": 15, "alternative_UNFROZEN": 15}
    assert all(e["layout"] != "LC" and not e["told_real"] for e in es)


def test_cheap_pilot_does_not_silently_expand_beyond_capacity():
    with pytest.raises(ValueError, match="capacity"):
        pilot_plan([f"p/m{i}" for i in range(6)], "0.03")


@pytest.mark.parametrize("cost", ["0", "-1", "NaN", "Infinity", "100"])
def test_invalid_or_unaffordable_cost_rejected(cost):
    with pytest.raises(ValueError):
        pilot_plan([f"p/m{i}" for i in range(6)], cost)
