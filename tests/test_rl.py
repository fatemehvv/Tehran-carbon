import copy

import numpy as np
import pytest

from tehran_carbon.config import Config
from tehran_carbon.rl.city import CityTransport
from tehran_carbon.rl.env import CarbonPolicyEnv
from tehran_carbon.rl.policies import StaticPolicy

RL = Config().resolve("rq3_policy_v1")[0]["rl"]
D = 3


def tiny_city(congested_kmh: float = 40.0) -> CityTransport:
    """3 districts, free flow 50 km/h, mild congestion; car faster than transit."""
    C = np.array([[1000.0, 400.0, 200.0], [300.0, 800.0, 100.0], [200.0, 100.0, 600.0]])
    L = np.array([[2.0, 8.0, 12.0], [8.0, 2.5, 6.0], [12.0, 6.0, 3.0]])
    days = np.array([31.0] * 6 + [30.0] * 5 + [29.0])
    share = np.full(12, 1 / 12)
    road = np.array([[5000.0, 3000.0, 2000.0]] * 12)
    return CityTransport([f"D{i}" for i in range(D)], C, L, share * days.sum(), days, np.linspace(0, 400, 12),
                         ef=2e-4, bus_co2=1200.0, bus_mj=1.6e7, bus_share=[0.5, 0.3, 0.2],
                         bus_riders=[3000.0, 2000.0, 1000.0], road_co2=road, road_mj=np.full(12, 1e8), speed=25.0,
                         h_free=np.full(D, 1 / 50), h0=np.full(D, 1 / congested_kmh))


def env(congested_kmh: float = 40.0, **reward) -> CarbonPolicyEnv:
    r = copy.deepcopy(RL)
    r["reward"].update(reward)
    e = CarbonPolicyEnv(tiny_city(congested_kmh), r)
    e.reset(seed=0)
    return e


def test_no_action_changes_nothing():
    e = env()
    out = e.outcome(0, np.zeros(D), np.zeros(D))
    assert out["reward"] == pytest.approx(0.0, abs=1e-12)
    assert out["co2_saved_t"] == pytest.approx(0.0, abs=1e-9) and out["hours_lost"] == pytest.approx(0.0, abs=1e-6)


def test_restriction_cuts_co2_and_costs_time():
    e = env()
    out = e.outcome(0, np.array([0.3, 0.0, 0.0]), np.zeros(D))
    assert out["co2_saved_t"] > 0 and out["hours_lost"] > 0
    assert 0 < out["shift_share"] <= 1 and out["energy_MJ"] == 0
    # a larger travel-time weight lowers the reward of the same action
    assert env(lambda_mobility=4.0).outcome(0, np.array([0.3, 0, 0]), np.zeros(D))["reward"] < out["reward"]


def test_a_trip_not_made_is_a_loss():
    e = env(congested_kmh=50.0)                                        # no congestion, so no time saved
    e.theta.update(shift=0.0, shift_boost=0.0)                         # every removed trip is suppressed
    out = e.outcome(0, np.array([0.3, 0.0, 0.0]), np.zeros(D))
    removed_hours = 0.3 * (e.city.C[:, 0] * e.noise[0] * e.city.L[:, 0]).sum() * e.city.m[0] * RL["occupancy"] / e.city.speed
    assert out["hours_lost"] == pytest.approx(removed_hours, rel=1e-9)


def test_bus_boost_costs_energy_and_respects_the_budget():
    e = env()
    used, total_energy = 0.0, 0.0
    for _ in range(12):
        _, _, done, _, out = e.step(np.ones(2 * D, np.float32))
        used += out["bus_used"]
        total_energy += out["energy_MJ"]
    assert done and total_energy > 0
    assert used == pytest.approx(RL["bus_budget"], rel=1e-6)          # max boost every month exhausts the budget


def test_delay_elasticity():
    c = tiny_city(25.0)                                                # delay = half of the time per km
    assert np.allclose(c.time_ratio(np.ones(D), 1.7), 1.0)
    assert np.allclose(c.time_ratio(np.full(D, 0.8), 1.0), 0.9)        # -20 % volume, eta 1 -> -20 % delay


def test_common_random_numbers():
    a, b = env(), env()
    a.reset(seed=7)
    b.reset(seed=7)
    assert a.theta == b.theta and np.array_equal(a.noise, b.noise)


def test_static_policies():
    policies = StaticPolicy.grid(RL["static"])
    assert len(policies) == 1 + 9 + 9 + 18 and len({p.name for p in policies}) == len(policies)
    e = env()
    hot = StaticPolicy("hotspot", 0.2, 0.5, districts=1)
    r, b = e.levers(hot.act(None, e))
    assert (r > 0).sum() == 1 and r[e.hot_rank[0]] == pytest.approx(0.2)
    winter = StaticPolicy("winter", 0.2, 0.0, months=[8])
    assert np.allclose(e.levers(winter.act(None, e))[0], 0.0)           # month 1: off


def test_gymnasium_api():
    from gymnasium.utils.env_checker import check_env
    check_env(env(), skip_render_check=True)


def test_daily_city_keeps_the_annual_totals():
    c = tiny_city()
    d = c.daily()
    assert d.T == int(c.days.sum()) and d.month[0] == 1 and d.month[-1] == 12
    assert np.isclose(d.m.sum(), c.m.sum()) and np.isclose(d.road_co2.sum(), c.road_co2.sum())


def test_daily_restriction_budget_weather_and_calm_rule():
    r = copy.deepcopy(RL)
    r.update(restriction_budget_days=5.0, bus_budget=0.0)
    city = tiny_city().daily()
    weather = np.zeros((city.T, 1))
    weather[::2, 0] = -2.0                                              # every second day is calm
    e = CarbonPolicyEnv(city, r, None, weather, np.ones(city.T))
    obs, _ = e.reset(seed=0)
    assert obs.shape == (8 + 2 * D + 2,)                                # + weather + remaining budget
    used = 0.0
    for _ in range(city.T):
        _, _, done, _, out = e.step(np.ones(2 * D, np.float32))
        used += out["restriction"].mean() / r["levers"]["restriction_max"]
        assert np.all(out["bus_boost"] == 0)
    assert done and used == pytest.approx(5.0, rel=1e-6)
    calm = StaticPolicy("calm", 0.3, 0.0, feature=0, below=-1.0)
    e.reset(seed=0)
    assert e.levers(calm.act(None, e))[0].max() > 0                    # day 0 is calm
    e.t = 1
    assert e.levers(calm.act(None, e))[0].max() == 0                   # day 1 is not
