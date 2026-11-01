import pytest
import yaml

from tehran_carbon.config import Config
from tehran_carbon.tracking import ConfigDriftError, Run, RunRegistry

BASE = {
    "defaults": {"seed": 0, "paths": {"raw": "raw", "interim": "i", "processed": "p", "runs": None},
                 "grid": {"H": 2}, "years": [1402], "data": {"x": 1},
                 "model": {"fusion": {"type": "gated"}, "modalities": ["traffic", "energy"]},
                 "train": {"lr": 0.1}},
    "experiments": {
        "a_v1": {"rq": "RQ1", "model": {"fusion": {"type": "early"}}},
        "b_v1": {"inherits": "a_v1", "train": {"lr": 0.01}, "model": {"modalities": ["traffic"]}},
        "s_v1": {"sweep": {"model.fusion.type": ["early", "late"], "train.lr": [0.1, 0.2]}},
    },
}


@pytest.fixture
def conf(tmp_path):
    base = yaml.safe_load(yaml.safe_dump(BASE))
    base["defaults"]["paths"]["runs"] = str(tmp_path / "runs")
    f = tmp_path / "config.yaml"
    f.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")
    return Config(f)


def test_inherit_and_replace_lists(conf):
    (c,) = conf.resolve("b_v1")
    assert c["model"]["fusion"]["type"] == "early"            # from parent
    assert c["train"]["lr"] == 0.01                            # own value
    assert c["model"]["modalities"] == ["traffic"]             # lists are replaced, not merged
    assert c["experiment"]["rq"] == "RQ1"                      # meta inherited
    assert c["experiment"]["version"] == 1


def test_sweep_expands_cartesian_product(conf):
    cs = conf.resolve("s_v1")
    assert len(cs) == 4
    assert {c["experiment"]["variant_label"] for c in cs} == {
        "fusion=early|lr=0.1", "fusion=early|lr=0.2", "fusion=late|lr=0.1", "fusion=late|lr=0.2"}


def test_overrides_are_recorded(conf):
    (c,) = conf.resolve("a_v1", ["train.lr=1e-3", "model.encoder.hidden=8"])
    assert c["train"]["lr"] == 1e-3 and c["model"]["encoder"]["hidden"] == 8
    assert c["experiment"]["overrides"] == {"train.lr": 1e-3, "model.encoder.hidden": 8}


def test_hash_ignores_bookkeeping(conf):
    (a,) = conf.resolve("a_v1")
    b = dict(a, experiment={**a["experiment"], "description": "changed"})
    assert Config.config_hash(a) == Config.config_hash(b)
    b = dict(a, train={"lr": 0.5})
    assert Config.config_hash(a) != Config.config_hash(b)


def test_unknown_experiment(conf):
    with pytest.raises(KeyError):
        conf.resolve("nope_v1")


def test_drift_guard(conf):
    (c,) = conf.resolve("a_v1")
    with Run(c):
        pass
    RunRegistry(c).check_drift(c)                              # unchanged config: fine
    changed = dict(c, train={"lr": 0.9})
    with pytest.raises(ConfigDriftError):
        RunRegistry(changed).check_drift(changed)
    adhoc = dict(changed, experiment={**c["experiment"], "overrides": {"train.lr": 0.9}})
    RunRegistry(adhoc).check_drift(adhoc)                      # ad hoc runs are exempt
    t = RunRegistry(c).table()
    assert len(t) == 1 and t.iloc[0].status == "done" and t.iloc[0].experiment == "a_v1"


def test_project_config_resolves():
    conf = Config()
    for name in conf.experiments:
        assert conf.resolve(name)


def test_source_file_per_year_and_stand_in_year(tmp_path):
    from tehran_carbon.data.grid import Grid
    from tehran_carbon.data.sources.base import Source
    for name in ["a_1403.csv", "b_1402.csv", "b_1404.csv"]:
        (tmp_path / name).write_text("x")
    cfg = {"paths": {"raw": str(tmp_path)}, "grid": Config().defaults["grid"],
           "data": {"inputs": {"t": {"inv": "a_{year}.csv", "year_proxy": {1404: 1403}, "sat": {1402: "b_1402.csv", 1404: "b_1404.csv"}}}}}
    s = type("T", (Source,), {"name": "t"})(cfg, Grid(cfg))
    assert s.file("inv", 1404).name == "a_1403.csv"                     # stand-in year
    assert s.file("sat", 1404).name == "b_1404.csv"                     # per-year file
