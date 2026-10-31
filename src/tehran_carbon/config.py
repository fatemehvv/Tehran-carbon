"""Configuration: ONE file, configs/config.yaml.

  defaults     every setting (paths, grid, data, model, train, eval, inventory)
  experiments  versioned entries (name ends in _v<N>) that list only what differs from the defaults, or from the
               experiment they `inherits`. A `sweep` block expands one entry into the Cartesian product of its values.
  hypotheses   pre-registered decision rules (H1, H2) evaluated on finished runs

Rule: an experiment that already has runs is never edited; a change is a new version, e.g.
    rq1_gated_v2: {inherits: rq1_gated_v1, train: {lr: 1.0e-3}}
RunRegistry.check_drift() enforces this by comparing the resolved config hash with earlier runs.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = ROOT / "configs" / "config.yaml"


class Config:
    META_KEYS = ("inherits", "sweep", "rq", "kind", "description")

    def __init__(self, path: Path | str = CONFIG_FILE):
        self.path = Path(path)
        raw = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.defaults = raw["defaults"]
        self.experiments = raw.get("experiments") or {}
        self.hypotheses = raw.get("hypotheses") or {}

    def resolve(self, name: str, overrides=()) -> list[dict]:
        """Resolved configs of one experiment: one per sweep combination (a single one without a sweep).

        overrides: ["train.lr=1e-3", ...], values parsed as YAML. They are recorded in experiment.overrides and the
        run is flagged ad hoc in the registry (never used for a hypothesis decision).
        """
        lineage, n = [], name
        while n:
            if n not in self.experiments:
                raise KeyError(f"experiment '{n}' is not in {self.path.name}; known: {', '.join(sorted(self.experiments))}")
            if n in lineage:
                raise ValueError(f"inheritance loop: {' -> '.join(lineage + [n])}")
            lineage.append(n)
            n = (self.experiments[n] or {}).get("inherits")
        entries = [self.experiments[n] or {} for n in lineage]

        cfg = copy.deepcopy(self.defaults)
        for e in reversed(entries):
            cfg = self.merge(cfg, {k: v for k, v in e.items() if k not in self.META_KEYS})
        meta = {k: next((e[k] for e in entries if e.get(k) is not None), None) for k in ("rq", "kind", "description")}
        version = re.search(r"_v(\d+)$", name)
        sweep = entries[0].get("sweep") or {}
        keys = list(sweep)

        out = []
        for combo in (itertools.product(*(sweep[k] for k in keys)) if keys else [()]):
            c = copy.deepcopy(cfg)
            variant = dict(zip(keys, combo))
            for k, v in variant.items():
                self.set(c, k, copy.deepcopy(v))
            adhoc = {}
            for ov in overrides:
                k, sep, v = ov.partition("=")
                if not sep:
                    raise ValueError(f"override '{ov}' must look like key=value")
                adhoc[k.strip()] = yaml.safe_load(v)
                self.set(c, k.strip(), adhoc[k.strip()])
            label = "|".join(
                re.sub(r"\.type$", "", re.sub(r"^(model|data|train|split|eval)\.", "", k)) + "="
                + ("+".join(map(str, v)) if isinstance(v, list) else str(v)) for k, v in variant.items())
            c["experiment"] = dict(name=name, rq=meta["rq"], kind=meta["kind"] or "train",
                                   description=meta["description"] or "",
                                   version=int(version.group(1)) if version else None,
                                   variant=variant, variant_label=label, overrides=adhoc)
            out.append(c)
        return out

    # ------------------------------------------------------------------ dict utilities used on resolved configs
    @staticmethod
    def merge(base: dict, over: dict) -> dict:
        """Recursive merge; dicts are merged, everything else (lists included) is replaced."""
        out = copy.deepcopy(base)
        for k, v in (over or {}).items():
            out[k] = Config.merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
        return out

    @staticmethod
    def get(cfg: dict, key: str, default=KeyError):
        node = cfg
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is KeyError:
                    raise KeyError(key)
                return default
            node = node[part]
        return node

    @staticmethod
    def set(cfg: dict, key: str, value) -> None:
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value

    @staticmethod
    def path(cfg: dict, key: str) -> Path:
        """Absolute path of cfg['paths'][key] (paths in the config are relative to the repository root)."""
        p = Path(cfg["paths"][key])
        return p if p.is_absolute() else ROOT / p

    # ------------------------------------------------------------------ hashes
    @staticmethod
    def digest(obj) -> str:
        return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()[:12]

    @staticmethod
    def config_hash(cfg: dict) -> str:
        """Everything that can change a training result. Left out: the `experiment` block (bookkeeping) and the
        `analysis` settings (they only affect analyses of finished runs, so tuning them is not config drift)."""
        return Config.digest({k: v for k, v in cfg.items() if k not in ("experiment", "analysis")})

    @staticmethod
    def data_hash(cfg: dict) -> str:
        """The processed dataset: grid, years, data section and the raw-file manifest."""
        manifest = Config.path(cfg, "raw") / "MANIFEST.csv"
        mtext = manifest.read_text(encoding="utf-8") if manifest.exists() else ""
        return Config.digest({"grid": cfg["grid"], "years": cfg["years"], "data": cfg["data"], "manifest": mtext})
