"""Run tracking: every run gets its own folder and one row in runs/registry.csv (the run history, kept in git).

runs/<YYYYMMDD-HHMMSS>_<experiment>_<config hash>/
    config.yaml        resolved config (exactly what ran)
    meta.json          experiment, variant, hashes, git commit, environment, timing, status, summary
    metrics.json       metrics          split.json / fold_info.csv / preds/ / models/ / figures/
    log.txt            everything printed by run.log()
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import time
import traceback
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config import ROOT, Config


class ConfigDriftError(RuntimeError):
    pass


class RunRegistry:
    COLUMNS = ["run_id", "started", "finished", "status", "experiment", "variant", "rq", "kind", "version",
               "config_hash", "data_hash", "git_commit", "git_dirty", "adhoc", "duration_s", "summary", "path"]

    def __init__(self, cfg: dict):
        self.file = Config.path(cfg, "runs") / "registry.csv"

    def table(self) -> pd.DataFrame:
        """Latest row of every run (a resumed run appends a new row)."""
        if not self.file.exists():
            return pd.DataFrame(columns=self.COLUMNS)
        df = pd.read_csv(self.file, dtype=str, keep_default_na=False)
        return df.drop_duplicates("run_id", keep="last")

    def append(self, row: dict) -> None:
        self.file.parent.mkdir(parents=True, exist_ok=True)
        new = not self.file.exists()
        with open(self.file, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLUMNS)
            if new:
                w.writeheader()
            w.writerow(row)

    def find(self, experiment: str, status: str = "done", include_adhoc: bool = False) -> pd.DataFrame:
        t = self.table()
        t = t[(t.experiment == experiment) & (t.status == status)]
        return t if include_adhoc else t[t.adhoc == "False"]

    def check_drift(self, cfg: dict) -> None:
        """Refuse to run an experiment whose definition changed after it was run (ad hoc runs are exempt)."""
        e = cfg["experiment"]
        if e["overrides"]:
            return
        same = self.find(e["name"])
        same = same[same.variant == e["variant_label"]]
        h = Config.config_hash(cfg)
        if len(same) and not (same.config_hash == h).any():
            nxt = f"{e['name'].rsplit('_v', 1)[0]}_v{(e['version'] or 1) + 1}"
            raise ConfigDriftError(
                f"experiment '{e['name']}'{' [' + e['variant_label'] + ']' if e['variant_label'] else ''} already ran "
                f"with config {', '.join(sorted(set(same.config_hash)))}, but its definition now gives {h}.\n"
                f"Do not edit an experiment that has runs: add a new version in configs/config.yaml, e.g.\n"
                f"    {nxt}: {{inherits: {e['name']}, ...}}\n(or use --set ... for a one-off ad hoc test)")


class Run:
    """with Run(cfg) as run: ...  -> status 'done' on success, 'failed' (with traceback) on an exception."""

    PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torch_geometric", "geopandas", "rasterio"]

    def __init__(self, cfg: dict, run_id: str | None = None):
        self.cfg = cfg
        self.registry = RunRegistry(cfg)
        e = cfg["experiment"]
        chash = Config.config_hash(cfg)
        self.id = run_id or f"{time.strftime('%Y%m%d-%H%M%S')}_{e['name']}_{chash[:6]}"
        self.dir = Config.path(cfg, "runs") / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        if run_id and (self.dir / "meta.json").exists():
            self.meta = json.loads((self.dir / "meta.json").read_text(encoding="utf-8"))
            self.meta.update(status="running", resumed=self.meta.get("resumed", 0) + 1)
        else:
            (self.dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                                                  encoding="utf-8")
            versions = {}
            for p in self.PACKAGES:
                try:
                    versions[p] = metadata.version(p)
                except metadata.PackageNotFoundError:
                    versions[p] = None
            self.meta = dict(run_id=self.id, experiment=e["name"], variant=e["variant"], variant_label=e["variant_label"],
                             rq=e["rq"], kind=e["kind"], version=e["version"], overrides=e["overrides"],
                             config_hash=chash, data_hash=Config.data_hash(cfg), git=self.git_state(),
                             env={"python": platform.python_version(), "platform": platform.platform(),
                                  "packages": versions},
                             started=time.strftime("%Y-%m-%d %H:%M:%S"), status="running")
        self.save_json("meta.json", self.meta)

    @classmethod
    def resume(cls, cfg_like: dict, run_id: str) -> "Run":
        d = Config.path(cfg_like, "runs") / run_id
        if not (d / "config.yaml").exists():
            raise FileNotFoundError(f"no run folder {d}")
        return cls(yaml.safe_load((d / "config.yaml").read_text(encoding="utf-8")), run_id=run_id)

    @staticmethod
    def git_state() -> dict:
        out = {}
        for key, args in [("commit", ["rev-parse", "--short", "HEAD"]), ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
                          ("changes", ["status", "--porcelain", "--untracked-files=no"])]:
            try:
                r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=20)
                out[key] = r.stdout.strip() if r.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                out[key] = ""
        return {"commit": out["commit"] or "none", "branch": out["branch"], "dirty": bool(out["changes"])}

    # ------------------------------------------------------------------ files
    def path(self, *parts: str) -> Path:
        p = self.dir.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def save_json(self, name: str, obj) -> Path:
        p = self.path(name)
        p.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                                default=lambda o: o.item() if isinstance(o, np.generic) else str(o)), encoding="utf-8")
        return p

    def save_table(self, name: str, df: pd.DataFrame) -> Path:
        p = self.path(name)
        df.to_csv(p, index=False, encoding="utf-8-sig")
        return p

    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(self.path("log.txt"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    # ------------------------------------------------------------------ lifecycle
    def finish(self, status: str = "done", summary: dict | None = None) -> None:
        m = self.meta
        m.update(status=status, finished=time.strftime("%Y-%m-%d %H:%M:%S"), duration_s=round(time.time() - self.t0, 1))
        if summary is not None:
            m["summary"] = summary
        self.save_json("meta.json", m)
        self.registry.append(dict(
            run_id=self.id, started=m["started"], finished=m["finished"], status=status, experiment=m["experiment"],
            variant=m["variant_label"], rq=m["rq"] or "", kind=m["kind"], version=m["version"] or "",
            config_hash=m["config_hash"], data_hash=m["data_hash"], git_commit=m["git"]["commit"],
            git_dirty=m["git"]["dirty"], adhoc=bool(m["overrides"]), duration_s=m["duration_s"],
            summary=json.dumps(m.get("summary", {}), ensure_ascii=False,
                               default=lambda o: o.item() if isinstance(o, np.generic) else str(o)),
            path=str(self.dir.relative_to(ROOT)) if self.dir.is_relative_to(ROOT) else str(self.dir)))

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            if self.meta["status"] == "running":
                self.finish("done")
        else:
            self.log("FAILED:\n" + "".join(traceback.format_exception(exc_type, exc, tb)))
            self.finish("failed")
        return False
