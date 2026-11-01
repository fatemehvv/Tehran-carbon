"""Tehran-carbon: single entry point. Every setting comes from configs/config.yaml.

    python main.py show      --exp rq1_gated_v1                 resolved config(s) and hashes
    python main.py manifest  [--write]                          verify (or rewrite) the sha256 of data/raw
    python main.py prepare   --exp rq1_gated_v1                 build / cache the dataset and print the split
    python main.py train     --exp rq1_ablation_v1 [--set train.lr=1e-3] [--smoke] [--variant 3]
    python main.py train     --resume <run_id>                  continue an interrupted run
    python main.py train     --exp rq3_policy_v1                RQ3: RL agent vs static policies (kind: rl)
    python main.py inventory --exp inv_l1_v1                    L1 sector totals (IPCC + Monte Carlo)
    python main.py evaluate  --hypothesis H1                    pre-registered decision rule on finished runs
    python main.py analyze   districts --exp rq2_sage_v1 rq2_base_slx_gbm_v1   district classes vs RQ1 (kappa)
    python main.py analyze   response  --exp rq2_sage_v1        response checks of the final model (for RQ3)
    python main.py summarize --exp rq1_ablation_v1              table of all finished runs / sweep variants
    python main.py runs      [--rq RQ1] [--exp rq1_] [--last 30]   run history (runs/registry.csv)
    python main.py figures                                      thesis figures and tables (docs/figures, docs/tables)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tehran_carbon.config import Config  # noqa: E402
from tehran_carbon.tracking import Run, RunRegistry  # noqa: E402

SMOKE = ["train.epochs=3", "train.seeds=[0]", "split.transfer=false", "train.save_final=false"]


def cmd_show(args):
    for cfg in Config().resolve(args.exp, args.set):
        e = cfg["experiment"]
        print(f"# {e['name']} [{e['variant_label'] or '-'}]  config {Config.config_hash(cfg)}  data {Config.data_hash(cfg)}")
        if args.full:
            import yaml
            print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))


def cmd_manifest(args):
    cfg = Config().defaults
    raw = Config.path(cfg, "raw")
    files = sorted((p for p in raw.rglob("*") if p.is_file() and p.name != "MANIFEST.csv"),
                   key=lambda p: p.relative_to(raw).as_posix().encode())
    lines = ["path,size_bytes,sha256"]
    for p in files:
        lines.append(f'"{p.relative_to(raw).as_posix()}",{p.stat().st_size},{hashlib.sha256(p.read_bytes()).hexdigest()}')
    text = "\n".join(lines) + "\n"
    target = raw / "MANIFEST.csv"
    if args.write:
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"written {target} ({len(files)} files)")
        return
    old = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    diff = sorted(set(old) ^ set(text.splitlines()))
    print("data/raw matches MANIFEST.csv" if not diff else "MISMATCH (lines only in one side):\n" + "\n".join(diff))
    if diff:
        sys.exit(1)


def cmd_prepare(args):
    import numpy as np
    from tehran_carbon.data.dataset import CellMonthDataset
    from tehran_carbon.data.grid import Grid
    from tehran_carbon.data.splits import SpatialSplit
    for cfg in Config().resolve(args.exp, args.set)[:1]:
        data = CellMonthDataset.build(cfg, force=args.force)
        split = SpatialSplit(cfg["split"], Grid(cfg), data.label_domain,
                             z=np.concatenate([data.z[y] for y in data.years]), seed=int(cfg["seed"]))
        print(json.dumps({"features": len(data.features), "label_cells": int(data.label_domain.sum()),
                          "split": split.info}, indent=2, ensure_ascii=False))


def cmd_train(args):
    from tehran_carbon.experiments.cross_validation import CrossValidation
    if args.resume:
        run = Run.resume(Config().defaults, args.resume)
        with run:
            run.finish("done", CrossValidation(run.cfg, run).run_all())
        return
    cfgs = Config().resolve(args.exp, list(args.set) + (SMOKE if args.smoke else []))
    if args.variant is not None:
        cfgs = [cfgs[args.variant]]
    for cfg in cfgs:
        RunRegistry(cfg).check_drift(cfg)
    for i, cfg in enumerate(cfgs):
        print(f"\n=== {i + 1}/{len(cfgs)}  {cfg['experiment']['name']} [{cfg['experiment']['variant_label'] or '-'}]")
        with Run(cfg) as run:
            if cfg["experiment"]["kind"] == "rl":
                from tehran_carbon.experiments.policy_search import PolicySearch
                run.finish("done", PolicySearch(cfg, run).run_all(smoke=args.smoke))
            else:
                run.finish("done", CrossValidation(cfg, run).run_all(smoke=args.smoke))


def cmd_inventory(args):
    from tehran_carbon.inventory.l1 import L1Inventory
    for cfg in Config().resolve(args.exp, args.set):
        RunRegistry(cfg).check_drift(cfg)
        with Run(cfg) as run:
            summary = {}
            for y in cfg["years"]:
                out = L1Inventory(cfg, y).run()
                for name, df in out.items():
                    run.save_table(f"L1_{name}_{y}.csv", df)
                s = out["summary"].set_index("group").MtCO2_central
                summary[f"scope1_core_Mt_{y}"] = round(float(s["Scope 1 total CORE"]), 3)
                run.log(f"{y}:\n{out['summary'].round(3).to_string(index=False)}")
                if "check" in out:
                    run.log(f"{y} vs frozen RQ1 v2:\n{out['check'].round(3).to_string(index=False)}")
            run.finish("done", summary)


def analysis_config(name: str, rq: str | None, description: str, **extra) -> dict:
    """Defaults + an `experiment` block for a tracked analysis run (it trains nothing)."""
    cfg = dict(Config().defaults)
    cfg["experiment"] = dict(name=name, rq=rq, kind="analysis", description=description, version=None, variant={},
                             variant_label="", overrides={}, **extra)
    return cfg


def cmd_evaluate(args):
    from tehran_carbon.evaluation.hypotheses import HypothesisTest
    from tehran_carbon.evaluation.policy_tests import PolicyHypothesis
    spec = Config().hypotheses[args.hypothesis]
    cfg = analysis_config(f"hypothesis_{args.hypothesis}", spec.get("rq"), f"decision rule {args.hypothesis}", spec=spec)
    test = PolicyHypothesis if spec.get("kind") == "policy" else HypothesisTest
    with Run(cfg) as run:
        run.finish("done", test(cfg, args.hypothesis, spec).run(run))


def cmd_analyze(args):
    if args.what == "districts":
        from tehran_carbon.evaluation.districts import DistrictComparison
        cfg = analysis_config("districts", "RQ2", "district classes vs RQ1 v2 (kappa)", experiments=args.exp)
        with Run(cfg) as run:
            run.finish("done", DistrictComparison(cfg, args.exp).run(run))
    else:
        from tehran_carbon.experiments.response import ResponseChecks
        cfg = analysis_config("response", "RQ2", "response checks of the final model (bridge to RQ3)",
                              experiments=args.exp[:1])
        with Run(cfg) as run:
            run.finish("done", ResponseChecks(cfg, args.exp[0]).run(run))


def cmd_summarize(args):
    import pandas as pd
    cfg = Config().defaults
    rows = []
    for _, r in RunRegistry(cfg).find(args.exp).iterrows():
        f = ROOT / r.path / "metrics.json"
        if not f.exists():
            continue
        row = {"run_id": r.run_id, "variant": r.variant}
        for part, values in json.loads(f.read_text(encoding="utf-8")).items():
            for key, metrics in values.items():
                row.update({f"{part}_{key}_{m}": round(v, 4) for m, v in metrics.items()})
        rows.append(row)
    if not rows:
        print(f"no finished runs of {args.exp}")
        return
    t = pd.DataFrame(rows)
    out = Config.path(cfg, "runs") / f"summary_{args.exp}.csv"
    t.to_csv(out, index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(t.to_string(index=False))
    print(f"\nwritten {out}")


def cmd_runs(args):
    import pandas as pd
    t = RunRegistry(Config().defaults).table()
    if args.rq:
        t = t[t.rq == args.rq]
    if args.exp:
        t = t[t.experiment.str.startswith(args.exp)]
    cols = ["run_id", "status", "experiment", "variant", "config_hash", "git_commit", "adhoc", "summary"]
    with pd.option_context("display.max_colwidth", 80, "display.width", 250):
        print(t[cols].tail(args.last).to_string(index=False) if len(t) else "no runs yet")


def cmd_figures(args):
    from tehran_carbon.viz.figures import ThesisFigures
    out = ThesisFigures(Config().defaults).run_all()
    print(f"{len(out['done'])} figures written, skipped: {out['skipped'] or 'none'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show"); p.add_argument("--exp", required=True); p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--full", action="store_true"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("manifest"); p.add_argument("--write", action="store_true"); p.set_defaults(fn=cmd_manifest)
    p = sub.add_parser("prepare"); p.add_argument("--exp", required=True); p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_prepare)
    p = sub.add_parser("train"); p.add_argument("--exp"); p.add_argument("--set", nargs="*", default=[])
    p.add_argument("--smoke", action="store_true", help="installation check: 1 fold, 3 epochs, 1 seed (ad hoc run)")
    p.add_argument("--variant", type=int, help="run only this sweep variant (0-based)")
    p.add_argument("--resume", help="run_id of an interrupted run"); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("inventory"); p.add_argument("--exp", required=True); p.add_argument("--set", nargs="*", default=[])
    p.set_defaults(fn=cmd_inventory)
    p = sub.add_parser("evaluate"); p.add_argument("--hypothesis", required=True); p.set_defaults(fn=cmd_evaluate)
    p = sub.add_parser("analyze"); p.add_argument("what", choices=["districts", "response"])
    p.add_argument("--exp", nargs="+", required=True); p.set_defaults(fn=cmd_analyze)
    p = sub.add_parser("summarize"); p.add_argument("--exp", required=True); p.set_defaults(fn=cmd_summarize)
    p = sub.add_parser("runs"); p.add_argument("--rq"); p.add_argument("--exp"); p.add_argument("--last", type=int, default=30)
    p.set_defaults(fn=cmd_runs)
    p = sub.add_parser("figures"); p.set_defaults(fn=cmd_figures)
    args = ap.parse_args()
    if args.cmd == "train" and not (args.exp or args.resume):
        ap.error("train needs --exp or --resume")
    t0 = time.time()
    args.fn(args)
    print(f"\n[{args.cmd} finished in {(time.time() - t0) / 60:.1f} min]")


if __name__ == "__main__":
    main()
