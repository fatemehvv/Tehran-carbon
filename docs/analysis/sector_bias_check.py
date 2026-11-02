"""Does the NO2 target favour some sectors? Measures and reading rule: docs/sector_bias_check.md.

Standalone and read-only: no project import, no training. Paths and the sector definitions come from configs/config.yaml.
usage (from the repository root): python docs/analysis/sector_bias_check.py
Output: docs/tables/t11_sector_bias.csv, t12_sector_bias_sigma.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
DOMAIN_RUN = "20261031-182906_rq2_sage_v2_7d87be"          # label domain and blocks of the main GNN
YEARS = [1402, 1403]
COLD = [8, 9, 10, 11]                                      # Aban..Bahman, the winter months of the policy grid
N_BOOT, SEED = 2000, 0

cfg = yaml.safe_load((ROOT / "configs/config.yaml").read_text(encoding="utf-8"))["defaults"]
inputs, target, grid = cfg["data"]["inputs"], cfg["data"]["target"], cfg["grid"]
raw = ROOT / cfg["paths"]["raw"]
SIGMAS = cfg["model"]["baseline"]["params"]["forward_operator"]["sigmas_km"]
SECTOR_LAYERS = {"road": ["inv_road"], "buildings": ["inv_residential", "inv_commercial"],
                 "industry": ["inv_industry"], "power": ["inv_power"]}
NAMES = list(SECTOR_LAYERS)
LAYER_SPEC = {**inputs["traffic"]["sectors"], **inputs["energy"]["sectors"]}
H, W = grid["H"], grid["W"]
km_y = grid["km_per_degree"] / grid["cells_per_degree"]
km_x = km_y * np.cos(np.radians(grid["lat0"] - H / grid["cells_per_degree"] / 2))

fold = np.load(ROOT / "runs" / DOMAIN_RUN / "preds" / "fold_map.npy")
domain = fold >= 0
blocks = fold[domain]
K = int(blocks.max()) + 1


def no2_months(year):
    """(12, H, W) NO2 of a Jalali year: Gregorian month of largest overlap, Farvardin -> April .. Esfand -> March."""
    gy = year + 621
    months = [(gy, m) for m in range(4, 13)] + [(gy + 1, m) for m in range(1, 4)]
    with rasterio.open(raw / inputs["satellite"]["file"]) as r:
        bands = {d: i + 1 for i, d in enumerate(r.descriptions)}
        return np.stack([r.read(bands[inputs["satellite"]["no2_band"].format(gy=a, gm=b)]).astype(float) for a, b in months])


def sector_months(year):
    """{sector: (12, H, W)} tonnes CO2 per cell and month: annual L2 x the L1 monthly share."""
    p = inputs["energy"]
    g = pd.read_csv(raw / p["l2_grid_file"].format(year=year), encoding="utf-8-sig").sort_values(p["cell_column"])
    prof = pd.read_csv(raw / p["l1_profiles_file"].format(year=year), encoding="utf-8-sig").sort_values(p["month_column"])
    out = {}
    for name, layers in SECTOR_LAYERS.items():
        total = 0.0
        for layer in layers:
            s = LAYER_SPEC[layer]
            annual = g[s["columns"]].sum(axis=1).values.reshape(H, W)
            total = total + prof[s["share"]].values[:, None, None] * annual[None]
        out[name] = total
    return out


def smooth(a, sigma):
    if sigma == 0:
        return a
    return gaussian_filter(a, sigma=(sigma / km_y, sigma / km_x), mode="constant")


def hotspot_capture(no2, sec):
    """Per block and month: sector tonnes inside the hotspots and in total, so that a resample only sums blocks."""
    hot = np.zeros((12, domain.sum()), bool)
    for m in range(12):
        v = np.log(np.clip(no2[m][domain], float(target["log_floor"]), None))
        ok = np.isfinite(v)
        z = (v - v[ok].mean()) / v[ok].std()
        hot[m] = ok & (z >= np.quantile(z[ok], float(target["hotspot_quantile"])))
    inside, total = {}, {}
    for s, a in sec.items():
        x = a[:, domain]
        inside[s] = np.stack([[x[m][(blocks == b) & hot[m]].sum() for b in range(K)] for m in range(12)])
        total[s] = np.stack([[x[m][blocks == b].sum() for b in range(K)] for m in range(12)])
    return hot, inside, total


def yield_stats(no2, sec, sigma):
    """Per month and block: X'X and X'y of NO2 ~ 1 + smoothed sectors; columns scaled to Mt so that X'X is well conditioned."""
    P = len(NAMES) + 1
    xtx, xty = np.zeros((12, K, P, P)), np.zeros((12, K, P))
    r2, corr = [], []
    for m in range(12):
        y = no2[m][domain]
        X = np.column_stack([np.ones(domain.sum())] + [smooth(sec[s][m], sigma)[domain] / 1e6 for s in NAMES])
        ok = np.isfinite(y)
        for b in range(K):
            i = ok & (blocks == b)
            xtx[m, b], xty[m, b] = X[i].T @ X[i], X[i].T @ y[i]
        beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        r2.append(1 - ((y[ok] - X[ok] @ beta) ** 2).sum() / ((y[ok] - y[ok].mean()) ** 2).sum())
        corr.append(np.corrcoef(X[ok, 1], X[ok, 2])[0, 1])
    return xtx, xty, float(np.mean(r2)), float(np.mean(corr))


def betas(xtx, xty, w, months):
    """(months, sectors) coefficients for block weights w (a bootstrap resample): NO2 column per Mt CO2 of the month."""
    out = []
    for m in months:
        A, b = (xtx[m] * w[:, None, None]).sum(0), (xty[m] * w[:, None]).sum(0)
        out.append(np.linalg.lstsq(A, b, rcond=None)[0][1:])
    return np.array(out)


def ci(values):
    return np.quantile(values, [0.025, 0.975])


rows, sens = [], []
rng = np.random.default_rng(SEED)
draws = np.stack([np.bincount(rng.integers(0, K, K), minlength=K) for _ in range(N_BOOT)]).astype(float)

for year in YEARS:
    no2, sec = no2_months(year), sector_months(year)
    hot, inside, total = hotspot_capture(no2, sec)

    # the inventory totals inside the domain, as a check of the layers
    share = {s: total[s].sum() for s in NAMES}
    grand = sum(share.values())
    for s in NAMES:
        rows.append(dict(year=year, measure="domain_CO2_Mt", item=s, value=share[s] / 1e6))
        rows.append(dict(year=year, measure="domain_CO2_share", item=s, value=share[s] / grand))

    # 1. capture of each sector inside the NO2 hotspots
    cap = {s: (inside[s] / np.maximum(total[s].sum(1, keepdims=True), 1e-12)).sum(1).mean() for s in NAMES}
    boot = {s: ((inside[s] @ draws.T) / np.maximum(total[s] @ draws.T, 1e-12)).mean(0) for s in NAMES}
    for s in NAMES:
        lo, hi = ci(boot[s])
        rows.append(dict(year=year, measure="hotspot_capture", item=s, value=cap[s], ci_low=lo, ci_high=hi))
    d1 = boot["road"] - boot["buildings"]
    rows.append(dict(year=year, measure="delta1_capture_road_minus_buildings", item="road-buildings",
                     value=cap["road"] - cap["buildings"], ci_low=ci(d1)[0], ci_high=ci(d1)[1]))

    # total CO2 in the NO2 hotspots against the best possible 10 % of the cells (raw and smoothed cells)
    tot = sum(sec[s] for s in NAMES)
    got, best = [], []
    for m in range(12):
        x = tot[m][domain]
        got.append(x[hot[m]].sum() / x.sum())
        best.append(np.sort(x)[::-1][:hot[m].sum()].sum() / x.sum())
    rows.append(dict(year=year, measure="total_CO2_in_NO2_hotspots", item="total", value=float(np.mean(got))))
    rows.append(dict(year=year, measure="total_CO2_in_best_10pct_cells", item="total", value=float(np.mean(best))))

    # 2. pattern correlation of z with each smoothed sector, each sector with its best sigma
    z = np.full((12, domain.sum()), np.nan)
    for m in range(12):
        v = np.log(np.clip(no2[m][domain], float(target["log_floor"]), None))
        z[m] = (v - np.nanmean(v)) / np.nanstd(v)
    for s in NAMES:
        by_sigma = {}
        for sg in SIGMAS:
            by_sigma[sg] = float(np.mean([spearmanr(z[m], smooth(sec[s][m], sg)[domain], nan_policy="omit")[0] for m in range(12)]))
            sens.append(dict(year=year, measure="spearman", item=s, sigma_km=sg, value=by_sigma[sg]))
        sg = max(by_sigma, key=by_sigma.get)
        rows.append(dict(year=year, measure="spearman_best_sigma", item=s, value=by_sigma[sg], sigma_km=sg))

    # 3. NO2 per tonne of CO2: one sigma for all sectors, the one with the best mean R2
    stats = {sg: yield_stats(no2, sec, sg) for sg in SIGMAS}
    for sg, (_, _, r2, corr) in stats.items():
        sens.append(dict(year=year, measure="yield_R2", item="all", sigma_km=sg, value=r2))
        sens.append(dict(year=year, measure="corr_road_buildings", item="road-buildings", sigma_km=sg, value=corr))
    sg = max(stats, key=lambda k: stats[k][2])
    xtx, xty, r2, corr = stats[sg]
    rows.append(dict(year=year, measure="yield_sigma_R2", item="all", value=r2, sigma_km=sg))
    rows.append(dict(year=year, measure="corr_road_buildings", item="road-buildings", value=corr, sigma_km=sg))
    one = np.ones(K)
    for label, months in [("all", list(range(12))), ("cold", [m - 1 for m in COLD]),
                          ("warm", [m for m in range(12) if m + 1 not in COLD])]:
        b0 = betas(xtx, xty, one, months).mean(0)
        bb = np.stack([betas(xtx, xty, w, months).mean(0) for w in draws])
        for j, s in enumerate(NAMES):
            lo, hi = ci(bb[:, j])
            rows.append(dict(year=year, measure=f"yield_{label}", item=s, value=b0[j], ci_low=lo, ci_high=hi, sigma_km=sg))
        d3 = bb[:, 0] - bb[:, 1]
        rows.append(dict(year=year, measure=f"delta3_yield_{label}_road_minus_buildings", item="road-buildings",
                         value=b0[0] - b0[1], ci_low=ci(d3)[0], ci_high=ci(d3)[1], sigma_km=sg))
    # sensitivity of delta3 to sigma
    for s2, (a, b, _, _) in stats.items():
        sens.append(dict(year=year, measure="delta3_yield_all", item="road-buildings", sigma_km=s2,
                         value=float((lambda v: v[0] - v[1])(betas(a, b, one, list(range(12))).mean(0)))))

out = pd.DataFrame(rows)
out.round(4).to_csv(ROOT / "docs/tables/t11_sector_bias.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(sens).round(4).to_csv(ROOT / "docs/tables/t12_sector_bias_sigma.csv", index=False, encoding="utf-8-sig")
with pd.option_context("display.width", 220, "display.max_rows", 200):
    print(out.round(3).to_string(index=False))
