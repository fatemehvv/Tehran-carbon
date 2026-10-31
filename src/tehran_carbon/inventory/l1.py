"""Layer 1 of the emission inventory: annual CO2 per sector and boundary from energy statistics (IPCC 2006 Tier 1).

    E = sum_f  A_f x NCV_f x density_f x EF_f
Every factor is triangular [central, low, high] in config inventory.l1; uncertainty by Monte Carlo (median and
5-95 % range). Boundaries: CORE = Tehran + Rey + Shemiranat counties; METRO adds the counties of the large ring
power plants (point sources only). Scope 1 = fuel burned inside the boundary; Scope 2 = grid electricity of the
province, reported apart (never added to Scope 1, to avoid double counting with the plants; GHG Protocol GPC).
Liquid fuels without a recent activity figure are a separate "1x" line outside the base total.
Monthly profiles: gasoline from historical NIOPDC monthly sales, buildings from heating degree-days, others flat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config
from ..utils.calendar import JalaliYear


class L1Inventory:
    def __init__(self, cfg: dict, year: int):
        self.cfg, self.year = cfg, int(year)
        self.c = cfg["inventory"]["l1"]
        self.raw = Config.path(cfg, "raw")
        self.jy = JalaliYear(self.year)

    def run(self) -> dict[str, pd.DataFrame]:
        c, year, raw, col = self.c, self.year, self.raw, self.c["columns"]
        yc = c["years"][year]
        N = int(c["draws"])
        rng = np.random.default_rng([int(self.cfg["seed"]), year])
        params = {**c["factors"], **{k: v for k, v in yc.items() if isinstance(v, list)}}
        draw = {k: rng.triangular(v[1], v[0], v[2], N) for k, v in params.items() if isinstance(v, list)}
        cen = {k: (v[0] if isinstance(v, list) else v) for k, v in params.items()}
        days = sum(self.jy.days)

        # emission factors: natural gas in t CO2 per million m3, liquids in kg CO2 per litre
        ng, ng_c = draw["ng_ncv_MJ_m3"] * draw["ng_ef_kg_GJ"], cen["ng_ncv_MJ_m3"] * cen["ng_ef_kg_GJ"]
        liq = {}
        for f in ("gas", "dsl", "fo"):
            liq[f] = draw[f"{f}_ncv_MJ_kg"] * draw[f"{f}_density_kg_L"] * draw[f"{f}_ef_kg_GJ"] / 1000.0
            liq[f + "_c"] = cen[f"{f}_ncv_MJ_kg"] * cen[f"{f}_density_kg_L"] * cen[f"{f}_ef_kg_GJ"] / 1000.0

        g = col["county_gas"]
        gas = pd.read_csv(raw / c["files"]["county_gas"], encoding="utf-8-sig")
        gas = gas[gas[g["year"]] == year].set_index(g["county"])
        missing = (set(c["core_counties_gas"]) | set(c["power_counties"])) - set(gas.index)
        if missing:
            raise ValueError(f"counties {missing} not in {c['files']['county_gas']} for {year}; available: {list(gas.index)}")
        lines = []                                              # sector, subsector, boundary, scope, activity, unit, samples (t)

        for county in c["core_counties_gas"]:
            for sub, key in (("residential", "residential"), ("commercial_public", "commercial")):
                a = float(gas.loc[county, g[key]])
                lines.append(("buildings", sub, f"county:{county}", "1", a, "Mm3 natural gas", a * ng))
            a = float(gas.loc[county, g["industry"]])
            lines.append(("industry", "industrial_gas", f"county:{county}", "1", a, "Mm3 natural gas", a * ng))
            a = float(gas.loc[county, g["cng"]])
            lines.append(("road_transport", "CNG", f"county:{county}", "1", a, "Mm3 natural gas", a * ng))

        # gasoline: province daily sales x CORE share x days (no official city figure)
        litres = draw["gasoline_province_ML_day"] * draw["gasoline_core_share"] * 1e6 * days
        a = cen["gasoline_province_ML_day"] * cen["gasoline_core_share"] * days
        lines.append(("road_transport", "gasoline", "CORE", "1", a, "ML gasoline", litres * liq["gas"] / 1000.0))
        b = col["bus_fuel"]
        bus = pd.read_csv(raw / c["files"]["bus_fuel"], encoding="utf-8-sig")
        bus = bus[bus[b["year"]] == year].set_index(b["fleet"])[b["value"]]
        bus_l = float(sum(bus.get(f, 0.0) for f in c["bus_fleets"]))
        lines.append(("road_transport", "diesel_municipal_bus", "city", "1", bus_l / 1e6, "ML",
                      bus_l * draw["bus_litres_diesel_share"] * liq["dsl"] / 1000.0))
        lines.append(("road_transport", "diesel_heavy_other", "CORE", "1", np.nan, "-",
                      draw["heavy_diesel_other_kt"] * 1000.0))

        for county, boundary in c["power_counties"].items():
            a = float(gas.loc[county, g["power"]])
            lines.append(("power_plants", f"gas:{county}", f"{boundary}:county:{county}", "1", a, "Mm3 natural gas", a * ng))

        # liquid fuels sold in the CORE NIOPDC areas (latest years available), outside the base total
        lf, fs = c["liquid_fuels"], col["fuel_sales"]
        fsa = pd.read_csv(raw / c["files"]["fuel_sales_annual"], encoding="utf-8-sig")
        fsa = fsa[fsa[fs["area"]].isin(c["niopdc_core_areas"]) & fsa[fs["year"]].between(*lf["hist_years"])]
        hist = fsa.groupby([fs["carrier"], fs["year"]])[fs["value"]].sum().groupby(level=0).mean() / 1e3   # ML/yr
        car = c["carriers"]
        road_dsl_ml = bus_l / 1e6 + cen["heavy_diesel_other_kt"] * 1e3 / liq["dsl_c"] / 1e3
        dsl_extra = max(float(hist[car["diesel"]]) * float(lf["diesel_national_scale"]) - road_dsl_ml, 0.0)
        s = draw["liquid_fuels_scale"]
        lines += [
            ("liquid_fuels", "diesel_non_road", "CORE", "1x", dsl_extra, "ML", dsl_extra * 1e6 * liq["dsl"] * s / 1000.0),
            ("liquid_fuels", "fuel_oil", "CORE", "1x", float(hist[car["fuel_oil"]]), "ML",
             float(hist[car["fuel_oil"]]) * 1e6 * liq["fo"] * s / 1000.0),
            ("liquid_fuels", "kerosene", "CORE", "1x", float(hist[car["kerosene"]]), "ML",
             float(hist[car["kerosene"]]) * 1e6 * cen["kerosene_kg_CO2_per_L"] * s / 1000.0),
            ("liquid_fuels", "LPG", "CORE", "1x", float(lf["lpg_kt"]), "kt",
             float(lf["lpg_kt"]) * 1e3 * cen["lpg_t_CO2_per_t"] * s)]

        # Scope 2: grid factor rebuilt from the national thermal-plant fuel / net generation
        npw = yc["national_power"]
        grid_ef = (npw["gas_Mm3"] * ng + npw["fo_ML"] * 1e6 * liq["fo"] / 1000
                   + npw["gasoil_ML"] * 1e6 * liq["dsl"] / 1000) / (npw["net_GWh"] * 1000)            # t / MWh
        e = col["electricity"]
        elec = pd.read_csv(raw / c["files"]["electricity"], encoding="utf-8-sig")
        elec = elec[elec[e["year"]] == year].set_index(e["sector"])[e["value"]]
        for sector, sub in c["electricity_sectors"].items():
            a = float(elec[sector])
            lines.append(("scope2_electricity", sub, "province", "2", a, "GWh", a * 1000 * grid_ef))

        # ------------------------------------------------------------ line table and group summary
        res = pd.DataFrame([dict(sector=l[0], subsector=l[1], boundary=l[2], scope=l[3], activity=l[4], unit=l[5],
                                 tCO2_central=float(np.median(np.broadcast_to(l[6], N))),
                                 tCO2_p05=float(np.percentile(np.broadcast_to(l[6], N), 5)),
                                 tCO2_p95=float(np.percentile(np.broadcast_to(l[6], N), 95))) for l in lines])
        S = np.vstack([np.broadcast_to(np.asarray(l[6], float), N) for l in lines])
        s1, core = res.scope == "1", ~res.boundary.str.startswith("METRO")
        groups = {
            "Buildings (CORE)": s1 & (res.sector == "buildings"),
            "Road transport (CORE)": s1 & (res.sector == "road_transport"),
            "Industry gas incl. refinery fuel (CORE)": s1 & (res.sector == "industry"),
            "Power plants inside CORE": s1 & (res.sector == "power_plants") & core,
            "Power plants METRO ring": s1 & (res.sector == "power_plants") & ~core,
            "Scope 1 total CORE": s1 & core,
            "Scope 1 total METRO (CORE + ring plants)": s1,
            "Scope 2 grid electricity (province, reported apart)": res.scope == "2",
            "Liquid fuels NIOPDC CORE (estimated, outside base)": res.scope == "1x",
            "Scope 1 extended CORE (base + liquid fuels)": (s1 & core) | (res.scope == "1x"),
        }
        summary = pd.DataFrame([dict(group=k, MtCO2_central=np.median(S[m.values].sum(0)) / 1e6,
                                     MtCO2_p05=np.percentile(S[m.values].sum(0), 5) / 1e6,
                                     MtCO2_p95=np.percentile(S[m.values].sum(0), 95) / 1e6) for k, m in groups.items()])

        # ------------------------------------------------------------ monthly profiles (Jalali months)
        mdays = np.array(self.jy.days, float)
        fm = pd.read_csv(raw / c["files"]["fuel_sales_monthly"], encoding="utf-8-sig")
        fm = fm[(fm[fs["carrier"]] == car["gasoline"]) & fm[fs["area"]].isin(c["niopdc_core_areas"])]
        fm = fm.assign(m=fm[fs["month"]].map({n: i for i, n in enumerate(c["niopdc_month_names"])}))
        by = fm.groupby([fs["year"], "m"])[fs["value"]].sum().unstack()
        gas_share = by.div(by.sum(axis=1), axis=0).mean().reindex(range(12)).values
        gas_share = gas_share / gas_share.sum()
        mo, w = c["monthly"], col["weather"]
        met = pd.read_csv(raw / yc["weather_file"], encoding="utf-8-sig").groupby(w["date"])[w["temperature"]].mean()
        month = self.jy.month_of(met.index).values
        hdd_d = (float(mo["hdd_base_C"]) - met.values).clip(min=0)
        hdd = np.array([hdd_d[month == m].sum() for m in range(1, 13)])
        flat = mdays / mdays.sum()
        rb, cb = float(mo["residential_base_share"]), float(mo["commercial_base_share"])
        profiles = pd.DataFrame({"month": range(1, 13), "days": mdays.astype(int),
                                 f"HDD{mo['hdd_base_C']:g}": hdd.round(1), "gasoline_share": gas_share,
                                 "residential_gas_share": rb * flat + (1 - rb) * hdd / hdd.sum(),
                                 "commercial_gas_share": cb * flat + (1 - cb) * hdd / hdd.sum(), "flat_share": flat})

        # ------------------------------------------------------------ comparison with Shahbazi et al. (2022)
        sb = c["shahbazi_2022_Mt"]
        ours = summary.set_index("group").MtCO2_central
        comparison = pd.DataFrame([
            dict(sector="road_transport", this_study=ours["Road transport (CORE)"], shahbazi_2022=sb["road_transport"]),
            dict(sector="buildings", this_study=ours["Buildings (CORE)"], shahbazi_2022=sb["buildings"]),
            dict(sector="industry", this_study=ours["Industry gas incl. refinery fuel (CORE)"], shahbazi_2022=sb["industry"]),
            dict(sector="power_plants", this_study=ours["Power plants inside CORE"], shahbazi_2022=sb["power_plants"]),
            dict(sector="total_scope1", this_study=ours["Scope 1 total CORE"], shahbazi_2022=sb["total"])])

        out = {"lines": res, "summary": summary, "profiles": profiles, "comparison": comparison}
        ref = raw / c["check_against"].format(year=year)
        if ref.exists():                 # agreement with the frozen RQ1 v2 numbers (regression check)
            old = pd.read_csv(ref, encoding="utf-8-sig").set_index("group").MtCO2_central
            chk = summary.set_index("group").MtCO2_central.to_frame("new").join(old.rename("frozen_rq1"), how="inner")
            chk["diff_pct"] = 100 * (chk.new / chk.frozen_rq1 - 1)
            out["check"] = chk.reset_index()
        return out
