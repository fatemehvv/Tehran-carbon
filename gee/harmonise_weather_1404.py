"""Put the 1404 ERA5-Land days of GEE_grid_NO2_weather_1404.js on the scale of the weather file used for 1402 / 1403.

The new export averages over a different area, so its values are shifted (t2m about +2 C). Every column is mapped with
a linear fit old = a + b * new, estimated on the overlap days (1402 and 1403); the bias-correction column of the old
file (one constant) is carried over. Output: the old rows unchanged, plus the mapped rows after the old file ends.

    python gee/harmonise_weather_1404.py data/raw/weather/weather_daily_era5_both.csv \
        data/raw/weather/Tehran_ERA5_daily_2023_2026.csv data/raw/weather/weather_daily_era5_1402_1404.csv
"""
import sys

import numpy as np
import pandas as pd

old_file, new_file, out_file = sys.argv[1:4]
old = pd.read_csv(old_file, parse_dates=["date"])
new = pd.read_csv(new_file, parse_dates=["date"])
both = old.merge(new, on="date", suffixes=("_old", "_new"))
later = new[new.date > old.date.max()].copy()
for c in [c for c in new.columns if c != "date"]:
    b, a = np.polyfit(both[c + "_new"], both[c + "_old"], 1)
    later[c] = a + b * later[c]
    print(f"{c}: old = {a:.3f} + {b:.4f} new (corr {np.corrcoef(both[c + '_new'], both[c + '_old'])[0, 1]:.4f})")
bias = old["bias_delta_C"].unique()
if len(bias) != 1:
    raise ValueError("the old bias correction is not a single constant; map it explicitly")
later["bias_delta_C"] = bias[0]
out = pd.concat([old, later[old.columns]], ignore_index=True)
out.assign(date=out.date.dt.strftime("%Y-%m-%d")).to_csv(out_file, index=False)
print(f"written {out_file}: {len(old)} old + {len(later)} mapped rows ({later.date.min().date()} .. {later.date.max().date()})")
