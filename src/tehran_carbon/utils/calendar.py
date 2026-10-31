"""A Jalali (Solar Hijri) study year: its 12 months and their Gregorian counterparts (via jdatetime)."""
from __future__ import annotations

import jdatetime
import pandas as pd


class JalaliYear:
    def __init__(self, year: int):
        self.year = int(year)
        self.spans = []                                   # [(first day, first day of next month)] x 12
        for m in range(1, 13):
            a = jdatetime.date(self.year, m, 1).togregorian()
            b = jdatetime.date(self.year + (m == 12), m % 12 + 1, 1).togregorian()
            self.spans.append((pd.Timestamp(a), pd.Timestamp(b)))
        self.days = [(b - a).days for a, b in self.spans]
        self.gregorian_year = self.spans[0][0].year       # year of Nowruz (1402 -> 2023)
        # Gregorian (year, month) with the largest overlap with each Jalali month (Farvardin -> April)
        self.gregorian_months = []
        for a, b in self.spans:
            days = pd.date_range(a, b - pd.Timedelta(days=1), freq="D")
            self.gregorian_months.append(pd.Series(list(zip(days.year, days.month))).value_counts().index[0])

    def month_of(self, dates) -> pd.Series:
        """Jalali month 1..12 of Gregorian dates (0 outside the year)."""
        d = pd.to_datetime(pd.Series(dates))
        out = pd.Series(0, index=d.index)
        for m, (a, b) in enumerate(self.spans, start=1):
            out[(d >= a) & (d < b)] = m
        return out
