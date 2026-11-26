"""Extend the pinned local inputs only for later outcomes on November entries."""
from fetch_weekly_lightgbm_sources import LIFECYCLE_SELECT, ROOT, STORE_DAY_SELECT, fetch

OUTPUT = ROOT / "data/weekly_lightgbm_extended"
CACHE = OUTPUT / "sources"
SOURCES = {
    "lifecycle": LIFECYCLE_SELECT + """
        WHERE currency = 'USD' AND dt BETWEEN DATE '2025-12-01' AND DATE '2025-12-27'
          AND observed_entry_date <= DATE '2025-11-30'
          AND dt BETWEEN date_add(observed_entry_date, 14) AND date_add(observed_entry_date, 27)
        """,
    "store_day": STORE_DAY_SELECT + "AND dt BETWEEN DATE '2025-12-14' AND DATE '2025-12-27'",
}

if __name__ == "__main__":
    fetch(SOURCES, CACHE)
