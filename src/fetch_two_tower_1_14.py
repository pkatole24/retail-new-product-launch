"""Fetch dated lifecycle attributes and extend the released store snapshot recipe."""
import argparse
from pathlib import Path

import duckdb

from fetch_weekly_lightgbm_sources import SCHEMA, ROOT, fetch

SNAPSHOTS = ROOT / "data/local_model_sources"
SOURCES = ROOT / "data/two_tower_1_14/sources"


def fetch_features(work, *, last_origin="2025-11-30", source_dir=SOURCES, released_before="2025-12-01"):
    fetch({"lifecycle_attributes": f"""
        SELECT DISTINCT dt, model_product_id_type, model_product_id, color_identity_type,
            color_identity_value, size_labels, business_line, category, subcategory
        FROM {SCHEMA}model_product_color_store_day_lifecycle_v1 VERSION AS OF 0
        WHERE currency='USD' AND dt BETWEEN DATE '2024-11-10' AND DATE '{last_origin}'
    """, "store_city": """
        SELECT storeid, nullif(trim(city),'') AS city
        FROM `msba-capstone-team13`.capstone.store_dim VERSION AS OF 15
    """}, source_dir)
    if not (work / "panel_validation.json").exists():
        return
    with duckdb.connect() as db:
        db.from_parquet(str(SNAPSHOTS / "store_snapshot.parquet")).filter(
            f"observed_entry_date < DATE '{released_before}'").create_view("released_store")
        files = list(work.glob("train_*.parquet")) + [work / "development.parquet"]
        db.from_parquet([str(p) for p in files]).create_view("panels")
        missing = db.sql("""
            WITH pairs AS (SELECT DISTINCT origin, storeid FROM panels)
            SELECT DISTINCT origin FROM pairs p ANTI JOIN released_store s
            ON p.origin=s.observed_entry_date AND p.storeid=s.storeid ORDER BY origin
        """).fetchall()
    if not missing:
        return
    dates = ", ".join(f"(DATE '{origin}')" for origin, in missing)
    # Use the published historical aggregation SQL with a new origin panel.
    # v29 has expired. The lifecycle release records v30–32 as OPTIMIZE and
    # metadata changes, so v32 preserves the source rows used by the snapshots.
    def definition(name):
        return (ROOT / f"sql/{name}.sql").read_text().split("\nWITH ", 1)[1].strip().rstrip(";")

    history_name = "model_store_history_snapshot_draft_v1"
    published_history = definition(history_name)
    history = published_history[published_history.index(", history AS"):].replace(
        "FROM panel p", "FROM date_store_panel p")
    common = f"""
        origin_dates AS (SELECT * FROM VALUES {dates} AS d(observed_entry_date)),
        date_store_panel AS (
            SELECT d.observed_entry_date,s.storeid FROM origin_dates d CROSS JOIN (
                SELECT DISTINCT storeid FROM {SCHEMA}model_store_day VERSION AS OF 1
                WHERE dataset_version='stage1_20261112_v1'
                  AND dt BETWEEN DATE '2024-11-10' AND DATE '{last_origin}'
            ) s
        ),
        extended_history AS (WITH unused AS (SELECT 1){history})
    """
    price = definition("model_store_price_history_v1")
    category = definition("model_store_category_history_v1")
    def extended(sql):
        return sql.replace(f"{SCHEMA}{history_name}", "extended_history").replace(
            "VERSION AS OF 29 f", "VERSION AS OF 32 f").replace(
            "WHERE f.currency='USD'", f"""WHERE f.currency='USD'
              AND f.dt BETWEEN date_sub(DATE '{missing[0][0]}',90)
                           AND date_sub(DATE '{missing[-1][0]}',1)""")

    print(f"Extending store snapshots for {len(missing)} origins from {missing[0][0]} to {missing[-1][0]}",
          flush=True)
    fetch({
        "extended_store": f"""
            WITH {common}, price AS (WITH {extended(price)})
            SELECT h.*, p.observed_product_days, p.priced_product_days,
                   p.multiple_price_product_days, p.zero_price_product_days,
                   p.median_product_day_price, p.latest_price_date, p.has_price_history
            FROM extended_history h JOIN price p USING (observed_entry_date,storeid)
        """,
        "extended_category": f"WITH {common}, c AS (WITH {extended(category)}) SELECT * FROM c",
    }, work)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    fetch_features(parser.parse_args().work.resolve())
