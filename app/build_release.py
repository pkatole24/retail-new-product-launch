"""Build the private app release from Databricks and saved final test results."""

import json
from pathlib import Path
import sys

import duckdb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fetch_two_tower_1_14 import fetch_features
from fetch_weekly_lightgbm_sources import fetch

WORK = ROOT / "data/launch_app_release_20260531"
FINAL = ROOT / "data/stage5_final_20261125"
ROSTER = """
WITH metro AS (
    SELECT storeid, max(metro_name) metro
    FROM `msba-capstone-team13`.capstone.store_metro
    GROUP BY storeid
    HAVING count(DISTINCT named_struct('id', metro_id, 'name', metro_name)) = 1
       AND count_if(metro_id IS NOT NULL AND metro_name IS NOT NULL) > 0
)
SELECT s.storeid, s.city, s.state, m.metro
FROM `msba-capstone-team13`.capstone.store_dim s LEFT JOIN metro m USING (storeid)
WHERE s.last_sale_dt BETWEEN DATE '2026-05-01' AND DATE '2026-05-31'
"""
CATALOG = """
WITH matched AS (
    SELECT model_product_id_type, model_product_id, color_identity_type, color_identity_value,
           observed_entry_date, product_seen_before_entry, sum(actual_units) observed
    FROM matched_28 GROUP BY ALL
), candidates AS (
    SELECT m.*, p.product_text, p.business_line, p.category, p.subcategory, p.size_labels
    FROM matched m JOIN products p
      USING (model_product_id_type, model_product_id, color_identity_type, color_identity_value, observed_entry_date)
    WHERE ends_with(p.filename, 'development.parquet') AND p.is_launch
      AND m.product_seen_before_entry IS NOT NULL AND nullif(trim(p.product_text), '') IS NOT NULL
      AND nullif(trim(p.subcategory), '') IS NOT NULL AND lower(p.subcategory) <> 'unknown'
      AND p.business_line IS NOT NULL AND p.category IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY p.business_line, p.category, p.subcategory, m.product_seen_before_entry,
                     m.model_product_id_type, m.model_product_id
        ORDER BY m.observed DESC, m.observed_entry_date, m.color_identity_type, m.color_identity_value
    ) = 1
)
SELECT * FROM candidates
QUALIFY row_number() OVER (
    PARTITION BY business_line, category, subcategory, product_seen_before_entry
    ORDER BY observed DESC, model_product_id_type, model_product_id
) <= 3
ORDER BY business_line, category, subcategory, product_seen_before_entry, observed DESC, model_product_id
"""


def catalog(db):
    db.read_parquet(str(FINAL / "reports/main/matched_28.parquet")).create_view("matched_28")
    db.read_parquet(str(FINAL / "panels/prelaunch_1_14/products.parquet")).create_view("products")
    result = db.sql(CATALOG)
    rows = [dict(zip(result.columns, row, strict=True)) for row in result.fetchall()]
    styles = [{"id": f"style-{i}", "product": row["product_text"], "businessLine": row["business_line"],
               "category": row["category"], "subcategory": row["subcategory"], "sizes": row["size_labels"] or [],
               "group": "New colorway" if row["product_seen_before_entry"] else "New product identity"}
              for i, row in enumerate(rows, 1)]
    return styles


def prepare_roster(db):
    exported = all((WORK / f"{name}.parquet").exists() for name in ("roster", "extended_store", "extended_category"))
    if not exported:
        fetch({"roster": ROSTER}, WORK)
    roster = db.read_parquet(str(WORK / "roster.parquet"))
    roster.create_view("roster")
    if not exported:
        db.sql("SELECT DATE '2026-06-01' origin, storeid FROM roster").write_parquet(str(WORK / "development.parquet"))
        # The snapshot is as of May 31 close. The training recipe uses a strict prior-day cutoff.
        (WORK / "panel_validation.json").write_text("{}", encoding="utf-8")
        fetch_features(WORK, last_origin="2026-05-31", source_dir=FINAL / "feature_sources")
    db.read_parquet(str(WORK / "extended_store.parquet")).create_view("features")
    db.read_parquet(str(WORK / "extended_category.parquet")).create_view("categories")
    pairs, = db.sql("SELECT count(*), count(DISTINCT r.storeid), count(f.storeid) FROM roster r LEFT JOIN features f USING(storeid)").fetchall()
    if len(set(pairs)) != 1:
        raise ValueError(f"The roster and feature join is incomplete or non-unique: {pairs}")
    invalid, = db.sql("""
        SELECT count(*) FROM features JOIN roster USING(storeid)
        WHERE observed_entry_date <> DATE '2026-06-01' OR latest_history_date > DATE '2026-05-31'
           OR latest_price_date > DATE '2026-05-31'
    """).fetchall()
    if invalid[0]:
        raise ValueError("Store features exceed the dataset-end cutoff")
    result = db.sql("""
        SELECT f.*, r.metro, r.city,
               CASE WHEN nullif(trim(r.city), '') IS NOT NULL THEN coalesce(f.state, '') || '/' || lower(trim(r.city)) END AS city_in_state,
               f.reported_sales_usable_days / nullif(f.usable_sales_days, 0) AS prior_units_per_usable_day
        FROM roster r JOIN features f USING(storeid) ORDER BY storeid
    """)
    from two_tower_features_1_14 import STORE_NUMERIC
    stores = {}
    for values in result.fetchall():
        row = dict(zip(result.columns, values, strict=True))
        identifier = row["storeid"]
        stores[identifier] = {"id": identifier, "name": f'{row["city"] or "Store"} · {identifier}',
                              "metro": row["metro"], "features": {name: row[name] for name in (*STORE_NUMERIC, "state", "city_in_state")},
                              "categoryShare": {}}
    for identifier, line, category, share in db.sql("""
        SELECT c.storeid, c.business_line, c.category, c.store_sales_share
        FROM categories c JOIN roster r USING(storeid) WHERE c.store_sales_share IS NOT NULL
    """).fetchall():
        shares = stores[identifier]["categoryShare"].setdefault(line, {})
        if category in shares:
            raise ValueError("Category shares are not unique at the store, business line and category grain")
        shares[category] = share
    return list(stores.values())


if __name__ == "__main__":
    with duckdb.connect() as db:
        styles = catalog(db)
        print(f"Prepared {len(styles)} style entries", flush=True)
        stores = prepare_roster(db)
        output = WORK / "release.json"
        output.write_text(json.dumps({"snapshotDate": "2026-05-31", "stores": stores,
                                      "styles": styles}, allow_nan=False), encoding="utf-8")
        print(f"Wrote private release with {len(stores)} stores to {output}", flush=True)
