"""Attach dated product and store inputs to the backed-up forecast panels."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import duckdb

from fetch_two_tower_1_14 import SNAPSHOTS, SOURCES
from fetch_weekly_lightgbm_sources import CACHE
from run_weekly_lightgbm_1_14 import check, peak_memory_mib, write_json

IDENTITY = "model_product_id_type, model_product_id, color_identity_type, color_identity_value"
PRODUCT_NUMERIC = (
    "product_price", "lifecycle_age", "is_launch", "origin_month",
    "prior_colorway_units", "prior_colorway_days", "size_count",
)
PRODUCT_CATEGORICAL = ("business_line", "category", "subcategory", "color_identity_type", "color_identity_value")
STORE_NUMERIC = (
    "latitude", "longitude", "reported_sales_usable_days", "prior_units_per_usable_day",
    "recorded_days", "healthy_days", "usable_sales_days", "has_prior_history",
    "median_product_day_price", "has_price_history", "priced_product_days",
)
def build(work, *, source_dir=SOURCES, lifecycle_cache=CACHE, released_before="2025-12-01"):
    started = perf_counter()
    panel_validation = json.loads((work / "panel_validation.json").read_text())
    week_one = panel_validation.get("issue_offset") == 7
    early_signals = ", p.first_week_sales, p.first_week_complete" if week_one else ""
    with duckdb.connect() as db:
        db.execute("SET threads=6")
        db.execute("SET memory_limit='5GB'")
        db.execute("SET temp_directory=?", [str(work / "spill")])
        db.execute("SET preserve_insertion_order=false")
        inputs = sorted(work.glob("train_*.parquet")) + [work / "development.parquet"]
        db.from_parquet([str(p) for p in inputs], filename=True, union_by_name=True).create_view("panels")
        db.from_parquet(str(source_dir / "lifecycle_attributes.parquet")).create_view("attrs")
        db.from_parquet(str(lifecycle_cache / "lifecycle.parquet")).create_view("daily")
        db.from_parquet(str(lifecycle_cache / "store_day.parquet")).create_view("coverage")
        db.from_parquet(str(SOURCES / "store_city.parquet")).create_view("cities")
        check(db.sql("SELECT count(*)=count(DISTINCT storeid) FROM cities").fetchall()[0][0],
              "Store dimension city is not unique by store ID")
        db.execute(f"""
            CREATE TEMP TABLE groups AS
            SELECT colorway, origin, first(filename) AS filename,
                   min(observed_entry_date) AS observed_entry_date, first(is_launch) AS is_launch,
                   {', '.join(f'first({key.strip()}) AS {key.strip()}' for key in IDENTITY.split(','))},
                   arg_max(product_text, name_date ORDER BY product_text) AS product_text,
                   max(name_date) AS text_date, median(DISTINCT price) AS product_price,
                   max(price_date) AS price_evidence_through, min(lifecycle_day) AS lifecycle_age,
                   first(start_month) AS origin_month, count(*) AS panel_size,
                   count(actual_units) AS covered_stores, sum(actual_units) AS covered_units
            FROM panels GROUP BY colorway, origin;
            CREATE TEMP TABLE product_keys AS SELECT DISTINCT colorway, {IDENTITY} FROM groups;
            CREATE TEMP TABLE dated_attrs AS
                SELECT k.colorway, a.* EXCLUDE ({IDENTITY}) FROM attrs a JOIN product_keys k
                ON k.model_product_id_type=a.model_product_id_type AND k.model_product_id=a.model_product_id
                   AND k.color_identity_type=a.color_identity_type
                   AND k.color_identity_value IS NOT DISTINCT FROM a.color_identity_value;
            CREATE TEMP TABLE daily_colors AS
                SELECT k.colorway, d.dt, sum(d.reported_sales) AS units
                FROM daily d JOIN product_keys k
                  ON k.model_product_id_type=d.model_product_id_type AND k.model_product_id=d.model_product_id
                   AND k.color_identity_type=d.color_identity_type
                   AND k.color_identity_value IS NOT DISTINCT FROM d.color_identity_value
                JOIN coverage c ON c.dt=d.dt AND c.storeid=d.storeid
                WHERE c.network_health_status='healthy'
                  AND c.missing_sales_observations=0 AND d.sales_label_complete
                GROUP BY ALL;
            CREATE TEMP TABLE prior_sales AS
                SELECT g.colorway,g.origin,sum(d.units) AS prior_colorway_units,
                       count(d.dt) AS prior_colorway_days
                FROM groups g LEFT JOIN daily_colors d ON g.colorway=d.colorway
                 AND d.dt>=g.origin-90 AND d.dt<g.origin GROUP BY g.colorway,g.origin;
            CREATE TEMP TABLE dated_sizes AS
                SELECT g.colorway,g.origin,arg_max(a.size_labels,a.dt) AS size_labels,
                       max(a.dt) AS size_date
                FROM groups g LEFT JOIN dated_attrs a ON g.colorway=a.colorway
                 AND ((g.is_launch AND a.dt=g.origin)
                   OR (NOT g.is_launch AND a.dt>=g.origin-90 AND a.dt<g.origin))
                 AND len(a.size_labels)>0 GROUP BY g.colorway,g.origin;
            CREATE TEMP TABLE products AS
                SELECT (row_number() OVER (ORDER BY g.origin,g.colorway)-1)::INTEGER AS product_index,
                       g.*, a.business_line, a.category, a.subcategory, s.size_labels, s.size_date,
                       len(s.size_labels) AS size_count, p.prior_colorway_units, p.prior_colorway_days
                FROM groups g JOIN prior_sales p USING (colorway,origin)
                JOIN dated_sizes s USING (colorway,origin)
                LEFT JOIN dated_attrs a ON g.colorway=a.colorway AND g.observed_entry_date=a.dt;
        """)
        check(db.sql("""
            SELECT count(*)=count(DISTINCT (colorway,origin))
                AND sum(panel_size)=(SELECT count(*) FROM panels)
                AND count_if(size_date>origin OR text_date>origin OR price_evidence_through>origin
                    OR (NOT is_launch AND (size_date=origin OR text_date=origin OR price_evidence_through=origin))
                    OR (is_launch AND prior_colorway_days>0))=0
            FROM products
        """).fetchall()[0][0], "Product features expanded panels or used future evidence")
        db.sql("SELECT * FROM products ORDER BY product_index").write_parquet(str(work / "products.parquet"))
        print("Prepared dated product features", flush=True)

        for name in ("store", "category"):
            released = SNAPSHOTS / f"{name}_snapshot.parquet"
            db.from_parquet(str(released)).filter(
                f"observed_entry_date < DATE '{released_before}'").create_view(f"released_{name}")
            extended = work / f"extended_{name}.parquet"
            relation = db.from_parquet(str(extended)) if extended.exists() else db.sql(
                f"SELECT * FROM released_{name} WHERE false")
            relation.create_view(f"extended_{name}")
            db.execute(f"""
                CREATE TEMP VIEW snapshot_{name} AS
                SELECT * FROM released_{name}
                UNION ALL BY NAME
                SELECT e.* FROM extended_{name} e ANTI JOIN (
                    SELECT DISTINCT observed_entry_date,storeid FROM released_store
                ) r USING(observed_entry_date,storeid)
            """)
        db.execute("""
            CREATE TEMP TABLE stores AS
            SELECT (row_number() OVER (ORDER BY s.observed_entry_date,s.storeid)-1)::INTEGER AS store_index,
                   s.*, c.city, CASE WHEN c.city IS NOT NULL
                       THEN coalesce(s.state,'') || '/' || lower(c.city) END AS city_in_state,
                   s.reported_sales_usable_days / nullif(s.usable_sales_days,0) AS prior_units_per_usable_day
            FROM snapshot_store s JOIN (SELECT DISTINCT origin,storeid FROM panels) p
              ON s.observed_entry_date=p.origin AND s.storeid=p.storeid
            LEFT JOIN cities c ON s.storeid=c.storeid
        """)
        check(db.sql("""
            SELECT count(*)=count(DISTINCT (observed_entry_date, storeid))
                AND count(*)=(SELECT count(DISTINCT (origin,storeid)) FROM panels)
                AND count_if(latest_history_date>=observed_entry_date OR latest_price_date>=observed_entry_date)=0
            FROM stores
        """).fetchall()[0][0], "Missing, duplicated or future-dated store features")
        check(db.sql("""
            SELECT count(*)=count(DISTINCT(observed_entry_date,storeid,business_line,category))
                AND count_if(latest_category_date>=observed_entry_date)=0 FROM snapshot_category
        """).fetchall()[0][0], "Duplicated or future-dated category snapshots")
        db.sql("SELECT * FROM stores ORDER BY store_index").write_parquet(str(work / "stores.parquet"))
        for path in inputs:
            db.execute("CREATE OR REPLACE TEMP VIEW current_panel AS SELECT * FROM read_parquet("
                       + "'" + path.as_posix().replace("'", "''") + "')")
            db.sql(f"""
                SELECT g.product_index, s.store_index, p.actual_units,
                       CASE WHEN s.reported_sales_usable_days>0 AND g.category IS NOT NULL
                            THEN coalesce(c.store_sales_share,0) END::FLOAT AS category_mix
                       {early_signals}
                FROM current_panel p JOIN products g USING (colorway,origin)
                JOIN stores s ON s.observed_entry_date=p.origin AND s.storeid=p.storeid
                LEFT JOIN snapshot_category c ON c.observed_entry_date=p.origin AND c.storeid=p.storeid
                    AND c.business_line=g.business_line AND c.category=g.category
                ORDER BY g.product_index,s.store_index
            """).write_parquet(str(work / f"features_{path.name}"), compression="zstd")
        example_path = next(
            path for path in inputs[:-1]
            if db.from_parquet(str(path)).filter("actual_units>0").limit(1).fetchall()
        )
        example = db.sql(f"""
            SELECT p.origin, p.window_end, p.storeid, p.actual_units, p.usable_days,
                   g.* EXCLUDE (filename), {', '.join('s.' + x for x in STORE_NUMERIC)},
                   s.city,s.state,s.latest_history_date,s.latest_price_date,f.category_mix
            FROM read_parquet('{example_path.as_posix()}') p
            JOIN products g USING (colorway,origin)
            JOIN stores s ON s.observed_entry_date=p.origin AND s.storeid=p.storeid
            JOIN read_parquet('{(work / f'features_{example_path.name}').as_posix()}') f
              ON g.product_index=f.product_index AND s.store_index=f.store_index
            WHERE p.actual_units>0 ORDER BY p.origin,p.colorway,p.store LIMIT 1
        """).fetchdf().to_dict("records")[0]
        write_json(work / "example.json", example)
        write_json(work / "feature_validation.json", {
            "feature_preparation_seconds": perf_counter() - started,
            "peak_process_mib": peak_memory_mib(),
            "product_groups": db.sql("SELECT count(*) FROM products").fetchall()[0][0],
            "dated_store_pairs": db.sql("SELECT count(*) FROM stores").fetchall()[0][0],
            "development_covered_rows": db.sql("""
                SELECT sum(covered_stores) FROM products WHERE origin >= DATE '2025-09-01'
            """).fetchall()[0][0],
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    build(parser.parse_args().work.resolve())
