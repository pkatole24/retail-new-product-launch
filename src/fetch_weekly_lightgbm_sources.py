"""Fetch the pinned, development-only inputs into the ignored local cache."""
import argparse
import os
from pathlib import Path
from time import perf_counter

import pyarrow.parquet as pq
from databricks import sql
from databricks.sdk.core import Config

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data/weekly_lightgbm_1_14/sources"
SCHEMA = "`msba-capstone-team13`.modelling."
HOST = "dbc-822edc7b-72df.cloud.databricks.com"
LIFECYCLE_SELECT = f"""
    SELECT dt, storeid, model_product_id_type, model_product_id,
           color_identity_type, color_identity_value, observed_entry_date,
           lifecycle_day, product_name, business_line, category,
           representative_observed_price, reported_sales, sales_label_complete
    FROM {SCHEMA}model_product_color_store_day_lifecycle_v1 VERSION AS OF 0
"""
STORE_DAY_SELECT = f"""
    SELECT dt, storeid, network_health_status, missing_sales_observations
    FROM {SCHEMA}model_store_day VERSION AS OF 1
    WHERE dataset_version = 'stage1_20261112_v1'
"""
SOURCES = {
    "lifecycle": LIFECYCLE_SELECT + """
        WHERE currency = 'USD' AND dt BETWEEN DATE '2024-11-10' AND DATE '2025-12-13'
          AND observed_entry_date <= DATE '2025-11-30'
          AND (dt <= DATE '2025-11-30' OR dt <= date_add(observed_entry_date, 13))
        """,
    "store_day": STORE_DAY_SELECT + """
          AND dt BETWEEN DATE '2024-11-10' AND DATE '2025-12-13'
        """,
    "membership": f"""
        SELECT model_product_id_type, model_product_id, color_identity_type,
               color_identity_value, observed_entry_date, scope_status,
               evaluation_period, product_seen_before_entry, calendar_mature,
               prediction_cutoff, window_start, window_end
        FROM {SCHEMA}model_evaluation_membership_fixed_v1 VERSION AS OF 0
        WHERE forecast_stage = 'prelaunch' AND horizon = '1_14'
          AND observed_entry_date <= DATE '2025-11-30'
        """,
}
PREVIOUS = {
    "previous_forecasts": """
        SELECT model_product_id_type, model_product_id, color_identity_type,
               color_identity_value, storeid, observed_entry_date AS forecast_date,
               prediction_cutoff AS previous_fit_cutoff,
               lightgbm_store_forecast AS previous_predicted_units,
               lightgbm_training_rows AS previous_training_rows,
               lightgbm_latest_training_label AS previous_latest_label,
               reported_sales_target AS previous_actual_units,
               healthy_store_days AS previous_healthy_days
        FROM `msba-capstone-team13`.modelling_results.store_level_forecast_results_20261119_v1
            VERSION AS OF 11
        WHERE forecast_stage = 'prelaunch' AND horizon = '1_14'
          AND evaluation_mode = 'new_color' AND lightgbm_available
          AND observed_entry_date BETWEEN DATE '2025-09-01' AND DATE '2025-11-30'
        """,
}


def fetch(sources, cache=CACHE):
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.pop("DATABRICKS_TOKEN", None)
    config = Config(profile="dbc-822edc7b-72df")
    warehouse = "/sql/1.0/warehouses/047dc47bf924314c"
    with sql.connect(server_hostname=HOST, http_path=warehouse,
                     credentials_provider=lambda: config.authenticate) as connection:
        with connection.cursor() as cursor:
            for name, query in sources.items():
                path = cache / f"{name}.parquet"
                if path.exists():
                    continue
                started = perf_counter()
                cursor.execute(query)
                batch = cursor.fetchmany_arrow(100_000)
                if not batch.num_rows:
                    raise RuntimeError(f"Empty source: {name}")
                temporary = path.with_suffix(".partial.parquet")
                count = 0
                with pq.ParquetWriter(temporary, batch.schema, compression="zstd") as writer:
                    while batch.num_rows:
                        writer.write_table(batch)
                        count += batch.num_rows
                        if count % 2_000_000 == 0:
                            print(f"{name}: {count:,} rows", flush=True)
                        batch = cursor.fetchmany_arrow(100_000)
                temporary.replace(path)
                print(f"{name}: saved {count:,} rows in {perf_counter() - started:.1f}s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", action="store_true",
                        help="Fetch the previous run's retained forecasts for matched evaluation.")
    fetch(PREVIOUS if parser.parse_args().previous else SOURCES)
