"""Prepare cutoff audits and pinned source extensions for the locked evaluation."""
import argparse
from collections import defaultdict
from datetime import date, timedelta
import json

import duckdb

from fetch_weekly_lightgbm_sources import CACHE, LIFECYCLE_SELECT, ROOT, SCHEMA, STORE_DAY_SELECT, fetch
from fetch_extended_weekly_lightgbm import CACHE as EXTENDED_SOURCES
import run_weekly_lightgbm_1_14 as lifecycle


OUTPUT = ROOT / "data/stage5_final_20261125"
CUTOFF = date(2025, 12, 1)
LAST_ENTRY = date(2026, 5, 4)
# Delta history records only layout and retention maintenance after creation.
RELEASED_TARGET = f"{SCHEMA}model_launch_target_name_merged_v1 VERSION AS OF 12"
WINDOWS = {"prelaunch_1_14": (0, 13), "prelaunch_15_28": (14, 27), "week1_8_28": (7, 27)}


def audit_training():
    work = OUTPUT / "audit"
    work.mkdir(parents=True, exist_ok=True)
    raw_files = sorted((ROOT / "data/raw/shop.lululemon.com").rglob("*.parquet"))
    lifecycle.check(bool(raw_files), "Raw sales history files missing")
    with duckdb.connect() as db:
        raw_start = db.from_parquet([str(path) for path in raw_files]).aggregate("min(dt)").fetchall()[0][0]
    anchors = lifecycle.regular_origins()
    while anchors[-1] + timedelta(days=14) < CUTOFF:
        anchors.append(anchors[-1] + timedelta(days=14))
    with lifecycle.open_sources(work, EXTENDED_SOURCES) as db:
        source_start = db.execute("SELECT min(dt) FROM daily").fetchall()[0][0]
        for name, (first, last) in WINDOWS.items():
            window_dir = work / name
            window_dir.mkdir(exist_ok=True)
            _, audit, _ = lifecycle.cache_training(
                db, window_dir, CUTOFF, lifecycle.FIRST_DAY, anchors,
                first, last, first == 7, retain_keys=True,
            )
            by_month = defaultdict(lambda: {"candidate_rows": 0, "training_rows": 0, "covered_zero_rows": 0})
            for month in duckdb.connect().execute("""
                SELECT strftime(month, '%Y-%m')
                FROM generate_series(date_trunc('month', ?::DATE),
                                     date_trunc('month', ?::DATE - 1), INTERVAL 1 MONTH) AS m(month)
            """, [raw_start, CUTOFF]).fetchall():
                for kind in ("established", "first_entry"):
                    by_month[month[0], kind]
            for row in audit:
                counts = by_month[row["month"], "first_entry" if row["is_launch"] else "established"]
                counts["candidate_rows"] += int(row["eligible"])
                counts["training_rows"] += int(row["training_rows"])
                if row["training_rows"]:
                    counts["covered_zero_rows"] += int(row["zero_rows"])
            valid = [row for row in audit if row["training_rows"]]
            first_outcome = min(row["first_outcome"] for row in valid)
            last_outcome = max(row["last_outcome"] for row in valid)
            lifecycle.check(last_outcome.date() < CUTOFF, "Training outcome crossed the frozen cutoff")
            receipt = {
                "window": name,
                "raw_start": raw_start,
                "lifecycle_start": source_start,
                "fit_cutoff": CUTOFF,
                "earliest_eligible_outcome": first_outcome,
                "latest_eligible_outcome": last_outcome,
                "monthly": [
                    dict(month=month, example_type=kind, **counts)
                    for (month, kind), counts in sorted(by_month.items())
                ],
            }
            lifecycle.write_json(window_dir / "cutoff.json", receipt)
            print(json.dumps(receipt, default=str), flush=True)


def fetch_final():
    last_outcome = LAST_ENTRY + timedelta(days=27)
    fetch({
        "membership": f"""
            SELECT * FROM {SCHEMA}model_evaluation_membership_fixed_v1 VERSION AS OF 0
            WHERE forecast_stage='prelaunch' AND horizon='1_14'
        """,
        "lifecycle": LIFECYCLE_SELECT + f"""
            WHERE currency='USD' AND dt BETWEEN DATE '{CUTOFF}' AND DATE '{last_outcome}'
        """,
        "store_day": STORE_DAY_SELECT + f"""
            AND dt BETWEEN DATE '{CUTOFF}' AND DATE '{last_outcome}'
        """,
        "lifecycle_attributes": f"""
            SELECT DISTINCT dt, model_product_id_type, model_product_id, color_identity_type,
                color_identity_value, size_labels, business_line, category, subcategory
            FROM {SCHEMA}model_product_color_store_day_lifecycle_v1 VERSION AS OF 0
            WHERE currency='USD' AND dt BETWEEN DATE '{CUTOFF}' AND DATE '{LAST_ENTRY}'
        """,
    }, OUTPUT / "sources")


def verify_december():
    sources = OUTPUT / "sources"
    fetch({"december_release": f"""
        SELECT t.model_product_id_type, t.model_product_id, t.color_identity_type,
               t.color_identity_value, t.observed_entry_date AS origin, t.storeid,
               t.horizon, t.reported_sales_target AS actual_units, t.target_complete
        FROM {RELEASED_TARGET} t
        WHERE t.observed_entry_date = DATE '{CUTOFF}' AND t.horizon IN ('1_14','15_28','8_28')
    """}, sources)
    inputs = {
        name: [CACHE / f"{name}.parquet", sources / f"{name}.parquet"]
        for name in ("lifecycle", "store_day")
    }
    inputs["membership"] = [sources / "membership.parquet"]
    for paths in inputs.values():
        lifecycle.check(all(path.is_file() for path in paths), "Final source fetch has not completed")
    work = OUTPUT / "audit"
    work.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        for name, paths in inputs.items():
            relation = db.from_parquet(str(paths[0]))
            if name != "membership":
                relation = relation.filter("dt < DATE '2025-12-01'").union(db.from_parquet(str(paths[1])))
            relation.create_view(f"{'store' if name == 'store_day' else name}_source")
        sql = (ROOT / "sql/model_weekly_lightgbm_1_14.sql").read_text(encoding="utf-8")
        sql = sql.replace("DATE '2025-11-30'", lifecycle.date_sql(CUTOFF)).replace(
            "AND w.evaluation_period IN ('outside_evaluation_periods', 'development')",
            f"AND w.observed_entry_date = {lifecycle.date_sql(CUTOFF)}",
        )
        db.execute("SET memory_limit='5GB'")
        db.execute("SET temp_directory=?", [str(work / "verify_spill")])
        db.execute(sql)
        db.execute("CREATE TEMP TABLE candidate AS SELECT * FROM launch_inputs")
        lifecycle.validate_candidates(db, CUTOFF)
        db.from_parquet(str(sources / "december_release.parquet")).create_view("released")
        receipt = {}
        for name, (first, last) in WINDOWS.items():
            lifecycle.build_target(db, CUTOFF, CUTOFF, first, last)
            lifecycle.prepare_inputs(db, CUTOFF, CUTOFF, first, first == 7)
            horizon = name.split("_", 1)[1]
            counts = db.execute("""
                WITH local AS (
                    SELECT w.model_product_id_type, w.model_product_id, w.color_identity_type,
                           w.color_identity_value, t.origin, s.storeid, t.actual_units
                    FROM target t JOIN colorway_keys w USING(colorway) JOIN store_keys s USING(store)
                ), parity AS (
                    SELECT a.origin AS local_origin, b.origin AS released_origin, a.actual_units,
                           b.actual_units AS released_units, b.target_complete
                    FROM local a FULL JOIN (SELECT * FROM released WHERE horizon=?) b
                      ON a.model_product_id_type=b.model_product_id_type AND a.model_product_id=b.model_product_id
                     AND a.color_identity_type=b.color_identity_type
                     AND a.color_identity_value IS NOT DISTINCT FROM b.color_identity_value
                     AND a.origin=b.origin AND a.storeid=b.storeid
                )
                SELECT count(*), count(actual_units), count_if(actual_units=0),
                       count_if(local_origin IS NULL OR released_origin IS NULL
                           OR actual_units IS DISTINCT FROM released_units
                           OR (actual_units IS NOT NULL) IS DISTINCT FROM target_complete)
                FROM parity
            """, [horizon]).fetchall()[0]
            receipt[name] = dict(zip(("issued", "covered", "covered_zeros", "release_differences"), counts))
            lifecycle.write_json(work / "december_inputs.json", receipt)
            lifecycle.check(counts[0] > 0 and counts[-1] == 0, f"December 1 {name} disagrees with released outcomes")
        print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("audit", "fetch", "verify"))
    step = parser.parse_args().step
    {"audit": audit_training, "fetch": fetch_final, "verify": verify_december}[step]()
    lifecycle.write_json(OUTPUT / f"{step}.done.json", {"step": step})
