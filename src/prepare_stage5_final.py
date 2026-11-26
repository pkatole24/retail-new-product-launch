"""Assemble final lifecycle panels with the same eligibility and target builders as development."""
from datetime import timedelta
import json
import shutil

import duckdb

import run_weekly_lightgbm_1_14 as base
from audit_stage5_final_inputs import CUTOFF, LAST_ENTRY, OUTPUT, WINDOWS
from fetch_two_tower_1_14 import SOURCES as DEVELOPMENT_ATTRIBUTES, fetch_features
from fetch_weekly_lightgbm_sources import CACHE
from two_tower_features_1_14 import build


def write_panel(db, first, last, offsets, path):
    start, end = offsets
    base.validate_candidates(db, LAST_ENTRY)
    base.build_target(db, first, last, start, end)
    base.prepare_inputs(db, first, last, start, start == 7)
    issue_offset = 7 if start == 7 else 0
    first_week = "" if start == 7 else ", NULL::BIGINT AS first_week_sales"
    db.sql(f"""
        SELECT c.*, w.model_product_id_type, w.model_product_id, w.color_identity_type,
               w.color_identity_value, w.product_seen_before_entry, s.storeid,
               t.window_end, t.actual_units, t.usable_days, n.product_name AS product_text,
               c.origin + {issue_offset} AS forecast_date {first_week}
        FROM window_inputs c JOIN target t USING (colorway, store, origin)
        JOIN colorway_keys w USING (colorway) JOIN store_keys s USING (store)
        LEFT JOIN names n ON c.product_name=n.name_code
    """).write_parquet(str(path), compression="zstd")


def prepare():
    sources = OUTPUT / "sources"
    merged = OUTPUT / "feature_sources"
    merged.mkdir(exist_ok=True)
    inputs = {"membership": [sources / "membership.parquet"]}
    with duckdb.connect() as db:
        for name in ("lifecycle", "store_day", "lifecycle_attributes"):
            earlier = DEVELOPMENT_ATTRIBUTES if name == "lifecycle_attributes" else CACHE
            path = merged / f"{name}.parquet"
            if not path.exists():
                temp = path.with_suffix(".partial.parquet")
                db.from_parquet(str(earlier / path.name)).filter("dt < DATE '2025-12-01'").union(
                    db.from_parquet(str(sources / path.name))
                ).write_parquet(str(temp), compression="zstd")
                temp.replace(path)
            inputs[name] = [path]
    if not (merged / "store_city.parquet").exists():
        shutil.copyfile(DEVELOPMENT_ATTRIBUTES / "store_city.parquet", merged / "store_city.parquet")
    anchors = base.regular_origins()
    while anchors[-1] + timedelta(days=14) < LAST_ENTRY:
        anchors.append(anchors[-1] + timedelta(days=14))
    established = (base.ROOT / "sql/model_weekly_lightgbm_1_14_established.sql").read_text(encoding="utf-8")
    with base.open_sources(OUTPUT, sources=inputs, last_entry=LAST_ENTRY) as db:
        db.sql("SELECT colorway, evaluation_product_groups AS groups FROM colorway_keys").write_parquet(
            str(OUTPUT / "links.parquet"), compression="zstd")
        for name, offsets in WINDOWS.items():
            work = OUTPUT / "panels" / name
            work.mkdir(parents=True, exist_ok=True)
            if (work / "feature_validation.json").exists():
                continue
            if (work / "panel_validation.json").exists():
                fetch_features(work, last_origin=str(LAST_ENTRY), source_dir=merged)
                build(work, source_dir=merged, lifecycle_cache=merged)
                continue
            end = offsets[1]
            max_fit = LAST_ENTRY if end == 13 else CUTOFF
            max_origin = max_fit - timedelta(days=end + 1)
            launches_end = min(max_origin, CUTOFF - timedelta(days=1))
            jobs = [(day, day, established, [day]) for day in anchors if day <= max_origin]
            jobs.append((base.FIRST_DAY, launches_end,
                         "SELECT * FROM launch_inputs WHERE origin BETWEEN ? AND ?",
                         [base.FIRST_DAY, launches_end]))
            if end == 13:
                jobs.append((launches_end + timedelta(days=1), CUTOFF - timedelta(days=1),
                             "SELECT * FROM launch_inputs WHERE origin BETWEEN ? AND ?",
                             [launches_end + timedelta(days=1), CUTOFF - timedelta(days=1)]))
            for index, (first, last, query, params) in enumerate(jobs):
                db.execute(f"CREATE OR REPLACE TEMP TABLE candidate AS {query}", params)
                if db.execute("SELECT count(*) FROM candidate").fetchall()[0][0]:
                    path = work / f"train_{index:02}.parquet"
                    write_panel(db, first, last, offsets, path)
            db.execute("""CREATE OR REPLACE TEMP TABLE candidate AS SELECT * FROM launch_inputs
                          WHERE origin BETWEEN ? AND ?""", [CUTOFF, LAST_ENTRY])
            write_panel(db, CUTOFF, LAST_ENTRY, offsets, work / "development.parquet")
            db.from_parquet(str(work / "train_*.parquet")).create_view("prepared", replace=True)
            count, audited, duplicates = db.execute("""
                SELECT count(actual_units), count(actual_units) FILTER (WHERE window_end < ?),
                       count(*)-count(DISTINCT (colorway,storeid,origin))
                FROM prepared
            """, [CUTOFF]).fetchall()[0]
            audit = json.loads((OUTPUT / "audit" / name / "cutoff.json").read_text())
            base.check(duplicates == 0 and audited == sum(row["training_rows"] for row in audit["monthly"]),
                       f"{name} does not match the frozen cutoff audit")
            base.write_json(work / "panel_validation.json", {
                "last_offset": end, "issue_offset": 7 if offsets[0] == 7 else 0,
                "last_entry": LAST_ENTRY, "prepared_training_rows": count,
                "control_path": str(work / "control.parquet"),
            })
            print(f"Prepared {name}: {audited:,} frozen training rows", flush=True)
            fetch_features(work, last_origin=str(LAST_ENTRY), source_dir=merged)
            build(work, source_dir=merged, lifecycle_cache=merged)


if __name__ == "__main__":
    prepare()
    base.write_json(OUTPUT / "prepare.done.json", {"step": "prepare"})
