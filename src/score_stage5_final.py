"""Check exact key and outcome parity, then score the locked final comparisons."""
import argparse
import csv
import json

import duckdb

from audit_stage5_final_inputs import OUTPUT, WINDOWS
from run_weekly_lightgbm_1_14 import check, write_json


EVENT = "model_product_id_type, model_product_id, color_identity_type, color_identity_value, observed_entry_date"
KEYS = EVENT + ", storeid"


def matches(left, right, keys=KEYS):
    return " AND ".join(f"{left}.{key} IS NOT DISTINCT FROM {right}.{key}" for key in keys.split(", "))


def scalar(db, query):
    return db.execute(query).fetchall()[0][0]


def parity(db, source, control, keys=KEYS):
    base = scalar(db, f"SELECT count(*) FROM {control}")
    scored = scalar(db, f"SELECT count(*) FROM {source}")
    unique = scalar(db, f"SELECT count(DISTINCT ({keys})) FROM {source}")
    matched = scalar(db, f"""SELECT count(*) FROM {source} s JOIN {control} c ON {matches('s', 'c', keys)}
                            WHERE s.actual_units IS NOT DISTINCT FROM c.actual_units""")
    check(base > 0 and base == scored == unique == matched, f"{source} has mismatched keys, coverage or actuals")


def summarize(db, source, comparison, model_columns):
    return " UNION ALL ".join(f"""
        (WITH launches AS (
            SELECT {EVENT}, product_seen_before_entry,
                   count(*) AS scored_store_cases, sum(actual_units) AS units,
                   sum(abs({column}-actual_units)) AS store_absolute,
                   sum({column}-actual_units) AS signed_error
            FROM {source} WHERE actual_units IS NOT NULL GROUP BY ALL
        )
        SELECT '{comparison}' AS comparison, '{model}' AS model,
               CASE WHEN grouping(product_seen_before_entry)=1 THEN 'overall'
                    WHEN product_seen_before_entry THEN 'new_color_known_product'
                    ELSE 'new_product_identity' END AS cohort,
               count(*) AS scored_launches, sum(scored_store_cases) AS scored_store_cases,
               sum(units) AS actual_units, sum(store_absolute)/nullif(sum(units),0) AS store_wape,
               sum(abs(signed_error))/nullif(sum(units),0) AS covered_network_wape,
               sum(signed_error)/nullif(sum(units),0) AS signed_volume_bias
        FROM launches GROUP BY GROUPING SETS ((), (product_seen_before_entry)))
    """ for model, column in model_columns.items())


def ensemble(db, source_glob, name, expected_members):
    db.from_parquet(str(source_glob), filename=True).create_view("raw_seed", replace=True)
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE {name} AS
        SELECT {KEYS},actual_units,avg(two_tower_units::DOUBLE) AS neural_units,
               first(frozen_predicted_units) AS lightgbm_units, first(first_week_sales) AS first_week_sales,
               count(*) AS members,count(DISTINCT filename) AS member_files,
               count(DISTINCT frozen_predicted_units) AS control_values
        FROM raw_seed GROUP BY {KEYS},actual_units
    """)
    check(scalar(db, f"""SELECT count_if(members<>{expected_members} OR member_files<>{expected_members}
                            OR control_values<>1 OR NOT isfinite(neural_units))=0 FROM {name}"""),
          f"{name} is missing seeds or has a mismatched LightGBM control")


def score(mode):
    report = OUTPUT / "reports" / mode
    report.mkdir(parents=True, exist_ok=True)
    models = {"two_part_nn": "neural_units", "lightgbm": "lightgbm_units", "categorical": "baseline_units"}
    audit, statements = {}, []
    with duckdb.connect() as db:
        db.execute("SET memory_limit='5GB'")
        db.execute("SET temp_directory=?", [str(report / "spill")])
        for index, name in enumerate(WINDOWS):
            panel = f"horizon_{index}"
            db.from_parquet(str(OUTPUT / "panels" / name / "development.parquet")).create_view(
                "candidates", replace=True)
            db.execute("CREATE OR REPLACE TEMP VIEW cohort AS SELECT * FROM candidates")
            pattern = (OUTPUT / "main" / name / "seed_*" / "predictions.parquet" if mode == "main" else
                       OUTPUT / "strict" / name / "*" / "seed_13" / "predictions.parquet")
            # The short main runner's control retains entry date explicitly, as do longer/strict controls.
            ensemble(db, pattern, "averaged", 3 if mode == "main" else 1)
            parity(db, "averaged", "cohort")
            db.from_parquet(str(OUTPUT / "baseline" / mode / f"{name}.parquet")).create_view("baseline", replace=True)
            parity(db, "baseline", "cohort")
            prediction = ("coalesce(0.25*b.prelaunch_component + 0.75*b.early_component, b.prelaunch_component)"
                          if name == "week1_8_28" else "b.prelaunch_component")
            db.execute(f"""
                CREATE TEMP TABLE {panel} AS
                SELECT a.* EXCLUDE (members,member_files,control_values,first_week_sales),
                       c.product_seen_before_entry,c.first_week_sales,{prediction} AS baseline_units
                FROM averaged a JOIN baseline b ON {matches('a','b')}
                JOIN cohort c ON {matches('a','c')}
            """)
            check(scalar(db, """SELECT count_if(latest_training_outcome >= DATE '2025-12-01')=0 FROM baseline"""),
                  f"{name}: baseline calibration crossed the frozen cutoff")
            check(scalar(db, f"""SELECT count_if(actual_units IS NOT NULL AND
                   (baseline_units IS NULL OR NOT isfinite(baseline_units) OR baseline_units<0))=0 FROM {panel}"""),
                  f"{name}: missing or invalid covered baseline forecasts")
            audit[name] = db.execute(f"SELECT count(*),count(actual_units),count_if(actual_units=0) FROM {panel}").fetchall()[0]
            db.sql(f"SELECT * FROM {panel}").write_parquet(str(report / f"{name}.parquet"), compression="zstd")
            statements.append(summarize(db, panel, name, models))
        parity_key_count = scalar(db, f"""SELECT count(*) FROM horizon_0 a
            JOIN horizon_1 b ON {matches('a','b')} JOIN horizon_2 w ON {matches('a','w')}""")
        check(all(counts[0] == parity_key_count for counts in audit.values()), "Horizon keys differ")
        summed = ", ".join(
            f"a.{column}+b.{column} AS prelaunch_{column}, w.first_week_sales+w.{column} AS updated_{column}"
            for column in models.values()
        )
        db.execute(f"""
            CREATE TEMP TABLE matched_28 AS
            SELECT a.* EXCLUDE (actual_units, neural_units, lightgbm_units, baseline_units, first_week_sales),
                   a.actual_units+b.actual_units AS actual_units, {summed}
            FROM horizon_0 a JOIN horizon_1 b ON {matches('a','b')} JOIN horizon_2 w ON {matches('a','w')}
            WHERE a.actual_units IS NOT NULL AND b.actual_units IS NOT NULL
              AND w.actual_units IS NOT NULL AND w.first_week_sales IS NOT NULL
        """)
        check(scalar(db, f"""
            SELECT count_if(a.actual_units+b.actual_units<>w.actual_units+w.first_week_sales)=0
            FROM horizon_0 a JOIN horizon_1 b ON {matches('a','b')} JOIN horizon_2 w ON {matches('a','w')}
        """), "28-day outcome decompositions disagree on covered cases")
        db.sql("SELECT * FROM matched_28").write_parquet(str(report / "matched_28.parquet"), compression="zstd")
        for plan in ("prelaunch", "updated"):
            statements.append(summarize(db, "matched_28", f"{plan}_1_28",
                                        {model: f"{plan}_{column}" for model, column in models.items()}))
        audit["matched_28_scored"] = scalar(db, "SELECT count(*) FROM matched_28")
        db.sql(" UNION ALL ".join(statements)).write_csv(str(report / "metrics.csv"), header=True)
    write_json(report / "coverage_audit.json", {
        "window_counts": ["eligible_store_cases", "covered_store_cases", "covered_zero_cases"], **audit
    })


def weekly():
    report = OUTPUT / "reports" / "weekly"
    report.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as db:
        ensemble(db, OUTPUT / "weekly" / "fit_*" / "seed_*" / "predictions.parquet", "weekly", 3)
        ensemble(db, OUTPUT / "main" / "prelaunch_1_14" / "seed_*" / "predictions.parquet", "frozen", 3)
        parity(db, "weekly", "frozen")
        db.from_parquet(str(OUTPUT / "panels/prelaunch_1_14/development.parquet")).create_view("cohort")
        db.execute("CREATE TEMP VIEW candidates AS SELECT * FROM cohort")
        parity(db, "weekly", "candidates")
        db.execute(f"""
            CREATE TEMP TABLE matched AS
            SELECT w.* EXCLUDE (member_files, members, control_values), c.product_seen_before_entry,
                   f.neural_units AS frozen_neural_units
            FROM weekly w JOIN frozen f ON {matches('w','f')} JOIN candidates c ON {matches('w','c')}
        """)
        db.sql("SELECT * FROM matched").write_parquet(str(report / "matched.parquet"), compression="zstd")
        db.sql(summarize(db, "matched", "prelaunch_1_14", {
            "weekly_nn": "neural_units", "frozen_nn": "frozen_neural_units"
        })).write_csv(str(report / "metrics.csv"), header=True)


def write_summary(mode):
    report = OUTPUT / "reports" / mode
    with (report / "metrics.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    overall = {(row["comparison"], row["model"]): row for row in rows if row["cohort"] == "overall"}

    def change(before, after, metric):
        return (float(overall[after][metric]) - float(overall[before][metric])) * 100

    lines = [f"# Stage 5 final evaluation · {mode}", ""]
    if mode == "weekly":
        frozen = ("prelaunch_1_14", "frozen_nn")
        updated = ("prelaunch_1_14", "weekly_nn")
        hours = sum(json.loads(path.read_text()).get("fit_seconds", 0)
                    for path in (OUTPUT / "weekly").glob("fit_*/seed_*/validation.json")) / 3600
        lines.extend([
            f"Weekly minus frozen store WAPE: {change(frozen, updated, 'store_wape'):+.2f} pp.",
            f"Weekly minus frozen signed volume bias: {change(frozen, updated, 'signed_volume_bias'):+.2f} pp.",
            f"Incremental model fitting time: {hours:.2f} GPU hours across 66 refits after December 1.",
            "The compute tradeoff depends on how much signed volume error the allocation decision can tolerate.",
        ])
    else:
        for window in WINDOWS:
            delta = change((window, "lightgbm"), (window, "two_part_nn"), "store_wape")
            lines.append(f"- {window} NN minus LightGBM store WAPE was {delta:+.2f} pp.")
        for model in ("lightgbm", "categorical", "two_part_nn"):
            delta = change(("prelaunch_1_28", model), ("updated_1_28", model), "store_wape")
            lines.append(f"- {model} updated minus prelaunch 28-day store WAPE was {delta:+.2f} pp on matched cases.")
    lines.extend(["", "See metrics.csv for overall and cohort scores, covered-network WAPE, signed volume bias and counts."])
    (report / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("main", "strict", "weekly"))
    mode = parser.parse_args().mode
    weekly() if mode == "weekly" else score(mode)
    write_summary(mode)
    write_json(OUTPUT / f"report_{mode}.done.json", {"mode": mode})
