"""Run final frozen, expanding weekly and date-specific strict lifecycle models."""
import argparse
from datetime import timedelta
import hashlib
import json

import duckdb
import lightgbm as lgb
import numpy as np

from audit_stage5_final_inputs import CUTOFF, LAST_ENTRY, OUTPUT, WINDOWS
from fit_two_tower_1_14 import STRUCTURED_SIZES_PREPARATION, TWO_PART, run as neural
from fit_strict_lifecycle import read_training, fit_on_disk
import run_weekly_lightgbm_1_14 as base


SEEDS = (13, 29, 47)
FIT_DATES = [CUTOFF + timedelta(weeks=i) for i in range(23)]
REFERENCES = {
    "prelaunch_1_14": "two_tower_1_14/stage5_two_part_structured_sizes_20261123_v1",
    **{name: f"two_tower_extended/stage5_20261123_v1/{name}/seed_13"
       for name in ("prelaunch_15_28", "week1_8_28")},
}


def run_neural(out, features, name, seed, fit=CUTOFF, weekly=None, strict=None):
    receipt = out / "validation.json"
    if receipt.exists() and json.loads(receipt.read_text())["development_exact_key_target_coverage_match"]:
        return
    checkpoint = out / "model.pt"
    if checkpoint.exists() and not (receipt.exists() and (out / "preprocessing.json").exists()):
        checkpoint.replace(out / "interrupted_model.pt")
    print(f"{name}: {fit} seed {seed}, {out.name}", flush=True)
    neural(out, features, False, checkpoint.exists(), TWO_PART,
           STRUCTURED_SIZES_PREPARATION, seed, base.ROOT / "data" / REFERENCES[name],
           strict=strict, weekly=weekly, final_fit=fit)


def load(db, features):
    db.from_parquet(str(features / "train_*.parquet")).create_view("training")
    db.from_parquet(str(features / "development.parquet")).create_view("candidates")


def control(db, features, numeric):
    path = features / "control.parquet"
    if path.exists():
        return path
    columns = numeric + base.CATEGORICAL
    model_path = features / "frozen_lightgbm.txt"
    maps_path = features / "frozen_category_maps.npz"
    if not model_path.exists() or not maps_path.exists():
        training = features / "frozen_training.parquet"
        earliest, latest, expected = db.execute("""
            SELECT min(window_end), max(window_end), count(*) FROM training
            WHERE window_end < ? AND actual_units IS NOT NULL
        """, [CUTOFF]).fetchall()[0]
        base.check(expected > 0 and latest < CUTOFF, "Frozen LightGBM has invalid training support")
        db.sql(f"""SELECT {', '.join(columns)}, actual_units FROM training
                   WHERE window_end < {base.date_sql(CUTOFF)} AND actual_units IS NOT NULL
                   ORDER BY origin,colorway,store""").write_parquet(str(training), compression="zstd")
        model, maps, assembly, fit = base.fit_model(db, [training], expected, numeric)
        model.booster_.save_model(str(model_path))
        np.savez(maps_path, *maps)
        base.write_json(features / "lightgbm_fit.json", {
            "fit_date": CUTOFF, "fit_settings": base.SETTINGS, "training_rows": expected,
            "earliest_training_outcome": earliest, "latest_training_outcome": latest,
            "assembly_seconds": assembly, "fit_seconds": fit,
        })
        training.unlink()
    model = lgb.Booster(model_file=str(model_path))
    with np.load(maps_path) as arrays:
        maps = [arrays[key] for key in arrays.files]
    signal_column = "" if "first_week_sales" in columns else ", first_week_sales"
    scored = db.sql(f"""SELECT {', '.join(columns)}, {', '.join(base.KEYS)}, origin AS observed_entry_date,
                        actual_units, product_seen_before_entry, colorway {signal_column}
                        FROM candidates ORDER BY origin,colorway,store""").df()
    raw = scored[columns].fillna({col: -1 for col in base.CATEGORICAL}).to_numpy(dtype=np.float32)
    signal = scored["first_week_sales"].copy()
    scored = scored.drop(columns=columns)
    scored["first_week_sales"] = signal
    scored["frozen_predicted_units"], _ = base.predict(model, maps, raw, numeric)
    db.from_df(scored).write_parquet(str(path), compression="zstd")
    return path


def main():
    for name in WINDOWS:
        features = OUTPUT / "panels" / name
        numeric = base.NUMERIC + (["first_week_sales", "first_week_complete"] if name == "week1_8_28" else [])
        with duckdb.connect() as db:
            load(db, features)
            # Keep the early-signal actual on the control for matched 28-day scoring.
            control(db, features, numeric)
        for seed in SEEDS:
            run_neural(OUTPUT / "main" / name / f"seed_{seed}", features, name, seed)


def weekly():
    name = "prelaunch_1_14"
    features = OUTPUT / "panels" / name
    base.check(FIT_DATES[-1] == LAST_ENTRY, "Weekly launch schedule does not end on May 4")
    with duckdb.connect() as db:
        load(db, features)
        audit = db.execute("""
            WITH fits AS (SELECT unnest(?::DATE[]) AS cutoff),
            panels AS (SELECT origin,window_end,actual_units,is_launch FROM training
                       UNION ALL SELECT origin,window_end,actual_units,is_launch FROM candidates)
            SELECT cutoff, max(window_end) AS latest_outcome, count(*) AS training_rows,
                   count_if(is_launch) AS first_entry_rows,
                   count_if(is_launch AND origin >= DATE '2025-12-01') AS matured_final_launch_rows
            FROM fits JOIN panels ON window_end < cutoff WHERE actual_units IS NOT NULL
            GROUP BY cutoff ORDER BY cutoff
        """, [FIT_DATES]).df()
        audit.to_csv(OUTPUT / "weekly_cutoffs.csv", index=False)
    for fit, next_fit in zip(FIT_DATES, FIT_DATES[1:] + [LAST_ENTRY + timedelta(days=1)]):
        for seed in SEEDS:
            out = OUTPUT / "weekly" / f"fit_{fit}" / f"seed_{seed}"
            if fit == CUTOFF:
                reference = OUTPUT / "main" / name / f"seed_{seed}"
                out.mkdir(parents=True, exist_ok=True)
                with duckdb.connect() as db:
                    db.from_parquet(str(reference / "predictions.parquet")).filter(
                        f"forecast_date < {base.date_sql(next_fit)}"
                    ).write_parquet(str(out / "predictions.parquet"), compression="zstd")
                base.write_json(out / "validation.json", {
                    "fit_date": fit, "seed": seed, "checkpoint": reference / "model.pt",
                    "first_issue": fit, "last_issue": next_fit - timedelta(days=1),
                })
            else:
                run_neural(out, features, name, seed, fit=fit, weekly=(fit, next_fit))


def strict():
    for name in WINDOWS:
        features = OUTPUT / "panels" / name
        out = OUTPUT / "strict" / name
        out.mkdir(parents=True, exist_ok=True)
        numeric = base.NUMERIC + (["first_week_sales", "first_week_complete"] if name == "week1_8_28" else [])
        columns = numeric + base.CATEGORICAL
        with duckdb.connect() as db:
            load(db, features)
            db.from_parquet(str(OUTPUT / "links.parquet")).create_view("links")
            training = out / "training.parquet"
            if not training.exists():
                db.sql(f"""SELECT {', '.join(columns)}, actual_units, colorway,
                           date_diff('day', DATE '1970-01-01', window_end)::INTEGER AS target_day
                           FROM training WHERE window_end < {base.date_sql(CUTOFF)} AND actual_units IS NOT NULL
                           ORDER BY window_end,origin,colorway,store""").write_parquet(str(training))
            matrix, labels, colorways, _ = read_training(training, columns, len(numeric), out / "matrix.float32")
            issues = db.execute("SELECT DISTINCT forecast_date FROM candidates ORDER BY 1").fetchall()
            base.check(db.execute("""
                SELECT count_if(l.groups IS NULL OR len(l.groups)=0)=0
                FROM (SELECT DISTINCT colorway FROM candidates) c LEFT JOIN links l USING(colorway)
            """).fetchall()[0][0], "A final candidate lacks released linkage")
            for issue, in issues:
                issue_dir = out / str(issue)
                issue_dir.mkdir(exist_ok=True)
                links = db.execute("""
                    SELECT list(DISTINCT linked) FROM candidates JOIN links USING(colorway),
                    UNNEST(groups) AS m(linked) WHERE forecast_date=?
                """, [issue]).fetchall()[0][0]
                base.check(bool(links), f"No held-out link groups on {issue}")
                allowed = db.execute("""SELECT colorway FROM links
                     WHERE len(groups)>0 AND NOT list_has_any(groups,?) ORDER BY colorway""", [links]).fetchnumpy()["colorway"]
                rows = np.flatnonzero(np.isin(colorways, allowed))
                digest = hashlib.sha256(json.dumps(sorted(links)).encode()).hexdigest()
                path = issue_dir / "control.parquet"
                if not path.exists():
                    scored = db.execute(f"""
                        SELECT {', '.join(columns)}, {', '.join(base.KEYS)}, origin AS observed_entry_date,
                               actual_units, product_seen_before_entry, colorway
                        FROM candidates WHERE forecast_date=? ORDER BY origin,colorway,store
                    """, [issue]).df()
                    if "first_week_sales" not in scored:
                        scored["first_week_sales"] = None
                    signal = scored["first_week_sales"].copy()
                    raw = scored[columns].fillna({col: -1 for col in base.CATEGORICAL}).to_numpy(dtype=np.float32)
                    scored["frozen_predicted_units"], model, maps = fit_on_disk(
                        matrix, labels, rows, raw, columns, len(numeric), base.SETTINGS,
                    )
                    scored = scored.drop(columns=columns)
                    scored["first_week_sales"] = signal
                    (issue_dir / "lightgbm.txt").write_text(model, encoding="utf-8")
                    base.write_json(issue_dir / "lightgbm_fit.json", {
                        "fit_date": CUTOFF, "issue": issue, "training_rows": len(rows),
                        "exclusion_sha256": digest, "category_maps": maps,
                    })
                    db.from_df(scored).write_parquet(str(path), compression="zstd")
                run_neural(issue_dir / "seed_13", features, name, 13, strict={
                    "issue": str(issue), "allowed": allowed, "training_rows": len(rows),
                    "exclusion_sha256": digest, "control": path,
                })
            del matrix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("main", "weekly", "strict"))
    step = parser.parse_args().step
    base.write_json(OUTPUT / "run_configuration.json", {
        "raw_history_start": base.FIRST_DAY, "frozen_fit": CUTOFF, "last_launch": LAST_ENTRY,
        "windows": WINDOWS, "offset_unit": "days from entry, zero-based",
        "weekly_fits": FIT_DATES, "main_and_weekly_seeds": SEEDS, "strict_seed": 13,
        "strict_exclusion": "issue-date union of linked groups; remove unmapped training colorways",
        "baseline": "released lifecycle store subcategory with category fallback",
        "baseline_week_one_weight": 0.75,
        "lightgbm": base.SETTINGS, "neural_head": TWO_PART,
        "neural_text_preparation": STRUCTURED_SIZES_PREPARATION,
        "product_numeric_bypass": False,
    })
    {"main": main, "weekly": weekly, "strict": strict}[step]()
    base.write_json(OUTPUT / f"{step}.done.json", {"step": step})
