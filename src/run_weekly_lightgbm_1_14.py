"""Compare expanding weekly fits with the first fit on development launches."""
import argparse
from datetime import date, timedelta
import json
import sys
from time import perf_counter

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from fetch_weekly_lightgbm_sources import CACHE, ROOT

FIRST_FIT = date(2025, 9, 1)
LAST_ENTRY = date(2025, 11, 30)
FIRST_DAY = date(2024, 11, 10)
FIT_DATES = [FIRST_FIT + timedelta(weeks=i) for i in range(13)]
LAST_DAY_OFFSET = timedelta(days=13)
NUMERIC = ["price", "price_missing", "is_launch", "lifecycle_day",
           "start_month", "start_iso_week", "start_day_of_week"]
CATEGORICAL = ["product_identity", "product_name", "color_identity",
               "business_line", "category", "store"]
FEATURES = NUMERIC + CATEGORICAL
MODEL_FEATURES = ["storeid" if name == "store" else name for name in FEATURES]
KEYS = ["model_product_id_type", "model_product_id", "color_identity_type",
        "color_identity_value", "storeid", "forecast_date"]
SETTINGS = dict(objective="poisson", n_estimators=50, num_leaves=15,
                learning_rate=0.05, min_child_samples=100, random_state=13,
                n_jobs=6, verbosity=-1, force_col_wise=True, deterministic=True)
OUTPUT = CACHE.parent


def peak_memory_mib():
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class MemoryCounters(ctypes.Structure):
            _fields_ = [("size", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                    "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile")
            ]

        counters = MemoryCounters()
        counters.size = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
        if not get_memory(get_process(), ctypes.byref(counters), counters.size):
            raise ctypes.WinError()
        return round(counters.peak_working_set / 2**20, 1)
    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / (2**20 if sys.platform == "darwin" else 2**10), 1)


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    temporary = path.with_suffix(".partial.json")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def date_sql(value):
    return f"DATE '{value.isoformat()}'"


def open_sources(work, supplements=None, *, sources=None, last_entry=LAST_ENTRY):
    db = duckdb.connect()
    db.execute("SET threads = 6")
    db.execute("SET memory_limit = '5GB'")
    db.execute("SET preserve_insertion_order = false")
    db.execute("SET temp_directory = ?", [str(work / "spill")])
    for name in ("lifecycle", "store", "membership"):
        file_name = "store_day" if name == "store" else name
        paths = [str(path) for path in sources[file_name]] if sources else [str(CACHE / f"{file_name}.parquet")]
        if sources is None and supplements is not None and name != "membership":
            paths.append(str(supplements / f"{file_name}.parquet"))
        db.from_parquet(paths).create_view(f"{name}_source")
    for source, keys in (
        ("store_source", "storeid,dt"),
        ("membership_source", "model_product_id_type,model_product_id,color_identity_type,color_identity_value"),
    ):
        check(db.execute(f"SELECT count(*) = count(DISTINCT ({keys})) FROM {source}").fetchall()[0][0],
              f"Duplicate {source} keys")
    check(db.execute("""
        SELECT count_if(
            observed_entry_date > ? OR window_start <> observed_entry_date
            OR prediction_cutoff <> observed_entry_date OR window_end <> window_start + 13
            OR (? = DATE '2025-11-30' AND evaluation_period NOT IN ('development', 'outside_evaluation_periods'))
        ) = 0 FROM membership_source WHERE observed_entry_date <= ?
    """, [last_entry, last_entry, last_entry]).fetchall()[0][0], "Invalid membership")
    definition = (ROOT / "sql/model_weekly_lightgbm_1_14.sql").read_text(encoding="utf-8")
    if sources:
        definition = definition.replace("DATE '2025-11-30'", date_sql(last_entry)).replace(
            "AND w.evaluation_period IN ('outside_evaluation_periods', 'development')",
            f"AND w.observed_entry_date <= {date_sql(last_entry)}",
        )
    else:
        check(db.execute("SELECT count_if(observed_entry_date > ?) = 0 FROM membership_source",
                         [last_entry]).fetchone()[0], "Final-test membership in development sources")
    db.execute(definition)
    check(db.execute("""
        SELECT count_if(b.colorway IS NULL OR b.name_count > 1 OR b.taxonomy_count <> 1) = 0
        FROM colorway_keys w LEFT JOIN entry_brief b USING (colorway)
        WHERE w.scope_status = 'preliminary_in_scope' AND w.calendar_mature
          AND w.observed_entry_date <= ?
    """, [last_entry]).fetchall()[0][0], "Missing or conflicting lifecycle entry brief")
    check(db.execute("""
        SELECT count_if(dt < observed_entry_date
                     OR lifecycle_day IS DISTINCT FROM date_diff('day', observed_entry_date, dt) + 1) = 0
        FROM daily
    """).fetchall()[0][0], "Lifecycle age disagrees with the released entry date")
    for start, end in db.execute("SELECT min(dt), max(dt) FROM daily GROUP BY date_trunc('month', dt)").fetchall():
        check(db.execute("""
            SELECT count(*) = count(DISTINCT (dt, colorway, store))
            FROM daily WHERE dt BETWEEN ? AND ?
        """, [start, end]).fetchall()[0][0], f"Duplicate lifecycle keys in {start:%Y-%m}")
    return db


def attach_outcomes(db, first, last, first_offset=0, last_offset=13, table="target"):
    start = date_sql(first + timedelta(days=first_offset))
    end = date_sql(last + timedelta(days=last_offset))
    width = last_offset - first_offset + 1
    db.execute("CREATE OR REPLACE TEMP TABLE target_origins AS SELECT DISTINCT origin FROM candidate")
    db.execute(f"""
        CREATE OR REPLACE TEMP TABLE {table} AS
        WITH events AS (SELECT DISTINCT colorway, origin FROM candidate),
        observed AS (
            SELECT e.colorway, d.store, e.origin, count(*) AS observed_days,
                   count_if(NOT coalesce(d.sales_label_complete, false)
                            OR d.reported_sales IS NULL) AS missing_measure_days,
                   sum(d.reported_sales) AS units
            FROM daily d JOIN events e ON d.colorway = e.colorway
             AND d.dt BETWEEN e.origin + {first_offset} AND e.origin + {last_offset}
            WHERE d.dt BETWEEN {start} AND {end}
            GROUP BY ALL
        )
        SELECT c.colorway, c.store, c.origin, c.origin + {last_offset} AS window_end,
               coalesce(s.recorded_days, 0) AS recorded_days,
               coalesce(s.healthy_days, 0) AS healthy_days,
               coalesce(s.usable_days, 0) AS usable_days,
               coalesce(o.observed_days, 0) AS observed_days,
               coalesce(o.missing_measure_days, 0) AS missing_measure_days,
               CASE WHEN (CASE WHEN c.is_launch THEN s.healthy_days ELSE s.usable_days END) = {width}
                         AND coalesce(o.missing_measure_days, 0) = 0
                    THEN coalesce(o.units, 0)::BIGINT END AS actual_units
        FROM candidate c
        LEFT JOIN covered('target_origins', {first_offset}, {last_offset}) s USING (origin, store)
        LEFT JOIN observed o USING (colorway, store, origin)
    """)


def validate_candidates(db, last_entry=LAST_ENTRY):
    check(db.execute("""
        SELECT count(*) = count(DISTINCT (colorway, store, origin))
            AND coalesce(count_if(
                origin > ? OR observed_entry_date > origin
                OR evidence_date IS NULL OR eligibility_date IS NULL
                OR (is_launch AND (origin <> observed_entry_date OR evidence_date <> origin
                                   OR eligibility_date > origin))
                OR (NOT is_launch AND (observed_entry_date >= origin
                   OR evidence_date >= origin OR eligibility_date >= origin
                   OR evidence_date < origin - 90 OR eligibility_date < origin - 90))
                OR price_date > evidence_date OR name_date > evidence_date
                OR (price_date IS NULL) <> (price IS NULL)
            ), 0) = 0
        FROM candidate
    """, [last_entry]).fetchall()[0][0], "Duplicate keys or unavailable candidate features")
    dates = db.execute("""
        SELECT DISTINCT origin, start_month, start_iso_week, start_day_of_week
        FROM model_inputs('candidate')
    """).fetchall()
    check(all(month == day.month and week == day.isocalendar().week and dow == day.isoweekday()
              for day, month, week, dow in dates), "Unexpected calendar features")


def build_target(db, first, last, first_offset=0, last_offset=13, table="target"):
    attach_outcomes(db, first, last, first_offset, last_offset, table)
    width = last_offset - first_offset + 1
    check(db.execute(f"""
        SELECT count(*) = (SELECT count(*) FROM candidate)
           AND coalesce(count_if((actual_units IS NOT NULL) IS DISTINCT FROM
                   ((CASE WHEN c.is_launch THEN healthy_days ELSE usable_days END) = {width}
                             AND missing_measure_days = 0)
                OR (actual_units IS NOT NULL AND observed_days = 0 AND actual_units <> 0)
                OR actual_units < 0), 0) = 0
        FROM {table} t JOIN candidate c USING (colorway, store, origin)
    """).fetchall()[0][0], "Invalid target keys, coverage, or zero-unit labels")


def prepare_inputs(db, first, last, first_offset, week_one):
    if week_one:
        build_target(db, first, last, 0, 6, "first_week")
    signal = (", w.actual_units AS first_week_sales, "
              "w.actual_units IS NOT NULL AS first_week_complete" if week_one else "")
    join = "LEFT JOIN first_week w USING (colorway, store, origin)" if week_one else ""
    db.execute(f"""
        CREATE OR REPLACE TEMP VIEW window_inputs AS
        SELECT c.*{signal} FROM model_inputs('candidate', {first_offset}) c {join}
    """)
    dates = db.execute("""
        SELECT DISTINCT origin, start_month, start_iso_week, start_day_of_week FROM window_inputs
    """).fetchall()
    for origin, month, week, dow in dates:
        start = origin + timedelta(days=first_offset)
        check((month, week, dow) == (start.month, start.isocalendar().week, start.isoweekday()),
              "Calendar features disagree with target-window start")


def cache_training(db, work, fit_date, previous, anchors, first_offset=0, last_offset=13,
                   week_one=False, retain_keys=False):
    started = perf_counter()
    last_day_offset = timedelta(days=last_offset)
    matured = [
        origin for origin in anchors
        if previous <= origin + last_day_offset < fit_date
    ]
    established = (ROOT / "sql/model_weekly_lightgbm_1_14_established.sql").read_text(encoding="utf-8")
    selections = [(origin, origin, established, [origin]) for origin in matured]
    launch_start, launch_end = previous - last_day_offset, fit_date - timedelta(days=last_offset + 1)
    launches = """
        SELECT * FROM launch_inputs WHERE origin BETWEEN CAST(? AS DATE) AND CAST(? AS DATE)
    """
    selections.append((launch_start, launch_end, launches, [launch_start, launch_end]))
    parts, counts = [], []
    for first, last, query, params in selections:
        db.execute(f"CREATE OR REPLACE TEMP TABLE candidate AS {query}", params)
        if db.execute("SELECT count(*) FROM candidate").fetchall()[0][0] == 0:
            continue
        validate_candidates(db)
        build_target(db, first, last, first_offset, last_offset)
        counts.extend(db.execute("""
            SELECT strftime(window_end, '%Y-%m') AS month, c.is_launch,
                   count(*) AS eligible, count(actual_units) AS training_rows,
                   count_if(actual_units = 0) AS zero_rows,
                   count_if(actual_units = 0 AND observed_days = 0) AS absent_zero_rows,
                   min(window_end) FILTER (WHERE actual_units IS NOT NULL) AS first_outcome,
                   max(window_end) FILTER (WHERE actual_units IS NOT NULL) AS last_outcome
            FROM target JOIN candidate c USING (colorway, store, origin)
            GROUP BY ALL ORDER BY month
        """).fetchdf().to_dict("records"))
        check(db.execute("SELECT count_if(window_end >= ?) = 0 FROM target", [fit_date]).fetchall()[0][0],
              "A label matured on or after its fit cutoff")
        if db.execute("SELECT count(actual_units) FROM target").fetchall()[0][0] == 0:
            continue
        prepare_inputs(db, first, last, first_offset, week_one)
        features = NUMERIC + (["first_week_sales", "first_week_complete"] if week_one else []) + CATEGORICAL
        part = work / f"training_{fit_date}_{len(parts):02}.parquet"
        keys = ", c.colorway, t.window_end" if retain_keys else ""
        db.sql(f"""
            SELECT {', '.join('c.' + f for f in features)}, t.actual_units{keys}
            FROM window_inputs c JOIN target t USING (colorway, store, origin)
            WHERE actual_units IS NOT NULL ORDER BY c.origin, c.colorway, c.store
        """).write_parquet(str(part), compression="zstd")
        parts.append(part)
    return parts, counts, perf_counter() - started


def map_categories(values, numeric=NUMERIC):
    vocabularies = []
    for idx in range(len(numeric), values.shape[1]):
        column = values[:, idx]
        codes = np.bincount(column[column >= 0].astype(np.int32)).nonzero()[0]
        mapping = np.full(int(codes[-1]) + 2 if len(codes) else 1, -1, dtype=np.int32)
        mapping[codes + 1] = np.arange(len(codes))
        column[:] = mapping[column.astype(np.int32) + 1]
        vocabularies.append(mapping)
    return vocabularies


def predict(model, mapping, raw, numeric=NUMERIC):
    categorical_names = MODEL_FEATURES[len(NUMERIC):]
    if not len(raw):
        return np.empty(0), dict.fromkeys(categorical_names, 0)
    values = raw.copy()
    unseen = {}
    for idx, vocabulary in enumerate(mapping, start=len(numeric)):
        codes = values[:, idx].astype(np.int32)
        known_range = codes < len(vocabulary) - 1
        encoded = np.full(len(values), -1, dtype=np.int32)
        encoded[known_range] = vocabulary[codes[known_range] + 1]
        unseen[categorical_names[idx - len(numeric)]] = int(((codes >= 0) & (encoded < 0)).sum())
        values[:, idx] = encoded
    booster = model if isinstance(model, lgb.Booster) else model.booster_
    scores = booster.predict(values)
    check(np.isfinite(scores).all() and (scores >= 0).all(), "Invalid predictions")
    return scores, unseen


def columns_to_numpy(batch, matrix, features):
    for idx, column in enumerate(features):
        values = batch[idx].to_numpy(zero_copy_only=False)
        if column in CATEGORICAL and batch[idx].null_count:
            values = np.nan_to_num(values, nan=-1)
        matrix[:, idx] = values


def fit_model(db, parts, count, numeric=NUMERIC):
    features = numeric + CATEGORICAL
    model_features = numeric + MODEL_FEATURES[len(NUMERIC):]
    started = perf_counter()
    values = np.empty((count, len(features)), dtype=np.float32, order="F")
    labels = np.empty(count, dtype=np.float32)
    offset = 0
    for part in parts:
        reader = db.from_parquet(str(part)).to_arrow_reader(250_000)
        for batch in reader:
            stop = offset + len(batch)
            columns_to_numpy(batch, values[offset:stop], features)
            labels[offset:stop] = batch[len(features)].to_numpy()
            offset = stop
    check(offset == count, "Training cache does not match the audited row count")
    mapping = map_categories(values, numeric)
    assembly_seconds = perf_counter() - started
    started = perf_counter()
    model = lgb.LGBMRegressor(**SETTINGS)
    model.fit(values, labels, feature_name=model_features,
              categorical_feature=list(range(len(numeric), len(features))))
    check(model.feature_name_ == model_features, "Unexpected LightGBM features")
    return model, mapping, assembly_seconds, perf_counter() - started


def issue_forecasts(db, fit_date, next_fit, model, mapping, first_model, first_mapping):
    end = min(next_fit - timedelta(days=1), LAST_ENTRY)
    db.execute("""
        CREATE OR REPLACE TEMP TABLE candidate AS SELECT * FROM launch_inputs
        WHERE origin BETWEEN ? AND ?
    """, [fit_date, end])
    validate_candidates(db)
    frame = db.execute("""
        SELECT c.*, w.model_product_id_type, w.model_product_id,
               w.color_identity_type, w.color_identity_value, s.storeid,
               w.product_seen_before_entry
        FROM model_inputs('candidate') c JOIN colorway_keys w USING (colorway)
        JOIN store_keys s USING (store)
        ORDER BY c.origin, c.colorway, c.store
    """).fetchdf()
    days = frame.origin.dt
    check((frame.lifecycle_day == 1).all()
          and (frame.start_month == days.month).all()
          and (frame.start_iso_week == days.isocalendar().week).all()
          and (frame.start_day_of_week == days.day_of_week + 1).all(),
          "Launch lifecycle age or calendar features are incorrect")
    raw = frame[FEATURES].to_numpy(dtype=np.float32, na_value=np.nan)
    raw[:, len(NUMERIC):] = np.nan_to_num(raw[:, len(NUMERIC):], nan=-1)
    issued = frame.rename(columns={"origin": "forecast_date"})[KEYS].copy()
    issued["weekly_fit_cutoff"] = pd.Timestamp(fit_date)
    issued["frozen_fit_cutoff"] = pd.Timestamp(FIRST_FIT)
    started = perf_counter()
    issued["weekly_predicted_units"], weekly_unseen = predict(model, mapping, raw)
    weekly_seconds = perf_counter() - started
    started = perf_counter()
    if fit_date == FIRST_FIT:
        issued["frozen_predicted_units"], frozen_unseen = issued.weekly_predicted_units, weekly_unseen
        frozen_seconds = weekly_seconds
    else:
        issued["frozen_predicted_units"], frozen_unseen = predict(first_model, first_mapping, raw)
        frozen_seconds = perf_counter() - started
    check(not issued.duplicated(KEYS).any(), "Duplicate issued forecast keys")
    return frame, issued, {"weekly": weekly_unseen, "frozen": frozen_unseen}, weekly_seconds, frozen_seconds


def score(db, work, frame, issued, first, last):
    build_target(db, first, last)
    target = db.execute("SELECT * FROM target").fetchdf()
    outcomes = frame[["colorway", "store", "origin", "product_seen_before_entry"]].merge(
        target, on=["colorway", "store", "origin"], validate="one_to_one"
    )
    check(len(outcomes) == len(frame), "Target is missing an issued key")
    columns = ["observed_days", "recorded_days", "healthy_days", "usable_days",
               "missing_measure_days", "actual_units", "product_seen_before_entry"]
    check(outcomes.product_seen_before_entry.notna().all(), "Missing product cohort")
    scored = pd.concat([issued.reset_index(drop=True), outcomes[columns]], axis=1)
    scored.to_parquet(work / f"score_{first}.parquet", index=False)


def summarize(work, fits):
    db = duckdb.connect()
    forecast_files = [str(work / f"forecast_{fit['fit_cutoff']}.parquet") for fit in fits]
    score_files = [str(work / f"score_{fit['fit_cutoff']}.parquet") for fit in fits]
    db.from_parquet(forecast_files).create_view("issued")
    db.from_parquet(score_files).create_view("scored")
    key_sql = ", ".join(KEYS)
    check(db.execute(f"""
        SELECT count(*) = count(DISTINCT ({key_sql}))
           AND count_if(weekly_fit_cutoff > forecast_date
                     OR forecast_date >= weekly_fit_cutoff + INTERVAL 7 DAY
                     OR frozen_fit_cutoff <> DATE '2025-09-01'
                     OR (weekly_fit_cutoff = frozen_fit_cutoff
                         AND weekly_predicted_units <> frozen_predicted_units)) = 0
        FROM issued
    """).fetchall()[0][0], "Incorrect weekly forecast cutoff or mismatched control")
    check(db.execute(f"""
        SELECT count(*) = count(DISTINCT ({key_sql})) AND count(*) = (SELECT count(*) FROM issued)
        FROM scored
    """).fetchall()[0][0], "Scoring population does not match issued forecast grain")
    db.sql("SELECT * FROM issued ORDER BY forecast_date").write_parquet(str(work / "forecasts.parquet"))
    db.sql("SELECT * FROM scored ORDER BY forecast_date").write_parquet(str(work / "matched_outcomes.parquet"))
    metrics = (ROOT / "sql/model_weekly_lightgbm_1_14_metrics.sql").read_text(encoding="utf-8")
    db.sql(metrics).write_csv(str(work / "metrics.csv"), header=True)


def regular_origins():
    anchor = FIRST_FIT - timedelta(days=90)
    while anchor - timedelta(days=14) >= FIRST_DAY:
        anchor -= timedelta(days=14)
    return [anchor + timedelta(days=14 * i)
            for i in range((FIT_DATES[-1] - anchor).days // 14 + 1)]


def run(limit):
    work = OUTPUT / ("pilot" if limit == 2 else "run")
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    db = open_sources(work)
    source_seconds = perf_counter() - started
    print(f"Verified sparse source and membership in {source_seconds:.1f}s", flush=True)
    anchors = regular_origins()
    parts, fits = [], []
    previous = FIRST_DAY
    first_model, first_mapping = None, None
    for index, fit_date in enumerate(FIT_DATES[:limit]):
        new_parts, audit, group_seconds = cache_training(db, work, fit_date, previous, anchors)
        parts.extend(new_parts)
        if index == 0:
            check(audit, "No candidate windows before the September 1 fit")
            counts = pd.DataFrame(audit)
        elif audit:
            counts = pd.concat([counts, pd.DataFrame(audit)], ignore_index=True)
        total = int(counts.training_rows.sum())
        new_count = sum(row["training_rows"] for row in audit)
        print(f"{fit_date}: grouped {new_count:,} newly mature "
              f"rows in {group_seconds:.1f}s, fitting {total:,} rows", flush=True)
        model, mapping, matrix_seconds, fit_seconds = fit_model(db, parts, total)
        if index == 0:
            first_model, first_mapping = model, mapping
        score_started = perf_counter()
        next_fit = FIT_DATES[index + 1] if index + 1 < len(FIT_DATES) else LAST_ENTRY + timedelta(days=1)
        frame, issued, unseen, weekly_predict, frozen_predict = issue_forecasts(
            db, fit_date, next_fit, model, mapping, first_model, first_mapping
        )
        issued.to_parquet(work / f"forecast_{fit_date}.parquet", index=False)
        predict_seconds = perf_counter() - score_started
        score(db, work, frame, issued, fit_date, next_fit - timedelta(days=1))
        months = pd.period_range(FIRST_DAY, fit_date - timedelta(days=1), freq="M").astype(str)
        month_counts = counts.groupby("month")[
            ["eligible", "training_rows", "zero_rows", "absent_zero_rows"]
        ].sum().reindex(months, fill_value=0).rename_axis("month").reset_index()
        fit = dict(
            fit_cutoff=str(fit_date), training_rows=total,
            latest_label=counts.last_outcome.max(), group_seconds=group_seconds,
            matrix_seconds=matrix_seconds, fit_seconds=fit_seconds,
            prediction_seconds=predict_seconds, weekly_predict_seconds=weekly_predict,
            frozen_predict_seconds=frozen_predict, peak_process_mib=peak_memory_mib(),
            issued_rows=len(issued), unseen_category_rows=unseen,
            training_by_outcome_month=month_counts.to_dict("records"), newly_matured=audit,
        )
        fits.append(fit)
        model.booster_.save_model(str(work / f"model_{fit_date}.txt"))
        np.savez_compressed(work / f"categories_{fit_date}.npz",
                            **dict(zip(MODEL_FEATURES[len(NUMERIC):], mapping)))
        write_json(work / "runtime.json", dict(
            fits=fits, features=MODEL_FEATURES,
            settings=SETTINGS, lightgbm=lgb.__version__, duckdb=duckdb.__version__,
            source_seconds=source_seconds, elapsed_seconds=perf_counter() - started,
        ))
        print(f"{fit_date}: fit in {fit_seconds:.1f}s, issued {len(issued):,} forecasts, "
              f"peak {fit['peak_process_mib']:.0f} MiB", flush=True)
        if index == 1:
            summarize(work, fits)
            print("September 1 and 8 validation passed", flush=True)
        previous = fit_date
    db.close()
    summarize(work, fits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fits", type=int, choices=(2, 13), default=13)
    run(parser.parse_args().fits)
