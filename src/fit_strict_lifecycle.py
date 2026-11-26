"""Fit date-wide linked-product holdouts on the prepared lifecycle examples."""
import argparse
from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

EPOCH = date(1970, 1, 1)
SCHEDULES = ("frozen", "weekly")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def feature_matrix(record, features, numeric_count, result):
    for idx, column in enumerate(features):
        values = record[column].to_numpy(zero_copy_only=False)
        if idx >= numeric_count:
            values = np.nan_to_num(values, nan=-1)
        result[:, idx] = values


def read_training(path, features, numeric_count, disk_matrix=None):
    file = pq.ParquetFile(path)
    rows = file.metadata.num_rows
    if disk_matrix is None:
        values = np.empty((rows, len(features)), dtype=np.float32, order="F")
    else:
        values = np.memmap(disk_matrix, dtype=np.float32, mode="w+", shape=(rows, len(features)))
    targets = np.empty(rows, dtype=np.float32)
    colorways = np.empty(rows, dtype=np.int32)
    days = np.empty(rows, dtype=np.int32)
    offset = 0
    for batch in file.iter_batches(250_000, columns=features + ["actual_units", "colorway", "target_day"]):
        end = offset + batch.num_rows
        feature_matrix(batch, features, numeric_count, values[offset:end])
        targets[offset:end] = batch["actual_units"].to_numpy()
        colorways[offset:end] = batch["colorway"].to_numpy()
        days[offset:end] = batch["target_day"].to_numpy()
        offset = end
    require(offset == rows and (days[:-1] <= days[1:]).all(), "Training rows are missing or unsorted")
    if disk_matrix is not None:
        values.flush()
    return values, targets, colorways, days


def category_map(codes):
    present = np.bincount(codes[codes >= 0]).nonzero()[0]
    mapping = np.full(int(present[-1]) + 2 if len(present) else 1, -1, dtype=np.int32)
    mapping[present + 1] = np.arange(len(present))
    return mapping


def fit_predict(train, labels, raw, features, numeric_count, settings):
    mappings = []
    for idx in range(numeric_count, len(features)):
        src = train[:, idx].astype(np.int32)
        mapping = category_map(src)
        train[:, idx] = mapping[src + 1]
        mappings.append(mapping)
    columns = [name if name != "store" else "storeid" for name in features]
    model = lgb.LGBMRegressor(**settings)
    model.fit(train, labels, feature_name=columns,
              categorical_feature=list(range(numeric_count, len(columns))))
    return predict(model.booster_, mappings, raw, columns, numeric_count)


class LinkedTrainingSequence(lgb.Sequence):
    batch_size = 65_536

    def __init__(self, matrix, rows, maps, numeric_count):
        self.matrix = matrix
        self.rows = rows
        self.maps = maps
        self.numeric_count = numeric_count

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        values = self.matrix[self.rows[index]]
        if values.ndim == 1:
            values = values.astype(np.float64)
        for idx, mapping in enumerate(self.maps, start=self.numeric_count):
            values[..., idx] = mapping[values[..., idx].astype(np.int32) + 1]
        return values


def fit_on_disk(train, labels, rows, raw, features, numeric_count, settings):
    maps = [category_map(train[rows, idx].astype(np.int32)) for idx in range(numeric_count, len(features))]
    columns = [name if name != "store" else "storeid" for name in features]
    data = lgb.Dataset(LinkedTrainingSequence(train, rows, maps, numeric_count), label=labels[rows],
                       feature_name=columns, categorical_feature=list(range(numeric_count, len(columns))))
    params = dict(settings)
    rounds = params.pop("n_estimators")
    params["num_threads"] = params.pop("n_jobs")
    params["seed"] = params.pop("random_state")
    booster = lgb.train(params, data, num_boost_round=rounds)
    return predict(booster, maps, raw, columns, numeric_count)


def predict(booster, mappings, raw, columns, numeric_count):
    require(booster.feature_name() == columns, "Saved model features differ from the manifest")
    values = raw.copy()
    for idx, mapping in enumerate(mappings, start=numeric_count):
        code = values[:, idx].astype(np.int32)
        in_range = code < len(mapping) - 1
        encoded = np.full(len(values), -1, dtype=np.int32)
        encoded[in_range] = mapping[code[in_range] + 1]
        values[:, idx] = encoded
    predictions = booster.predict(values)
    require(np.isfinite(predictions).all() and (predictions >= 0).all(), "Invalid strict forecast")
    return predictions, booster.model_to_string(), [m.tolist() for m in mappings]


def run_shard(work, result_root, shard, shards, max_dates, disk_matrix):
    start = perf_counter()
    m = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    features = m["features"]
    num = len(m["numeric"])
    fit_dates = [date.fromisoformat(day) for day in m["fit_dates"]]
    link_table = pq.read_table(work / "links.parquet").to_pylist()
    links = {r["colorway"]: set(r["groups"] or []) for r in link_table}
    by_group = defaultdict(list)
    linked = np.zeros(max(links) + 1, dtype=bool)
    for colorway, groups in links.items():
        linked[colorway] = bool(groups)
        for group in groups:
            by_group[group].append(colorway)
    table = pq.read_table(work / "candidates.parquet")
    days = table["forecast_date"].combine_chunks().cast(pa.int32()).to_numpy()
    require(len(days) == m["expected_rows"] and (days[:-1] <= days[1:]).all(),
            "Forecast panel is missing or unsorted")
    issue_days, offsets, sizes = np.unique(days, return_index=True, return_counts=True)
    selected = list(zip(issue_days, offsets, sizes))[shard::shards]
    if max_dates is not None:
        selected = selected[:max_dates]
    result_root.mkdir(parents=True, exist_ok=True)
    train, targets, history_colors, matured_days = read_training(work / "training.parquet", features, num, disk_matrix)
    require(len(targets) == m["training_rows"], "Training manifest row count mismatch")
    print(f"{m['name']} shard {shard + 1}/{shards}: {len(selected)} dates, "
          f"{len(targets):,} source rows, loaded in {perf_counter()-start:.1f}s", flush=True)
    for issue_day, offset, count in selected:
        issue = date.fromordinal(EPOCH.toordinal() + int(issue_day))
        file = result_root / f"forecast_{issue}.parquet"
        artifact = result_root / f"models_{issue}.parquet"
        receipt = result_root / f"done_{issue}.json"
        if receipt.exists():
            require(pq.ParquetFile(file).metadata.num_rows == count
                    and pq.ParquetFile(artifact).metadata.num_rows == 2, "Invalid existing strict output")
            continue
        batch = table.slice(int(offset), int(count))
        colors = set(batch["colorway"].to_pylist())
        require(all(links[color] for color in colors), "An issued colorway has no link group")
        heldout = set().union(*(links[color] for color in colors))
        blocked = {code for group in heldout for code in by_group[group]}
        allow = linked.copy()
        allow[list(blocked)] = False
        require(not allow[list(colors)].any(), "A held-out colorway entered its own fit")
        raw = np.empty((count, len(features)), dtype=np.float32)
        feature_matrix(batch, features, num, raw)
        retained = [col for col in batch.column_names if col not in features or col.startswith("first_week_")]
        issued = batch.select(retained)
        artifacts = []
        for schedule, cutoff in zip(SCHEDULES, (fit_dates[0], max(day for day in fit_dates if day <= issue))):
            started = perf_counter()
            stop = int(np.searchsorted(matured_days, (cutoff-EPOCH).days, side="left"))
            colors_before_fit = history_colors[:stop]
            kept = allow[colors_before_fit]
            used = int(kept.sum())
            require(used >= 2 and not allow[list(blocked)].any(), "Invalid strict training panel")
            latest = date.fromordinal(EPOCH.toordinal() + int(matured_days[stop-1-int(kept[::-1].argmax())]))
            require(latest < cutoff, "Training outcome reaches the cutoff")
            if disk_matrix is None:
                pred, text, mappings = fit_predict(train[:stop][kept], targets[:stop][kept], raw,
                                                   features, num, m["settings"])
            else:
                rows = np.flatnonzero(kept).astype(np.int32)
                pred, text, mappings = fit_on_disk(train, targets, rows, raw, features, num, m["settings"])
            issued = issued.append_column(f"{schedule}_fit_cutoff", pa.repeat(pa.scalar(cutoff), len(issued)))
            issued = issued.append_column(f"{schedule}_predicted_units", pa.array(pred))
            artifacts.append(dict(forecast_date=issue, schedule=schedule, fit_cutoff=cutoff,
                                  latest_target_end=latest, training_rows=used, prior_rows=stop,
                                  heldout_groups=len(heldout), excluded_colorways=len(blocked),
                                  exclusion_sha256=hashlib.sha256(json.dumps(sorted(heldout)).encode()).hexdigest(),
                                  fit_seconds=perf_counter()-started, model_text=text,
                                  features=features, mappings=mappings))
        pq.write_table(pa.Table.from_pylist(artifacts), artifact, compression="zstd")
        pq.write_table(issued, file, compression="zstd")
        require(pq.ParquetFile(file).metadata.num_rows == count, "Forecast write lost rows")
        receipt.write_text(json.dumps({"rows": int(count), "fit_seconds": [a["fit_seconds"] for a in artifacts]}),
                           encoding="utf-8")
        print(f"{m['name']} {issue}: {count:,} rows, {len(heldout)} held groups, "
              f"fits {artifacts[0]['fit_seconds']:.1f}s/{artifacts[1]['fit_seconds']:.1f}s", flush=True)
    print(f"Finished {m['name']} shard {shard + 1}/{shards}", flush=True)


def run(work, result_root, shard, shards, max_dates=None, disk_backed=False):
    if disk_backed:
        with TemporaryDirectory() as scratch:
            run_shard(work, result_root, shard, shards, max_dates, Path(scratch) / "training.bin")
    else:
        run_shard(work, result_root, shard, shards, max_dates, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prepared", type=Path)
    parser.add_argument("results", type=Path)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-dates", type=int)
    args = parser.parse_args()
    require(0 <= args.shard < args.shards, "Invalid shard")
    run(args.prepared, args.results, args.shard, args.shards, args.max_dates)
