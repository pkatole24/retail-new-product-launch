"""Pilot or fit one frozen two-tower prelaunch days 1–14 checkpoint."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from run_weekly_lightgbm_1_14 import KEYS, OUTPUT, check, peak_memory_mib, write_json
from two_tower_features_1_14 import PRODUCT_CATEGORICAL, PRODUCT_NUMERIC, STORE_NUMERIC
from two_tower_text_1_14 import (
    COMBINED_PREPARATION, ORDERED_PREPARATION, STRUCTURED_SIZES_PREPARATION,
    SUBCATEGORY_PREPARATION, TEXT_PREPARATION,
    colorway_text, vectors,
)

SEED = 13
BATCH_SIZE = 128
FULL_EPOCHS = 8
PILOT_GROUPS = 1024
LEARNING_RATE = 0.001
DIRECT = "direct_store_units"
TWO_PART = "two_part"
INTERACTION = "two_part_interaction"


class Tower(nn.Module):
    def __init__(self, numeric, categories):
        super().__init__()
        self.embeddings = nn.ModuleList(nn.Embedding(size, 8, padding_idx=0) for size in categories)
        self.projection = nn.Sequential(nn.Linear(numeric + 8 * len(categories), 64),
                                        nn.ReLU(), nn.Linear(64, 32))

    def forward(self, numeric, codes):
        embedded = [embedding(codes[..., i]) for i, embedding in enumerate(self.embeddings)]
        return self.projection(torch.cat([numeric, *embedded], dim=-1))


class TwoTower(nn.Module):
    def __init__(self, product_size, product_levels, store_size, store_levels, unit_scale, head=DIRECT,
                 product_numeric_bypass=0):
        super().__init__()
        self.product_numeric_bypass = product_numeric_bypass
        self.product = Tower(product_size, product_levels)
        self.store = Tower(store_size, store_levels)
        self.head = head
        self.forecast = nn.Sequential(nn.Linear((96 if head == INTERACTION else 64) + product_numeric_bypass, 64),
                                      nn.ReLU(), nn.Linear(64, 1 if head == DIRECT else 2))
        self.register_buffer("unit_scale", torch.tensor(unit_scale, dtype=torch.float32))

    def outputs(self, product, product_codes, store, store_codes):
        if self.product_numeric_bypass:
            product, numeric = product.split(
                [product.shape[-1] - self.product_numeric_bypass, self.product_numeric_bypass], dim=-1)
        product_vector = self.product(product, product_codes)
        store_vector = self.store(store, store_codes)
        product_by_store = product_vector[:, None, :].expand_as(store_vector)
        features = [product_by_store, store_vector]
        if self.head == INTERACTION:
            features.append(product_by_store * store_vector)
        if self.product_numeric_bypass:
            features.append(numeric[:, None, :].expand(-1, store_vector.shape[1], -1))
        return self.forecast(torch.cat(features, dim=-1))

    def predictions(self, outputs):
        if self.head == DIRECT:
            return {"two_tower_units": F.softplus(outputs[..., 0]) * self.unit_scale}
        probability = outputs[..., 0].sigmoid()
        positive_units = 1 + F.softplus(outputs[..., 1]) * self.unit_scale
        return {"two_tower_units": probability * positive_units,
                "probability_positive": probability, "expected_positive_units": positive_units}

    def forward(self, *inputs):
        return self.predictions(self.outputs(*inputs))["two_tower_units"]

    def loss(self, outputs, targets, covered):
        if self.head == DIRECT:
            return covered_loss(self.predictions(outputs)["two_tower_units"], targets, covered)
        units = targets[covered]
        logits, raw_excess = outputs[covered].unbind(dim=-1)
        occurs = units > 0
        occurrence_loss = F.binary_cross_entropy_with_logits(logits, occurs.float(), reduction="sum")
        # A shifted Poisson puts the conditional unit distribution on positive counts.
        excess_units = units[occurs] - 1
        mean_excess = (F.softplus(raw_excess[occurs]) * self.unit_scale).clamp_min(1e-8)
        positive_loss = mean_excess - excess_units * mean_excess.log() + (excess_units + 1).lgamma()
        return (occurrence_loss / covered.sum().clamp_min(1)
                + positive_loss.sum() / occurs.sum().clamp_min(1))


def covered_loss(forecast, targets, covered):
    mean = forecast[covered].clamp_min(1e-8)
    units = targets[covered]
    poisson = mean - units * mean.log() + (units + 1).lgamma()
    return poisson.sum() / covered.sum().clamp_min(1)


class Panels:
    def __init__(self, files, group_count, week_one=False):
        self.starts = np.zeros(group_count, dtype=np.int64)
        self.widths = np.zeros(group_count, dtype=np.int64)
        fields = {name: [] for name in ("store_index", "actual_units", "category_mix")}
        signals = []
        offset = 0
        for path in files:
            frame = pd.read_parquet(path)
            keys, starts, widths = np.unique(frame.product_index.to_numpy(), return_index=True,
                                             return_counts=True)
            check(not self.widths[keys].any(), f"Forecast panel split across files at {path.name}")
            self.starts[keys] = starts + offset
            self.widths[keys] = widths
            fields["store_index"].append(frame.store_index.to_numpy(dtype=np.int32))
            fields["actual_units"].append(frame.actual_units.to_numpy(dtype=np.float32))
            fields["category_mix"].append(frame.category_mix.to_numpy(dtype=np.float32))
            if week_one:
                signal = frame.first_week_sales.to_numpy(dtype=np.float32)
                check(np.array_equal(np.isfinite(signal), frame.first_week_complete.to_numpy()),
                      "First-week sales disagree with their completeness mask")
                signals.append(np.log1p(signal))
            offset += len(frame)
        self.store_index, self.units, self.mix = (np.concatenate(x) for x in fields.values())
        self.week_one_sales = np.concatenate(signals) if week_one else None

    def indexes(self, keys):
        widths = self.widths[keys]
        step = np.arange(widths.max())
        exists = step[None, :] < widths[:, None]
        rows = self.starts[keys, None] + np.where(exists, step, 0)
        return rows, exists

    def selected_rows(self, keys):
        for start in range(0, len(keys), 512):
            rows, exists = self.indexes(keys[start:start + 512])
            yield rows[exists]


def numeric(frame, columns, training, log_columns=()):
    values = frame[list(columns)].to_numpy(dtype=np.float32, copy=True)
    for name in log_columns:
        index = columns.index(name)
        values[:, index] = np.log1p(np.maximum(values[:, index], 0))
    missing = ~np.isfinite(values)
    values[missing] = np.nan
    sample = values[training]
    valid_count = np.maximum(np.isfinite(sample).sum(axis=0), 1)
    mean = np.nansum(sample, axis=0) / valid_count
    centered = np.where(np.isfinite(sample), sample - mean, 0)
    scale = np.sqrt((centered ** 2).sum(axis=0) / valid_count)
    scale[scale < 1e-6] = 1
    output = np.where(missing, 0, (values - mean) / scale)
    return np.column_stack([output, missing]).astype(np.float32), {
        "columns": columns, "log1p": log_columns, "impute_and_center": mean.tolist(),
        "scale": scale.tolist(), "append_missing_flags": True,
    }


def categories(frame, columns, training):
    levels = {name: sorted(frame[name].iloc[training].dropna().unique().tolist()) for name in columns}
    codes = np.column_stack([pd.Index(levels[name]).get_indexer(frame[name]) + 1 for name in columns])
    return codes, levels


def sizes(values, training):
    levels = sorted({str(label) for ix in training for label in (
        values[ix] if values[ix] is not None else ())})
    indexes = {label: ix + 1 for ix, label in enumerate(levels)}
    result = np.zeros((len(values), 1 + len(levels)), dtype=np.float32)
    for row, labels in enumerate(values):
        for label in labels if labels is not None else ():
            result[row, indexes.get(str(label), 0)] = 1
    return result, levels


def prepare(work, pilot, device, text_preparation=TEXT_PREPARATION, seed=SEED,
            replay_encoder_initialization=False, strict=None, weekly=None, bypass_product_numeric=False,
            final_fit=None):
    scope = json.loads((work / "panel_validation.json").read_text())
    week_one = scope.get("issue_offset") == 7
    training_rows = (scope["prepared_training_rows"] if final_fit and "prepared_training_rows" in scope
                     else scope["frozen_fit_audit"]["training_rows"])
    development_rows = json.loads((work / "feature_validation.json").read_text()).get(
        "development_covered_rows", 1_120_957)
    combined_product_text = text_preparation != TEXT_PREPARATION
    structured_sizes = text_preparation in (TEXT_PREPARATION, STRUCTURED_SIZES_PREPARATION)
    products = pd.read_parquet(work / "products.parquet")
    stores = pd.read_parquet(work / "stores.parquet")
    origins = pd.to_datetime(products.origin)
    in_development = (products.filename.str.endswith("development.parquet") if weekly or final_fit
                      else origins >= pd.Timestamp("2025-09-01"))
    train_ids = products.index[~in_development].to_numpy()
    development = products.index.difference(train_ids).to_numpy()
    if weekly or final_fit:
        development_rows = int(products.covered_stores.iloc[development].sum())
    check(origins.max() <= pd.Timestamp(scope["last_entry"] if final_fit else "2025-11-30"),
          "Launch date outside the evaluation scope")
    files = sorted(work.glob("features_train_*.parquet"))
    panels = Panels(files + ([] if pilot else [work / "features_development.parquet"]), len(products), week_one)
    check(np.array_equal(products.panel_size.to_numpy()[train_ids], panels.widths[train_ids])
          and np.isfinite(panels.units).sum() == training_rows + (0 if pilot else development_rows),
          "Feature panels changed the frozen store membership or targets")
    if final_fit:
        completed = (origins + pd.Timedelta(days=scope["last_offset"])) < pd.Timestamp(final_fit)
        train_ids = products.index[completed].to_numpy()
        if weekly:
            fit, next_fit = map(pd.Timestamp, weekly)
            development = products.index[in_development & origins.between(fit, next_fit, inclusive="left")].to_numpy()
    elif weekly:
        fit, next_fit = map(pd.Timestamp, weekly)
        completed = (origins + pd.Timedelta(days=13)) < fit
        train_ids = products.index[completed].to_numpy()
        development = products.index[in_development & origins.between(fit, next_fit, inclusive="left")].to_numpy()
    if strict:
        allowed = products.colorway.isin(strict["allowed"]).to_numpy()
        train_ids = train_ids[allowed[train_ids]]
        issues = origins + pd.to_timedelta(scope.get("issue_offset", 0), unit="D")
        development = development[(issues.iloc[development] == pd.Timestamp(strict["issue"])).to_numpy()]
        check(int(products.covered_stores.iloc[train_ids].sum()) == strict["training_rows"]
              and not allowed[development].any() and len(development) > 0,
              "Date-specific strict training support disagrees with the released fit")
    if pilot:
        rng = np.random.default_rng(SEED)
        train_ids = np.sort(rng.choice(train_ids[products.covered_stores.to_numpy()[train_ids] > 0],
                                       size=PILOT_GROUPS, replace=False))
    covered = train_ids[products.covered_stores.to_numpy()[train_ids] > 0]
    selected = list(panels.selected_rows(covered))
    store_fit = np.unique(np.concatenate([panels.store_index[rows] for rows in selected]))
    mix_fit = np.concatenate([panels.mix[rows] for rows in selected])
    mix_fit = mix_fit[np.isfinite(mix_fit)]
    mix_mean = float(mix_fit.mean())
    mix_std = max(float(mix_fit.std()), 1e-6)
    if week_one:
        early_fit = np.concatenate([panels.week_one_sales[rows] for rows in selected])
        early_mean = float(np.nanmean(early_fit))
        early_scale = max(float(np.nanstd(early_fit)), 1e-6)
    product_columns = PRODUCT_NUMERIC if structured_sizes else tuple(
        name for name in PRODUCT_NUMERIC if name != "size_count")
    product_numeric, product_scaling = numeric(
        products, product_columns, covered, ("product_price", "prior_colorway_units", "lifecycle_age"))
    store_numeric, store_scaling = numeric(stores, STORE_NUMERIC, store_fit, (
        "reported_sales_usable_days", "prior_units_per_usable_day", "median_product_day_price",
        "priced_product_days"))
    store_codes, store_levels = categories(stores, ("state", "city_in_state"), store_fit)
    selected_ids = train_ids if pilot else products.index.to_numpy()
    size_values, size_levels = sizes(products.size_labels.to_numpy(), covered) if structured_sizes else (
        np.empty((len(products), 0), dtype=np.float32), [])
    if combined_product_text:
        texts = colorway_text(products.iloc[selected_ids], text_preparation)
        text, encoder = vectors(texts, device, text_preparation, replay_encoder_initialization)
        product_fields = [product_numeric[selected_ids], size_values[selected_ids]]
        if bypass_product_numeric:
            product_fields.reverse()
        product_features = np.column_stack([text, *product_fields])
        product_codes = np.empty((len(products), 0), dtype=np.int64)
        product_levels = {}
    else:
        product_codes, product_levels = categories(products, PRODUCT_CATEGORICAL, covered)
        names = products.product_text.iloc[selected_ids]
        text, encoder = vectors(names, device, replay_encoder_initialization=replay_encoder_initialization)
        product_features = np.column_stack([text, product_numeric[selected_ids], size_values[selected_ids],
                                            names.isna().to_numpy()])
    lookup = np.zeros(len(products), dtype=np.int64)
    lookup[selected_ids] = np.arange(len(selected_ids))
    product_features = torch.as_tensor(product_features, dtype=torch.float32, device=device)
    product_codes = torch.as_tensor(product_codes[selected_ids], device=device)
    store_numeric = torch.as_tensor(store_numeric, device=device)
    store_codes = torch.as_tensor(store_codes, device=device)
    complete_units = products.covered_units.iloc[covered].sum()
    complete_rows = products.covered_stores.iloc[covered].sum()
    unit_scale = max(1.0, float(complete_units / complete_rows)) / np.log(2)
    configuration = {
        "product_size": product_features.shape[1], "product_levels": [len(v) + 1 for v in product_levels.values()],
        "store_size": store_numeric.shape[1] + 2 + 2 * week_one,
        "store_levels": [len(v) + 1 for v in store_levels.values()],
        "unit_scale": unit_scale,
    }
    if bypass_product_numeric:
        configuration["product_numeric_bypass"] = len(product_columns)
        configuration["product_size"] -= len(product_columns)
    metadata = {
        "forecast_head": "direct_store_units", "configuration": configuration, "product_scaling": product_scaling, "store_scaling": store_scaling,
        "product_levels": product_levels, "store_levels": store_levels, "size_levels": size_levels,
        "mix_impute_and_center": mix_mean, "mix_scale": mix_std, "encoder": encoder,
        "seed": seed, "batch_groups": BATCH_SIZE, "learning_rate": LEARNING_RATE, "optimizer": "AdamW",
        "weight_decay": 0.01, "epoch_count": 2 if pilot else FULL_EPOCHS,
    }
    if final_fit:
        maturity = origins.iloc[covered] + pd.Timedelta(days=scope["last_offset"])
        check(maturity.max() < pd.Timestamp(final_fit), "Learned preprocessing used immature outcomes")
        metadata["training_cutoff"] = {
            "fit_date": str(final_fit), "earliest_label": str(maturity.min().date()),
            "latest_label": str(maturity.max().date()), "covered_training_rows": int(complete_rows),
        }
    if week_one:
        metadata["first_week_signal"] = {"log1p": True, "impute_and_center": early_mean,
                                         "scale": early_scale, "append_completeness": True}

    def batch(keys):
        rows, exists = panels.indexes(keys)
        store_idx = torch.as_tensor(panels.store_index[rows], device=device)
        mix = panels.mix[rows]
        mix_missing = ~np.isfinite(mix)
        mix_values = np.stack([np.where(mix_missing, 0, (mix - mix_mean) / mix_std), mix_missing], axis=-1)
        store_inputs = [store_numeric[store_idx], torch.as_tensor(mix_values, device=device)]
        if week_one:
            early = panels.week_one_sales[rows]
            complete = np.isfinite(early)
            early_inputs = np.stack([np.where(complete, (early - early_mean) / early_scale, 0),
                                     complete], axis=-1)
            store_inputs.append(torch.as_tensor(early_inputs, device=device))
        store_features = torch.cat(store_inputs, dim=-1)
        present = torch.as_tensor(exists, device=device)
        units = torch.as_tensor(panels.units[rows], device=device)
        product_idx = torch.as_tensor(lookup[keys], device=device)
        inputs = (product_features[product_idx], product_codes[product_idx],
                  store_features, store_codes[store_idx])
        return inputs, units, present, present & units.isfinite()

    return batch, metadata, panels, train_ids, development, products


def validate_model(model, batch, keys):
    args, units, present, covered = batch(keys)
    outputs = model.outputs(*args)
    forecasts = model.predictions(outputs)["two_tower_units"]
    loss = model.loss(outputs, units, covered)
    loss.backward()
    gradients = {}
    for name in ("product", "store", "forecast"):
        grad = [param.grad for param in getattr(model, name).parameters()]
        check(all(x is not None and x.isfinite().all() for x in grad), f"{name} has invalid gradients")
        gradients[name] = sum(float(x.norm()) for x in grad)
        check(gradients[name] > 0, f"{name} receives no gradient")
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        check(forecasts.isfinite().all() and (forecasts >= 0).all(), "Invalid store predictions")
        actual_missing = present & ~covered
        check(actual_missing.any(), "Pilot did not exercise incomplete outcome masking")
        changed = torch.where(covered, units, 1_000_000)
        check(torch.equal(loss.detach(), model.loss(outputs, changed, covered)),
              "Masked outcome values changed loss")
        check((covered & (units == 0)).any(), "Pilot missed complete zero targets")
        check(model.loss(outputs, changed, torch.zeros_like(covered)).item() == 0,
              "An empty batch contributes loss")
        product, product_codes, store, store_codes = args
        changed_store = store.clone()
        changed_codes = store_codes.clone()
        changed_store[:, 1:] = 0
        changed_codes[:, 1:] = 0
        counterfactual = model(product, product_codes, changed_store, changed_codes)
        check(torch.equal(counterfactual[:, 0], forecasts[:, 0]),
              "Changing other stores altered a direct store forecast")
        if model.head != DIRECT:
            positive = covered & (units > 0)
            check(positive.any(), "Pilot missed positive targets")
            changed_outputs = outputs.clone()
            changed_outputs[..., 1].masked_fill_(~positive, 100)
            check(torch.equal(loss.detach(), model.loss(changed_outputs, units, covered)),
                  "Conditional unit branch used zero or incomplete outcomes")
    return {"gradient_norms": gradients, "masked_label_invariance": True, "empty_batch_zero_loss": True,
            "real_masked_stores_in_check": int(actual_missing.sum()), "nonnegative_finite": True,
            "store_forecast_independent_of_other_panel_stores": True}


def score(work, features, model, batch, panels, development, device, strict=None, weekly=None):
    started = perf_counter()
    scope = json.loads((features / "panel_validation.json").read_text())
    extended = scope.get("last_offset", 13) != 13
    issue_offset = scope.get("issue_offset", 0)
    control_path = strict["control"] if strict else scope.get("control_path", OUTPUT / "run/matched_outcomes.parquet")
    extended = extended or bool(strict)
    entry_field = ", p.origin AS observed_entry_date" if extended else ""
    entry_date = "observed_entry_date" if extended else "forecast_date"
    join_keys = [*KEYS, "observed_entry_date"] if extended else KEYS
    predictions = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(development), BATCH_SIZE):
            keys = development[start:start+BATCH_SIZE]
            args, _, present, _ = batch(keys)
            components = model.predictions(model.outputs(*args))
            forecast = components["two_tower_units"]
            check(forecast.isfinite().all() and (forecast >= 0).all(), "Invalid development forecast")
            rows, exists = panels.indexes(keys)
            predictions.append(pd.DataFrame({
                "product_index": np.broadcast_to(keys[:, None], exists.shape)[exists],
                "store_index": panels.store_index[rows[exists]],
                **{name: value[present].cpu().numpy() for name, value in components.items()},
                "model_actual_units": panels.units[rows[exists]],
            }))
    predictions = pd.concat(predictions, ignore_index=True)
    with duckdb.connect() as db:
        db.register("issued", predictions)
        for name, path in (("products", features / "products.parquet"), ("stores", features / "stores.parquet"),
                           ("control", control_path)):
            db.from_parquet(str(path)).create_view(name)
        keys = ", ".join(join_keys)
        if weekly:
            fit, next_fit = weekly
            db.execute(f"""CREATE TEMP TABLE weekly_control AS SELECT * FROM control
                           WHERE forecast_date >= DATE '{fit}' AND forecast_date < DATE '{next_fit}'""")
        control = "weekly_control" if weekly else "control"
        db.execute(f"""
            CREATE TEMP TABLE prediction_keys AS SELECT
                p.model_product_id_type,p.model_product_id,p.color_identity_type,p.color_identity_value,
                s.storeid,p.origin + {issue_offset} AS forecast_date {entry_field},
                i.* EXCLUDE (product_index,store_index)
            FROM issued i JOIN products p USING(product_index) JOIN stores s USING(store_index)
        """)
        condition = " AND ".join(f"p.{key} IS NOT DISTINCT FROM c.{key}" for key in join_keys)
        db.execute(f"""
            CREATE TEMP TABLE matched AS SELECT c.*,p.* EXCLUDE ({keys}, model_actual_units)
            FROM prediction_keys p JOIN {control} c ON {condition}
            WHERE p.model_actual_units IS NOT DISTINCT FROM c.actual_units
        """)
        check(db.sql(f"""SELECT count(*)=count(DISTINCT({keys})) AND count(*)=(SELECT count(*) FROM {control})
                         AND count(*)=(SELECT count(*) FROM prediction_keys) FROM matched""").fetchall()[0][0],
              "Development key, target or coverage parity failed")
        db.sql(f"SELECT * FROM matched ORDER BY forecast_date,{keys}").write_parquet(
            str(work / "predictions.parquet"), compression="zstd")
        db.sql(f"""
            WITH errors AS (
                SELECT *, strftime({entry_date},'%Y-%m') AS month,
                    CASE WHEN product_seen_before_entry THEN 'new_color_known_product'
                         ELSE 'new_product_identity' END AS cohort,
                    CASE WHEN actual_units IS NOT NULL THEN two_tower_units::DOUBLE-actual_units END AS neural_error,
                    CASE WHEN actual_units IS NOT NULL THEN frozen_predicted_units-actual_units END AS frozen_error
                FROM matched
            ), events AS (
                SELECT forecast_date, model_product_id_type, model_product_id, color_identity_type,
                    color_identity_value,month,cohort,count(*) AS issued,count(actual_units) AS covered,
                    sum(actual_units) AS units,sum(neural_error) AS neural_signed,sum(frozen_error) AS frozen_signed,
                    sum(abs(neural_error)) AS neural_store,sum(abs(frozen_error)) AS frozen_store
                FROM errors GROUP BY ALL
            )
            SELECT coalesce(month,cohort,'overall') AS slice, count(*) AS launches, sum(issued) AS eligible_stores,
                   sum(covered) AS covered_stores,sum(units) AS actual_units,
                   sum(neural_store)/nullif(sum(units),0) AS two_tower_store_wape,
                   sum(frozen_store)/nullif(sum(units),0) AS lightgbm_store_wape,
                   sum(abs(neural_signed))/nullif(sum(units),0) AS two_tower_covered_network_wape,
                   sum(abs(frozen_signed))/nullif(sum(units),0) AS lightgbm_covered_network_wape,
                   sum(neural_signed)/nullif(sum(units),0) AS two_tower_signed_bias,
                   sum(frozen_signed)/nullif(sum(units),0) AS lightgbm_signed_bias,
                   sum(covered)::DOUBLE/sum(issued) AS coverage
            FROM events GROUP BY GROUPING SETS ((),(month),(cohort))
            ORDER BY CASE WHEN slice='overall' THEN 0 ELSE 1 END,slice
        """).write_csv(str(work / "metrics.csv"), header=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return perf_counter() - started


def run(work, features, pilot, score_only, head=DIRECT, text_preparation=TEXT_PREPARATION,
        seed=SEED, reference_run=None, strict=None, weekly=None, bypass_product_numeric=False,
        final_fit=None):
    check(not bypass_product_numeric or text_preparation == STRUCTURED_SIZES_PREPARATION,
          "Numeric bypass requires structured sizes")
    work.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.set_num_threads(6)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.reset_peak_memory_stats()
    replay_encoder_initialization = False
    if not pilot:
        pilot_dir = reference_run or work
        pilot_preprocessing = pilot_dir / "pilot_preprocessing.json"
        check((pilot_dir / "pilot_validation.json").is_file() and pilot_preprocessing.is_file(),
              "Complete the Stage 5 pilot first")
        pilot_settings = json.loads(pilot_preprocessing.read_text())
        check(pilot_settings["forecast_head"] == head and
              pilot_settings["encoder"]["text_preparation"] == text_preparation,
              "Complete the Stage 5 pilot with the same head and product inputs first")
        if reference_run is not None:
            original_encoder = json.loads((reference_run / "preprocessing.json").read_text())["encoder"]
            # Encoder construction happened before tower initialization in the original fit.
            replay_encoder_initialization = original_encoder["newly_encoded_texts"] > 0
        check((work / "model.pt").exists() == score_only,
              "Use --score-only to resume scoring an existing frozen checkpoint")
    batch, metadata, panels, train, development, products = prepare(
        features, pilot, device, text_preparation, seed, replay_encoder_initialization, strict, weekly,
        bypass_product_numeric, final_fit)
    metadata["forecast_head"] = head
    if head != DIRECT:
        metadata["configuration"]["head"] = head
        metadata["loss"] = "mean_binary_cross_entropy + mean_positive_shifted_poisson"
    fit_started = perf_counter()
    rng = np.random.default_rng(seed)
    model = TwoTower(**metadata["configuration"]).to(device)
    if score_only:
        validation_path = work / "validation.json"
        receipt = json.loads(validation_path.read_text())
        original = json.loads((work / "preprocessing.json").read_text())
        for key in original.keys() - {"encoder"}:
            check(json.loads(json.dumps(metadata[key])) == original[key],
                  f"Frozen preprocessing changed for {key}")
        for key in ("revision", "text_preparation", "dimension", "open_clip", "transformers"):
            check(metadata["encoder"][key] == original["encoder"][key], f"Frozen encoder changed for {key}")
        model.load_state_dict(torch.load(work / "model.pt", map_location=device, weights_only=True))
        receipt["score_seconds"] = score(work, features, model, batch, panels, development, device, strict, weekly)
        receipt["development_exact_key_target_coverage_match"] = True
        write_json(validation_path, receipt)
        return
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    num_epochs = metadata["epoch_count"]
    probe = rng.choice(train, size=BATCH_SIZE, replace=False)
    validation = validate_model(model, batch, probe)
    print(f"Gradient and panel checks passed on {device}; training {len(train):,} panels", flush=True)
    epoch_reports = []
    for epoch in range(num_epochs):
        epoch_started = perf_counter()
        shuffled = rng.permutation(train)
        losses = []
        for start in range(0, len(shuffled), BATCH_SIZE):
            ids = shuffled[start:start+BATCH_SIZE]
            args, units, _, covered = batch(ids)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss(model.outputs(*args), units, covered)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            losses.append(loss.detach())
        mean_loss = float(torch.stack(losses).mean())
        check(np.isfinite(mean_loss), "Training diverged")
        report = {"epoch": epoch + 1, "mean_batch_loss": mean_loss, "seconds": perf_counter() - epoch_started}
        epoch_reports.append(report)
        print(report, flush=True)
    validation["after_training"] = validate_model(model, batch, probe)
    fit_seconds = perf_counter() - fit_started
    prefix = "pilot_" if pilot else ""
    torch.save(model.state_dict(), work / f"{prefix}model.pt")
    write_json(work / f"{prefix}preprocessing.json", metadata)
    validation_path = work / f"{prefix}validation.json"
    receipt = {
        **validation, "fit_date": str(final_fit or (weekly[0] if weekly else "2025-09-01")),
        "recorded_at_utc": datetime.now(timezone.utc),
        "feature_directory": str(features), "forecast_head": head,
        "seed": seed, "reference_run": str(reference_run) if reference_run else None,
        "device": str(device), "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "torch": torch.__version__, "training_groups": len(train),
        "training_covered_rows": int(products.covered_stores.iloc[train].sum()),
        "training_panel_rows": int(products.panel_size.iloc[train].sum()),
        "development_exact_key_target_coverage_match": None,
        "epochs": epoch_reports, "fit_seconds": fit_seconds,
        "matrix_preparation_seconds": fit_started - started - metadata["encoder"]["encoding_seconds"],
        "encoder": metadata["encoder"], "score_seconds": None, "elapsed_seconds": perf_counter() - started,
        "process_peak_mib": peak_memory_mib(), "cuda_peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None),
    }
    if strict:
        receipt["strict_issue"] = strict["issue"]
        receipt["exclusion_sha256"] = strict["exclusion_sha256"]
    write_json(validation_path, receipt)
    if not pilot:
        receipt["score_seconds"] = score(work, features, model, batch, panels, development, device, strict, weekly)
        receipt["development_exact_key_target_coverage_match"] = True
        receipt["elapsed_seconds"] = perf_counter() - started
        receipt["process_peak_mib"] = peak_memory_mib()
        write_json(validation_path, receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    parser.add_argument("--features", type=Path, help="Reuse prepared training and development features.")
    parser.add_argument("--head", choices=(DIRECT, TWO_PART, INTERACTION), default=DIRECT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bypass-product-numeric", action="store_true")
    parser.add_argument("--reference-run", type=Path,
                        help="Reuse a saved pilot and replay its encoder initialization for a seed rerun.")
    text = parser.add_mutually_exclusive_group()
    text.add_argument("--combined-product-text", dest="text_preparation", action="store_const",
                      const=ORDERED_PREPARATION, default=TEXT_PREPARATION)
    text.add_argument("--structured-sizes", dest="text_preparation", action="store_const",
                      const=STRUCTURED_SIZES_PREPARATION)
    text.add_argument("--historical-product-text", dest="text_preparation",
                      choices=(COMBINED_PREPARATION, SUBCATEGORY_PREPARATION),
                      help="Reproduce a saved experiment using its original prompt order.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pilot", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    run(args.work.resolve(), (args.features or args.work).resolve(), args.pilot, args.score_only,
        args.head, args.text_preparation, args.seed, args.reference_run,
        bypass_product_numeric=args.bypass_product_numeric)
