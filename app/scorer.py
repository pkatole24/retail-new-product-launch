"""Private inference worker for the Vercel launch planner."""

import hmac
import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import open_clip

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fit_two_tower_1_14 import TwoTower
from two_tower_text_1_14 import REVISION, STRUCTURED_SIZES_PREPARATION, colorway_text

WINDOWS = {"days1to14": ("prelaunch_1_14", 1), "days15to28": ("prelaunch_15_28", 15)}
STYLE_FIELDS = ("id", "product", "businessLine", "category", "subcategory", "sizes")


def scale(rows, settings):
    values = np.array([[row.get(name) for name in settings["columns"]] for row in rows], dtype=np.float32)
    for name in settings["log1p"]:
        index = settings["columns"].index(name)
        values[:, index] = np.log1p(np.maximum(values[:, index], 0))
    missing = ~np.isfinite(values)
    scaled = (values - np.asarray(settings["impute_and_center"])) / np.asarray(settings["scale"])
    return np.column_stack((np.where(missing, 0, scaled), missing)).astype(np.float32)


class Forecast:
    def __init__(self, release, model_dir, encoder_dir):
        self.release = release
        self.stores = release["stores"]
        self.styles = {style["id"]: {**{key: style[key] for key in STYLE_FIELDS}, "group": style.get("group")}
                       for style in release["styles"]}
        if len({store["id"] for store in self.stores}) != len(self.stores):
            raise ValueError("The release contains duplicate store identifiers")
        encoder_id = f"local-dir:{encoder_dir.as_posix()}"
        self.tokenizer = open_clip.get_tokenizer(encoder_id)
        self.encoder = open_clip.create_model(encoder_id, device="cpu", require_pretrained=True).eval()
        self.models = {}
        for field, (window, _) in WINDOWS.items():
            self.models[field] = []
            for seed in (13, 29, 47):
                checkpoint = model_dir / window / f"seed_{seed}"
                settings = json.loads((checkpoint / "preprocessing.json").read_text(encoding="utf-8"))
                encoder = settings["encoder"]
                if encoder["revision"] != REVISION or encoder["text_preparation"] != STRUCTURED_SIZES_PREPARATION:
                    raise ValueError("Checkpoint encoder provenance differs from the scoring release")
                model = TwoTower(**settings["configuration"]).eval()
                model.load_state_dict(torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=True))
                self.models[field].append((model, settings))

    @torch.inference_mode()
    def predict(self, brief):
        if brief["kind"] == "colorway":
            style = self.styles[brief["style"]]
            brief = {**brief, **{key: style[key] for key in STYLE_FIELDS if key not in ("id", "sizes")},
                     "sizes": style["sizes"] or brief["sizes"]}
        sizes = set(brief["sizes"] or ())
        product = {
            "product_text": brief["product"], "color_identity_value": brief["color"],
            "business_line": brief["businessLine"], "category": brief["category"],
            "subcategory": brief["subcategory"], "product_price": brief["price"],
            "is_launch": 1, "origin_month": brief["month"],
            "prior_colorway_units": None, "prior_colorway_days": 0, "size_count": len(sizes) if sizes else None,
        }
        description = colorway_text(pd.DataFrame([product]), STRUCTURED_SIZES_PREPARATION).iloc[0]
        tokens = self.tokenizer([" ".join(description.split())])
        vector = self.encoder.encode_text(tokens, normalize=True).float()
        mix = []
        for store in self.stores:
            row = store["features"]
            category_share = store["categoryShare"].get(brief["businessLine"], {}).get(brief["category"], 0)
            mix.append(category_share if (row.get("reported_sales_usable_days") or 0) > 0 else None)
        mix = np.asarray(mix, dtype=np.float32)
        predictions = {}
        for field, checkpoints in self.models.items():
            # The later head was fit with the lifecycle age at its target window start.
            product["lifecycle_age"] = WINDOWS[field][1]
            seed_predictions = []
            for model, settings in checkpoints:
                levels = {label: index for index, label in enumerate(settings["size_levels"], 1)}
                size_features = np.zeros((1, len(levels) + 1), dtype=np.float32)
                for size in sizes:
                    size_features[0, levels.get(size, 0)] = 1
                product_features = torch.cat((vector, torch.from_numpy(scale([product], settings["product_scaling"])),
                                              torch.from_numpy(size_features)), dim=1)
                store_numeric = scale([store["features"] for store in self.stores], settings["store_scaling"])
                missing = ~np.isfinite(mix)
                mix_values = np.where(missing, 0, (mix - settings["mix_impute_and_center"]) / settings["mix_scale"])
                store_numeric = np.column_stack((store_numeric, mix_values, missing)).astype(np.float32)
                category_codes = []
                # The saved JSON object order is not the model's embedding order.
                for name in ("state", "city_in_state"):
                    codes = {label: i for i, label in enumerate(settings["store_levels"][name], 1)}
                    category_codes.append([codes.get(store["features"].get(name), 0) for store in self.stores])
                output = model(product_features, torch.empty((1, 0), dtype=torch.long),
                               torch.from_numpy(store_numeric)[None], torch.as_tensor(category_codes).T[None])
                seed_predictions.append(output[0].numpy())
            predictions[field] = np.mean(seed_predictions, axis=0)
        return {"snapshotDate": self.release["snapshotDate"], "stores": [
            {**{key: store.get(key) for key in ("id", "name", "metro")},
             **{key: store["features"].get(key) for key in ("latitude", "longitude")},
             **{field: float(values[index]) for field, values in predictions.items()}}
            for index, store in enumerate(self.stores)
        ]}


def serve(forecast, token):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            data = json.dumps(payload, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def handle_request(self):
            if self.command == "GET" and self.path == "/healthz":
                return self.respond(200, {"status": "ready"})
            databricks = self.path.startswith("/api/")
            path = self.path.removeprefix("/api") if databricks else self.path
            authorization = self.headers.get("X-Forecast-Token", "").encode() if databricks else self.headers.get("Authorization", "").encode()
            expected = token if databricks else b"Bearer " + token
            if not hmac.compare_digest(authorization, expected):
                return self.respond(401, {"error": "Unauthorized"})
            if self.command == "GET" and path == "/catalog":
                return self.respond(200, {"styles": list(forecast.styles.values())})
            if self.command == "POST" and path == "/predict":
                try:
                    size = int(self.headers.get("Content-Length", "0"))
                    if not 0 < size <= 8192:
                        raise ValueError("Invalid request length")
                    result = forecast.predict(json.loads(self.rfile.read(size)))
                except (KeyError, ValueError, TypeError):
                    return self.respond(400, {"error": "Invalid product brief"})
                return self.respond(200, result)
            return self.respond(404, {"error": "Resource unavailable"})

        do_GET = handle_request
        do_POST = handle_request

    host = os.getenv("FORECAST_BIND", "127.0.0.1")
    port = int(os.getenv("FORECAST_PORT") or os.getenv("DATABRICKS_APP_PORT") or "8001")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    release = json.loads(Path(os.environ["FORECAST_RELEASE_JSON"]).read_text(encoding="utf-8"))
    token = os.environ["FORECAST_SERVICE_TOKEN"].encode()
    if not token or release["snapshotDate"] != "2026-05-31" or not release["stores"]:
        raise ValueError("An authorized dataset-end release and service token are required")
    forecast = Forecast(release, Path(os.environ["FORECAST_MODEL_DIR"]), Path(os.environ["FORECAST_ENCODER_DIR"]))
    serve(forecast, token)
