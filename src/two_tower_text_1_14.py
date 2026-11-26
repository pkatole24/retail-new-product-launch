"""Cache normalized FashionSigLIP vectors without fine-tuning the encoder."""
from hashlib import sha256
from importlib.metadata import version
from time import perf_counter

import numpy as np
import pandas as pd
import torch

from fetch_two_tower_1_14 import SOURCES
from run_weekly_lightgbm_1_14 import check, write_json

CHECKPOINT = "Marqo/marqo-fashionSigLIP"
REVISION = "c56244cc94f92419e8369fa71efdaf403b124ce8"
TEXT_PREPARATION = "source_product_name_whitespace_v1"
COMBINED_PREPARATION = "colorway_name_color_sizes_business_line_category_v1"
SUBCATEGORY_PREPARATION = "colorway_name_color_subcategory_sizes_business_line_category_v1"
ORDERED_PREPARATION = "colorway_name_color_business_line_category_subcategory_sizes_v1"
STRUCTURED_SIZES_PREPARATION = "colorway_name_color_business_line_category_subcategory_v1"


def colorway_text(products, text_preparation=ORDERED_PREPARATION):
    def known(value):
        return str(value).strip() if pd.notna(value) and str(value).strip() else "unknown"

    labels = {"product_text": "Product", "color_identity_value": "Color",
              "business_line": "Business line", "category": "Category",
              "subcategory": "Subcategory", "size_labels": "Sizes"}
    orders = {
        ORDERED_PREPARATION: tuple(labels),
        STRUCTURED_SIZES_PREPARATION: tuple(field for field in labels if field != "size_labels"),
        COMBINED_PREPARATION: ("product_text", "color_identity_value", "size_labels", "business_line", "category"),
        SUBCATEGORY_PREPARATION: ("product_text", "color_identity_value", "subcategory", "size_labels",
                                  "business_line", "category"),
    }
    fields = orders[text_preparation]
    format_value = {
        **dict.fromkeys(labels, known),
        "size_labels": lambda sizes: ", ".join(map(str, sizes)) if sizes is not None and len(sizes) else "unknown",
    }
    texts = [
        " ".join(f"{labels[field]} {format_value[field](value)}." for field, value in zip(fields, row))
        for row in products[list(fields)].itertuples(index=False, name=None)
    ]
    return pd.Series(texts, index=products.index, dtype="str")


def vectors(names, device, text_preparation=TEXT_PREPARATION, replay_encoder_initialization=False):
    started = perf_counter()
    checkpoint = SOURCES.parent / "encoder" / REVISION
    cache = SOURCES.parent / "text_cache" / REVISION / text_preparation
    cache.mkdir(parents=True, exist_ok=True)
    names = names.fillna("").map(lambda name: " ".join(name.split()))
    unique = {sha256(name.encode("utf-8")).hexdigest(): name for name in names.unique() if name}
    required = names.map(lambda name: sha256(name.encode("utf-8")).hexdigest() if name else "")
    available = {}
    for file in sorted(cache.glob("*.parquet")):
        for row in pd.read_parquet(file).itertuples(index=False):
            available[row.text_hash] = (row.vector, row.tokens)
    missing = sorted(set(unique) - available.keys())
    context = 64
    dimension = 768
    if missing or replay_encoder_initialization:
        import open_clip

        model_id = f"local-dir:{checkpoint.as_posix()}"
        tokenizer = open_clip.get_tokenizer(model_id)
        model = open_clip.create_model(model_id, device=device, require_pretrained=True).eval()
        model.requires_grad_(False)
        context = tokenizer.context_length
        dimension = model.text.text_projection.shape[-1] if isinstance(
            model.text.text_projection, torch.Tensor) else model.text.text_projection.out_features
        with torch.inference_mode():
            for shard in range(0, len(missing), 512):
                rows = []
                for start in range(shard, min(shard + 512, len(missing)), 64):
                    keys = missing[start:min(start + 64, shard + 512)]
                    text = [unique[key] for key in keys]
                    lengths = [len(tokenizer.encode(name, add_special_tokens=True)) for name in text]
                    encoded = model.encode_text(tokenizer(text).to(device), normalize=True).float().cpu().numpy()
                    check(encoded.shape == (len(keys), dimension) and np.isfinite(encoded).all()
                          and np.allclose(np.linalg.norm(encoded, axis=1), 1, atol=1e-4),
                          "Invalid frozen text vectors")
                    for key, length, vector in zip(keys, lengths, encoded):
                        rows.append((key, vector, length))
                        available[key] = (vector, length)
                chunk_hash = sha256("".join(row[0] for row in rows).encode("ascii")).hexdigest()
                path = cache / f"{chunk_hash}.parquet"
                temporary = path.with_suffix(".partial")
                pd.DataFrame(rows, columns=["text_hash", "vector", "tokens"]).to_parquet(
                    temporary, index=False)
                temporary.replace(path)
                print(f"FashionSigLIP encoded {min(shard + 512, len(missing)):,}/{len(missing):,} new texts",
                      flush=True)
    zero = np.zeros(dimension, dtype=np.float32)
    output = np.stack([available[key][0] if key else zero for key in required])
    metadata = {
        "checkpoint": CHECKPOINT, "revision": REVISION, "text_preparation": text_preparation,
        "open_clip": version("open_clip_torch"), "transformers": version("transformers"),
        "context_tokens": context, "dimension": output.shape[1], "unique_texts": len(unique),
        "newly_encoded_texts": len(missing), "missing_text_groups": int((required == "").sum()),
        "encoder_initialization_replayed": replay_encoder_initialization and not missing,
        "truncated_unique_texts": sum(available[key][1] > context for key in unique),
        "truncated_product_examples": sum(bool(key) and available[key][1] > context for key in required),
        "maximum_tokens": max((available[key][1] for key in unique), default=0),
        "encoding_seconds": perf_counter() - started,
    }
    if missing:
        write_json(cache / "encoder.json", metadata)
    return output, metadata
