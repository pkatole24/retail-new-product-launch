# Retail Launch Planner

**Build a Launch** forecasts first-month unit sales for a new product or colorway. Enter the product brief, run the forecast, compare pilot stores, and allocate a network-wide inventory budget. For a catalog style without released sizes, enter the sizes manually. **View sample workflow** uses fictional results and works without the live service.

The planner compares the highest-forecast pilot shortlist with a same-size shortlist that maximizes known metro coverage while retaining at least 90% of the top shortlist's forecast. Network allocation uses store forecast shares and preserves the entered whole-unit budget. The maps use bundled [U.S. Census boundaries](https://github.com/topojson/us-atlas).

## Vercel and Databricks Model Serving

Set the Vercel project root to `app`. The interface lives in `public/`. The server-side gateway in `api/forecast.mjs` requests the style catalog and forecasts from a Databricks Model Serving endpoint using OAuth. Configure these variables in Vercel's server environment:

- `DATABRICKS_HOST` and `DATABRICKS_SERVING_ENDPOINT` identify the workspace and endpoint.
- `DATABRICKS_CLIENT_ID` and `DATABRICKS_CLIENT_SECRET` authenticate a dedicated service principal with permission to query the endpoint.

The scorer uses six frozen PyTorch checkpoints and the pinned FashionSigLIP encoder. It averages seeds 13, 29, and 47 for each prelaunch window. Live forecasts use the serving release's store snapshot. A missing or unavailable endpoint prevents live forecasts, and the sample workflow remains available.

## Local live inference

From the full capstone checkout, run `node app/local.mjs` and open `http://127.0.0.1:8080`. The launcher starts the Python scorer on loopback and requires the private release JSON, checkpoints, encoder, and Python environment. Press Ctrl+C to stop it.

- Override artifact locations with `FORECAST_RELEASE_JSON`, `FORECAST_MODEL_DIR`, and `FORECAST_ENCODER_DIR`. The model directory must contain `prelaunch_1_14` and `prelaunch_15_28`, each with the three seed directories containing `model.pt` and `preprocessing.json`.
- Override the interpreter with `FORECAST_PYTHON` or change loopback ports with `FORECAST_UI_PORT` and `FORECAST_WORKER_PORT`.

## Dataset-end release contract

Run `python app/build_release.py` from the full capstone checkout using its model environment and authenticated Databricks profile. Release generation requires the team snapshot SQL and saved final-model outputs. It writes a private bundle under `data/launch_app_release_20260531/`. The roster includes stores with positive reported sales in May, with a left join to `store_metro` by store ID. Ambiguous or missing map entries stay unknown. Store features use the existing strict prior-day recipe with a June 1 forecast origin, covering sales through May 31.

The style catalog uses the saved final main evaluation. Rank distinct base products by observed units across comparable stores in the first 28 days, keeping up to three per business line, category, subcategory and launch group. Each product is represented by its highest-observed-sales eligible colorway within that group. When a style has no released size information, the planner supplies sizes.

- `snapshotDate` is `2026-05-31`. `stores` has one row per eligible store, with string `id` and `name`, a metro label or null, and `features` holding the raw columns named in each checkpoint's `store_scaling.columns` plus `state` and `city_in_state`. `categoryShare` is keyed by business line and category. Missing category shares for stores with positive usable sales use the training preparation's zero-share rule.
- `styles` contains approved base products, each with `id`, `product`, `businessLine`, `category`, `subcategory`, and `sizes` as a string array. The worker inherits the catalog values for a new colorway.
- Store coordinates from the feature snapshot place the forecast, pilot, and allocation markers.
