# Retail Launch Planner

The Vercel project root is `app`. The browser files are in `public/`, and `api/forecast.mjs` forwards live requests to the private inference worker. The app offers a labeled fictional workflow for UI review without access to any private artifacts. The **Run sales forecast** action calls the live worker.

To run live forecasts locally from the repository root, run `node app/local.mjs` and open `http://127.0.0.1:8080`. The launcher uses the existing Windows model environment and private release, starts the Python worker, and serves the browser and API on loopback. Press Ctrl+C to stop both. Override `FORECAST_UI_PORT` or `FORECAST_WORKER_PORT` if a port is busy. Override `FORECAST_PYTHON`, `FORECAST_RELEASE_JSON`, `FORECAST_MODEL_DIR`, or `FORECAST_ENCODER_DIR` if those files are stored elsewhere.

## Live inference

The worker in `scorer.py` runs the six frozen PyTorch checkpoints and the pinned FashionSigLIP text encoder. Run it in the existing model environment with the repository's `src/` directory available. Keep the encoder, checkpoints, and approved release file on the worker host. Configure HTTPS access via a reverse proxy or private hosting integration. The worker listens on loopback by default.

Worker environment variables

- `FORECAST_MODEL_DIR` points to the directory with `prelaunch_1_14/seed_{13,29,47}` and `prelaunch_15_28/seed_{13,29,47}`, each with `model.pt` and `preprocessing.json`.
- `FORECAST_ENCODER_DIR` points to the local encoder at the revision recorded in the checkpoints.
- `FORECAST_RELEASE_JSON` points to the authorized dataset-end release described below.
- `FORECAST_SERVICE_TOKEN` is a shared worker token. `FORECAST_PORT` defaults to `8001`.

Set `FORECAST_SERVICE_URL` and the same `FORECAST_SERVICE_TOKEN` in Vercel's server environment. Deploy from the `app` root after the intended store outputs and style catalog have been cleared for the app's audience.

For a container host, build from the repository root with `docker build -f app/Dockerfile.worker -t retail-forecast-worker .`. Mount the authorized release, model, and encoder artifacts read-only at `/release`, `/models`, and `/encoder`. Supply `FORECAST_SERVICE_TOKEN` from the host's secret manager. The container listens on port `8001` and `/healthz` becomes available after the models load. Use the host's HTTPS ingress and scope the Vercel secrets to the protected preview environment.

For the Databricks Apps host, keep the model and data artifacts in a restricted Unity Catalog volume, grant the worker service principal read access to that volume, and configure their paths through the worker's `FORECAST_*` settings. Set `FORECAST_BIND=0.0.0.0` in the app runtime. Configure a dedicated Vercel caller identity with `CAN USE` on the app. In addition to the worker URL and token, the Vercel preview requires `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, and `DATABRICKS_CLIENT_SECRET`. The gateway exchanges those credentials for an OAuth token and sends the application token separately. Keep both credentials in server-side secret storage.

## Dataset-end release contract

Run `python app/build_release.py` from the full capstone checkout using its model environment and authenticated Databricks profile. Release generation requires the team snapshot SQL and saved final-model outputs. It writes a private bundle under `data/launch_app_release_20260531/`. The roster includes stores with positive reported sales in May, with a left join to `store_metro` by store ID. Ambiguous or missing map entries stay unknown. Store features use the existing strict prior-day recipe with a June 1 forecast origin, covering sales through May 31.

The style catalog uses the saved final main evaluation. Rank distinct base products by observed units across comparable stores in the first 28 days, keeping up to three per business line, category, subcategory and launch group. Each product is represented by its highest-observed-sales eligible colorway within that group. When a style has no released size information, the planner supplies sizes.

- `snapshotDate` is `2026-05-31`. `stores` has one row per eligible store, with string `id` and `name`, a metro label or null, and `features` holding the raw columns named in each checkpoint's `store_scaling.columns` plus `state` and `city_in_state`. `categoryShare` is keyed by business line and category. Missing category shares for stores with positive usable sales use the training preparation's zero-share rule.
- `styles` contains approved base products, each with `id`, `product`, `businessLine`, `category`, `subcategory`, and `sizes` as a string array. The worker inherits the catalog values for a new colorway.
- Store coordinates from the feature snapshot place the forecast, pilot, and allocation markers. Maps use the U.S. Census boundaries packaged by [us-atlas](https://github.com/topojson/us-atlas) and run without a third-party map service.

The planner compares the highest-forecast pilot shortlist with a metro-coverage shortlist that keeps at least 90% of its expected sales. Network allocation uses forecast shares and preserves the entered whole-unit budget. Live scoring requires a verified dataset-end release and a reachable inference worker.
