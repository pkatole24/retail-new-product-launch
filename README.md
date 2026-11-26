# Retail New Product Launch Forecasting

How many units might a new colorway sell in each store during its first four weeks? This project builds prelaunch forecasts from product attributes and store history, then evaluates whether first-week sales improve the forecast for the remaining launch window.

My work covers the Databricks modeling tables, the Python forecasting workflow, and the evaluation of launch forecasts. The broader team capstone supplied upstream retail data and shared analysis.

## Databricks data pipeline

Databricks SQL prepares daily sales history at the colorway, store, and day grain. The modeling pipeline uses this history to define launch cohorts, build store features, and assemble outcomes for the forecast windows. Versioned Delta inputs record which data releases an evaluation used. Coverage checks distinguish stores with observed zero units from stores with missing outcomes.

## Models and evaluation

The evaluation compares a category-based baseline, lifecycle LightGBM, and a two-part neural model. The neural model estimates the probability of positive observed sales and the expected units when sales are positive. Their product is the store-level unit forecast.

The prelaunch workflow forecasts days 1–14 and 15–28. A separate workflow uses first-week observations to forecast days 8–28. Historical evaluation respects the forecast issue date and checks how results change when linked product history is excluded. Store-level error, network-level error, and volume bias describe different aspects of forecast performance.

## Forecast tool

The planned tool will use a versioned model release with its matching feature preparation and a dated store-history snapshot. The original retail data and Databricks workspace are private.

## Source map

- [Databricks lifecycle table](sql/model_product_color_store_day_lifecycle_v1.sql) builds the daily colorway and store history.
- [Final feature preparation](src/prepare_stage5_final.py) assembles inputs for the forecast windows.
- [Model fitting](src/run_stage5_final_models.py) runs the LightGBM and neural comparisons.
- [Evaluation](src/score_stage5_final.py) computes store and network metrics from matched predictions.
