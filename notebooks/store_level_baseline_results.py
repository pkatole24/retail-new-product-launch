# Databricks notebook source
# MAGIC %md
# MAGIC # Lifecycle-based store baseline and matched development refresh
# MAGIC Sales and taxonomy come from the lifecycle table; missing days remain
# MAGIC governed by store coverage. Yesterday's LightGBM forecasts are read
# MAGIC from the existing results snapshot. All staging views are session-only.
# COMMAND ----------
CATALOG = "`msba-capstone-team13`"
SOURCE = f"{CATALOG}.modelling"
RESULTS = f"{CATALOG}.modelling_results"
DATASET = "product_color_name_merged_stage2_20261116_v1"
FEATURES = "stage3_features_20261117_v1"
BASE, PRE, WEEK = "lifecycle_base_tmp", "lifecycle_pre_tmp", "lifecycle_week_tmp"
STORE_RESULTS, MATCHED, METRICS = "lifecycle_results_tmp", "lifecycle_matched_tmp", "lifecycle_metrics_tmp"
ORIGINAL_RESULTS = f"{RESULTS}.store_level_forecast_results_20261119_v1"
def cache_view(name):
    spark.sql(f"CREATE TEMP TABLE {name}_materialized AS SELECT * FROM {name}")
    spark.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS SELECT * FROM {name}_materialized")
    count = spark.sql(f"SELECT count(*) n FROM {name}").first().n
    print(f"Materialized session table {name}: {count:,} rows", flush=True)
    return count
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW lifecycle_saved_models AS
SELECT forecast_stage,horizon,evaluation_mode,prediction_cutoff,model_product_id_type,
 model_product_id,color_identity_type,color_identity_value,observed_entry_date,storeid,
 lightgbm_store_forecast store_forecast,lightgbm_product_known product_known,
 lightgbm_store_known store_known,product_seen_before_entry,
 lightgbm_training_rows training_rows,lightgbm_latest_training_label latest_training_label
FROM {ORIGINAL_RESULTS}
WHERE lightgbm_available AND ((forecast_stage='prelaunch' AND horizon IN ('1_14','15_28'))
 OR (forecast_stage='week1' AND horizon='8_28'))
""")
assert cache_view("lifecycle_saved_models")==7716540
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW lifecycle_saved_metadata AS
SELECT dataset_version,feature_version,forecast_stage,model_product_id_type,model_product_id,
 color_identity_type,color_identity_value,observed_entry_date,storeid,
 state,latitude,longitude,product_names,sizes,observed_prices,reporting_group
FROM {ORIGINAL_RESULTS} WHERE evaluation_mode='new_color'
AND ((forecast_stage='prelaunch' AND horizon='1_14')
 OR (forecast_stage='week1' AND horizon='8_28'))
""")
assert cache_view("lifecycle_saved_metadata")==2572180

# COMMAND ----------
spark.sql("-- Seven-day signal at entry-store grain. Incomplete observed windows stay null;\n-- the partial amount and coverage are available separately for downstream masks.\nCREATE OR REPLACE TEMP VIEW lifecycle_week1_signal AS\nWITH entries AS (\n SELECT * FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1\n WHERE scope_status='preliminary_in_scope' AND date_add(observed_entry_date,6)<=DATE '2026-05-31'\n), entry_stores AS (\n SELECT DISTINCT observed_entry_date,storeid\n FROM `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1\n WHERE observed_entry_date IN (SELECT DISTINCT observed_entry_date FROM entries)\n), coverage AS (\n SELECT e.observed_entry_date,e.storeid,count(s.dt) recorded_days,\n        count_if(s.network_health_status='healthy') healthy_days\n FROM entry_stores e LEFT JOIN `msba-capstone-team13`.modelling.model_store_day s\n   ON s.dataset_version='stage1_20261112_v1' AND s.storeid=e.storeid\n  AND s.dt BETWEEN e.observed_entry_date AND date_add(e.observed_entry_date,6)\n GROUP BY ALL\n), observations AS (\n SELECT c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,\n        c.observed_entry_date,p.storeid,count_if(s.network_health_status='healthy') observed_healthy_days,\n        count_if(s.network_health_status='healthy' AND (NOT p.sales_label_complete OR p.reported_sales IS NULL)) missing_measure_days,\n        sum(CASE WHEN s.network_health_status='healthy' THEN p.reported_sales END) partial_healthy_sales\n FROM entries c JOIN `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1 p\n   ON c.model_product_id_type=p.model_product_id_type AND c.model_product_id=p.model_product_id\n  AND c.color_identity_type=p.color_identity_type AND c.color_identity_value <=> p.color_identity_value\n  AND p.currency='USD' AND p.dt BETWEEN c.observed_entry_date AND date_add(c.observed_entry_date,6)\n LEFT JOIN `msba-capstone-team13`.modelling.model_store_day s\n   ON s.dataset_version='stage1_20261112_v1' AND s.storeid=p.storeid AND s.dt=p.dt\n GROUP BY ALL\n), assembled AS (\n SELECT c.dataset_version,c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,\n        c.observed_entry_date,m.storeid,v.recorded_days,v.healthy_days,7-v.recorded_days missing_store_days,\n        coalesce(o.observed_healthy_days,0) observed_healthy_days,coalesce(o.missing_measure_days,0) missing_measure_days,\n        coalesce(o.partial_healthy_sales,CAST(0 AS BIGINT)) partial_healthy_sales\n FROM entries c JOIN `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1 m\n   ON c.model_product_id_type=m.model_product_id_type AND c.model_product_id=m.model_product_id\n  AND c.color_identity_type=m.color_identity_type AND c.color_identity_value <=> m.color_identity_value\n  AND c.observed_entry_date=m.observed_entry_date\n JOIN coverage v ON v.observed_entry_date=c.observed_entry_date AND v.storeid=m.storeid\n LEFT JOIN observations o ON o.model_product_id_type=c.model_product_id_type AND o.model_product_id=c.model_product_id\n  AND o.color_identity_type=c.color_identity_type AND o.color_identity_value <=> c.color_identity_value\n  AND o.observed_entry_date=c.observed_entry_date AND o.storeid=m.storeid\n)\nSELECT *,healthy_days-observed_healthy_days assumed_zero_days,\n       healthy_days=7 AND missing_measure_days=0 AS signal_complete,\n       CASE WHEN healthy_days=7 AND missing_measure_days=0 THEN partial_healthy_sales END AS week1_reported_sales,\n       date_add(observed_entry_date,7) prediction_cutoff\nFROM assembled;\n")
HISTORY_QUERY = "-- Session-scoped development baseline features. Store-day health and frozen\n-- entry-store membership remain separate from the sparse lifecycle source.\nWITH membership AS (\n SELECT * FROM `msba-capstone-team13`.modelling.model_evaluation_membership_fixed_v1\n WHERE forecast_stage='prelaunch' AND horizon='1_28'\n   AND scope_status='preliminary_in_scope' AND calendar_mature\n   AND evaluation_period<>'locked_test' AND observed_entry_date<=DATE '2025-11-30'\n), dates AS (\n SELECT DISTINCT observed_entry_date FROM membership\n), entries AS (\n SELECT DISTINCT l.model_product_id_type,l.model_product_id,l.color_identity_type,l.color_identity_value,\n  l.observed_entry_date,l.business_line,l.category,\n  CASE WHEN l.category<>'unknown' THEN l.subcategory END subcategory\n FROM `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1 l\n JOIN dates d ON l.observed_entry_date=d.observed_entry_date\n WHERE l.dt=l.observed_entry_date\n), daily AS (\n SELECT l.dt,l.storeid,l.business_line,l.category,l.subcategory,sum(l.reported_sales) sales\n FROM `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1 l\n JOIN `msba-capstone-team13`.modelling.model_store_day s\n  ON s.dataset_version='stage1_20261112_v1' AND s.dt=l.dt AND s.storeid=l.storeid\n  AND s.network_health_status='healthy' AND s.missing_sales_observations=0\n WHERE l.currency='USD' AND l.sales_label_complete\n GROUP BY ALL\n), history AS (\n SELECT h.observed_entry_date,h.storeid,d.business_line,d.category,d.subcategory,sum(d.sales) sales\n FROM `msba-capstone-team13`.modelling.model_store_snapshot_v1 h\n JOIN dates e ON e.observed_entry_date=h.observed_entry_date\n JOIN daily d ON d.storeid=h.storeid\n  AND d.dt>=date_sub(h.observed_entry_date,90) AND d.dt<h.observed_entry_date\n GROUP BY ALL\n), store_category AS (\n SELECT observed_entry_date,storeid,business_line,category,sum(sales) sales\n FROM history GROUP BY ALL\n), portfolio_total AS (\n SELECT h.observed_entry_date,sum(h.reported_sales_usable_days) sales,\n  percentile(try_divide(h.reported_sales_usable_days,h.usable_sales_days),0.5) median_daily_rate\n FROM `msba-capstone-team13`.modelling.model_store_snapshot_v1 h\n JOIN dates d ON h.observed_entry_date=d.observed_entry_date GROUP BY h.observed_entry_date\n), portfolio_subcategory AS (\n SELECT observed_entry_date,business_line,category,subcategory,sum(sales) sales\n FROM history GROUP BY ALL\n), portfolio_category AS (\n SELECT observed_entry_date,business_line,category,sum(sales) sales\n FROM history GROUP BY ALL\n), store_rows AS (\n SELECT e.model_product_id_type,e.model_product_id,e.color_identity_type,e.color_identity_value,\n  e.observed_entry_date,k.storeid,e.business_line,e.category,m.evaluation_period,\n  m.prediction_cutoff,m.evaluation_product_groups,e.subcategory,\n  CASE WHEN e.subcategory IS NOT NULL THEN 'entry_subcategory' ELSE 'category_fallback' END taxonomy_source,\n  coalesce(try_divide(sc.sales,h.reported_sales_usable_days),\n           CASE WHEN pt.sales>0 THEN try_divide(coalesce(pc.sales,0),pt.sales) END) category_mix_feature,\n  CASE WHEN sc.sales IS NOT NULL AND h.reported_sales_usable_days>0 THEN 'store_history'\n       WHEN pt.sales>0 THEN 'portfolio_fallback'\n       ELSE 'unavailable' END category_feature_source,\n  h.reported_sales_usable_days,h.usable_sales_days,pt.median_daily_rate,\n  CASE WHEN h.usable_sales_days>0 THEN try_divide(h.reported_sales_usable_days,h.usable_sales_days)\n       ELSE pt.median_daily_rate END daily_rate,\n  CASE WHEN h.usable_sales_days>0 THEN 'store_history'\n       WHEN pt.median_daily_rate IS NOT NULL THEN 'portfolio_fallback'\n       ELSE 'unavailable' END rate_source,\n  sh.sales store_subcategory_sales,ps.sales portfolio_subcategory_sales,pt.sales portfolio_sales\n FROM membership m\n JOIN entries e ON e.model_product_id_type=m.model_product_id_type AND e.model_product_id=m.model_product_id\n  AND e.color_identity_type=m.color_identity_type AND e.color_identity_value <=> m.color_identity_value\n  AND e.observed_entry_date=m.observed_entry_date\n JOIN `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1 k\n  ON k.model_product_id_type=m.model_product_id_type AND k.model_product_id=m.model_product_id\n  AND k.color_identity_type=m.color_identity_type AND k.color_identity_value <=> m.color_identity_value\n  AND k.observed_entry_date=m.observed_entry_date\n JOIN `msba-capstone-team13`.modelling.model_store_snapshot_v1 h\n  ON h.observed_entry_date=m.observed_entry_date AND h.storeid=k.storeid\n JOIN portfolio_total pt ON pt.observed_entry_date=m.observed_entry_date\n LEFT JOIN history sh ON sh.observed_entry_date=m.observed_entry_date AND sh.storeid=k.storeid\n  AND sh.business_line=e.business_line AND sh.category=e.category AND sh.subcategory=e.subcategory\n LEFT JOIN portfolio_subcategory ps ON ps.observed_entry_date=m.observed_entry_date\n  AND ps.business_line=e.business_line AND ps.category=e.category AND ps.subcategory=e.subcategory\n LEFT JOIN store_category sc ON sc.observed_entry_date=m.observed_entry_date AND sc.storeid=k.storeid\n  AND sc.business_line=e.business_line AND sc.category=e.category\n LEFT JOIN portfolio_category pc ON pc.observed_entry_date=m.observed_entry_date\n  AND pc.business_line=e.business_line AND pc.category=e.category\n), proxies AS (\n SELECT *,\n CASE WHEN taxonomy_source='category_fallback' THEN category_mix_feature\n      WHEN store_subcategory_sales IS NOT NULL AND reported_sales_usable_days>0\n        THEN try_divide(store_subcategory_sales,reported_sales_usable_days)\n      ELSE try_divide(portfolio_subcategory_sales,portfolio_sales) END mix,\n CASE WHEN taxonomy_source='category_fallback' THEN category_feature_source\n      WHEN store_subcategory_sales IS NOT NULL AND reported_sales_usable_days>0 THEN 'store_history'\n      WHEN portfolio_subcategory_sales IS NOT NULL AND portfolio_sales>0 THEN 'portfolio_fallback'\n      ELSE 'unavailable' END mix_source\n FROM store_rows\n)\nSELECT *,daily_rate*mix daily_proxy FROM proxies\n"
spark.sql(f"CREATE OR REPLACE TEMP VIEW {BASE} AS {HISTORY_QUERY}")
base_count = cache_view(BASE)
old_base_count = spark.table(f"{RESULTS}.store_subcategory_feature_history_20261119_v1").count()
assert base_count == old_base_count, (base_count, old_base_count)
bad_base = spark.sql(f"""SELECT count(*) n FROM (
 SELECT model_product_id_type,model_product_id,color_identity_type,color_identity_value,observed_entry_date,storeid,
 count(*) n FROM {BASE} GROUP BY ALL HAVING n<>1)""").first().n
assert bad_base == 0
print("Lifecycle history key and row-count checks passed", flush=True)
membership = f"""
 SELECT * FROM {SOURCE}.model_lightgbm_sales_membership_v1
 UNION ALL SELECT * FROM {SOURCE}.model_evaluation_membership_fixed_v1
 WHERE forecast_stage='prelaunch' AND horizon IN ('1_28','8_28')
"""

# COMMAND ----------
def prelaunch_sql(horizon, days):
    return f"""
    WITH m AS (
     SELECT * FROM ({membership}) WHERE forecast_stage='prelaunch' AND horizon='{horizon}'
      AND scope_status='preliminary_in_scope' AND calendar_mature AND evaluation_period<>'locked_test'
      AND observed_entry_date<=DATE '2025-11-30'
    ), labeled AS (
     SELECT b.*,m.window_end,m.product_seen_before_entry,t.target_complete,t.reported_sales_target,
      t.recorded_store_days,t.healthy_store_days,t.missing_store_days,t.missing_measure_days,t.assumed_zero_days
     FROM {BASE} b JOIN m ON b.model_product_id_type=m.model_product_id_type AND b.model_product_id=m.model_product_id
      AND b.color_identity_type=m.color_identity_type AND b.color_identity_value <=> m.color_identity_value
      AND b.observed_entry_date=m.observed_entry_date
     JOIN {SOURCE}.model_launch_target_name_merged_v1 t
      ON t.dataset_version='{DATASET}' AND t.model_product_id_type=m.model_product_id_type
      AND t.model_product_id=m.model_product_id AND t.color_identity_type=m.color_identity_type
      AND t.color_identity_value <=> m.color_identity_value AND t.observed_entry_date=m.observed_entry_date
      AND t.horizon=m.horizon AND t.storeid=b.storeid
    ), eligible AS (
     SELECT window_end,evaluation_product_groups,business_line,category,subcategory,taxonomy_source,
      sum(reported_sales_target) sales,sum(daily_proxy*{days}D) proxy,count(*) n,
      count_if(reported_sales_target=0) zero_n
     FROM labeled WHERE target_complete AND daily_proxy>0 GROUP BY ALL
    ), heldout AS (
     SELECT prediction_cutoff,array_distinct(flatten(collect_list(evaluation_product_groups))) groups
     FROM (SELECT DISTINCT model_product_id_type,model_product_id,color_identity_type,color_identity_value,
           observed_entry_date,prediction_cutoff,evaluation_product_groups
           FROM labeled WHERE evaluation_period='development') GROUP BY prediction_cutoff
    ), modes AS (
     SELECT * FROM VALUES ('new_color'),('known_linked_product_holdout') AS x(evaluation_mode)
    ), fit AS (
     SELECT p.prediction_cutoff,p.business_line,p.category,p.subcategory,p.taxonomy_source,m.evaluation_mode,
      coalesce(sum(t.n),0) eligible_rows,coalesce(sum(t.zero_n),0) zero_targets,
      coalesce(sum(t.n) FILTER (WHERE t.business_line=p.business_line AND t.category=p.category
       AND t.taxonomy_source=p.taxonomy_source AND t.subcategory <=> p.subcategory),0) matching_rows,
      coalesce(sum(t.zero_n) FILTER (WHERE t.business_line=p.business_line AND t.category=p.category
       AND t.taxonomy_source=p.taxonomy_source AND t.subcategory <=> p.subcategory),0) matching_zero_targets,
      max(t.window_end) latest_training_label,
      coalesce(try_divide(
        sum(t.sales) FILTER (WHERE t.business_line=p.business_line AND t.category=p.category
         AND t.taxonomy_source=p.taxonomy_source AND t.subcategory <=> p.subcategory),
        sum(t.proxy) FILTER (WHERE t.business_line=p.business_line AND t.category=p.category
         AND t.taxonomy_source=p.taxonomy_source AND t.subcategory <=> p.subcategory)),
       try_divide(sum(t.sales),sum(t.proxy))) factor
     FROM (SELECT DISTINCT prediction_cutoff,business_line,category,subcategory,taxonomy_source
           FROM labeled WHERE evaluation_period='development') p
     CROSS JOIN modes m JOIN heldout h ON h.prediction_cutoff=p.prediction_cutoff
     LEFT JOIN eligible t ON t.window_end<p.prediction_cutoff
      AND (m.evaluation_mode='new_color' OR NOT arrays_overlap(t.evaluation_product_groups,h.groups))
     GROUP BY ALL
    ), predictions AS (
     SELECT r.*,m.evaluation_mode,f.factor,f.eligible_rows,f.zero_targets,f.matching_rows,
      f.matching_zero_targets,f.latest_training_label,
      CASE WHEN f.matching_rows>0 THEN 'matched_group' ELSE 'pooled_horizon' END calibration_source,
      r.daily_proxy*{days}D*f.factor store_forecast
     FROM labeled r CROSS JOIN modes m LEFT JOIN fit f
      ON f.evaluation_mode=m.evaluation_mode AND f.prediction_cutoff=r.prediction_cutoff
      AND f.business_line <=> r.business_line AND f.category <=> r.category
      AND f.taxonomy_source=r.taxonomy_source AND f.subcategory <=> r.subcategory
     WHERE r.evaluation_period='development'
    )
    SELECT '{DATASET}' dataset_version,'{FEATURES}' feature_version,
     'store_subcategory_lifecycle_aggregate_ratio_v1' baseline_version,'prelaunch' forecast_stage,'{horizon}' horizon,
     evaluation_mode,model_product_id_type,model_product_id,color_identity_type,color_identity_value,
     observed_entry_date,storeid,prediction_cutoff,latest_training_label,product_seen_before_entry,
     business_line,category,subcategory,taxonomy_source,mix_source,rate_source,daily_rate,mix,daily_proxy,
     factor,calibration_source,eligible_rows,matching_rows,zero_targets,matching_zero_targets,
     target_complete,reported_sales_target,recorded_store_days,healthy_store_days,missing_store_days,
     missing_measure_days,assumed_zero_days,store_forecast,
     CASE WHEN count_if(store_forecast IS NULL) OVER panel=0 THEN sum(store_forecast) OVER panel END total_forecast,
     CASE WHEN count_if(store_forecast IS NULL) OVER panel=0 THEN try_divide(store_forecast,sum(store_forecast) OVER panel) END store_share
    FROM predictions WINDOW panel AS (PARTITION BY evaluation_mode,model_product_id_type,model_product_id,
       color_identity_type,color_identity_value,observed_entry_date)
    """

pre_parts = []
for horizon,days in [("1_14",14),("15_28",14),("8_28",21)]:
    name = f"lifecycle_pre_{horizon}_tmp"
    spark.sql(f"CREATE OR REPLACE TEMP VIEW {name} AS {prelaunch_sql(horizon,days)}")
    cache_view(name)
    pre_parts.append(name)
spark.sql(f"CREATE OR REPLACE TEMP VIEW {PRE} AS " + " UNION ALL ".join(f"SELECT * FROM {p}" for p in pre_parts))
cache_view(PRE)
pre_check = spark.sql(f"""SELECT horizon,evaluation_mode,count(*) rows,
 count(DISTINCT to_json(named_struct('t',model_product_id_type,'p',model_product_id,'ct',color_identity_type,
 'cv',color_identity_value,'d',observed_entry_date,'s',storeid))) keys,
 count_if(store_forecast<0 OR latest_training_label>=prediction_cutoff) invalid
 FROM {PRE} GROUP BY ALL""")
assert pre_check.count()==6 and pre_check.filter("rows<>keys OR invalid>0").count()==0

# COMMAND ----------
def week_sql(horizon):
    return f"""
    WITH m AS (
     SELECT * FROM {SOURCE}.model_lightgbm_sales_membership_v1
     WHERE forecast_stage='week1' AND horizon='{horizon}' AND scope_status='preliminary_in_scope'
      AND calendar_mature AND evaluation_period<>'locked_test' AND observed_entry_date<=DATE '2025-11-30'
    ), historical_events AS (
     SELECT m.window_end,m.evaluation_product_groups,sum(t.reported_sales_target) future_sales,
      sum(w.week1_reported_sales) early_sales,count(*) training_rows
     FROM m JOIN {SOURCE}.model_launch_target_name_merged_v1 t
      ON t.dataset_version=m.dataset_version AND t.model_product_id_type=m.model_product_id_type
      AND t.model_product_id=m.model_product_id AND t.color_identity_type=m.color_identity_type
      AND t.color_identity_value <=> m.color_identity_value AND t.observed_entry_date=m.observed_entry_date
      AND t.horizon=m.horizon AND t.target_complete
     JOIN lifecycle_week1_signal w
      ON w.dataset_version=m.dataset_version AND w.model_product_id_type=t.model_product_id_type
      AND w.model_product_id=t.model_product_id AND w.color_identity_type=t.color_identity_type
      AND w.color_identity_value <=> t.color_identity_value AND w.observed_entry_date=t.observed_entry_date
      AND w.storeid=t.storeid AND w.signal_complete
     GROUP BY ALL
    ), heldout AS (
     SELECT prediction_cutoff,array_distinct(flatten(collect_list(evaluation_product_groups))) groups
     FROM m WHERE evaluation_period='development' GROUP BY prediction_cutoff
    ), modes AS (
     SELECT * FROM VALUES ('new_color'),('known_linked_product_holdout') AS x(evaluation_mode)
    ), fit AS (
     SELECT h.prediction_cutoff,mode.evaluation_mode,
      try_divide(sum(t.future_sales),sum(t.early_sales)) early_factor,
      coalesce(sum(t.training_rows),0) early_training_rows,max(t.window_end) latest_early_training_label
     FROM heldout h CROSS JOIN modes mode LEFT JOIN historical_events t
      ON t.window_end<h.prediction_cutoff
      AND (mode.evaluation_mode='new_color' OR NOT arrays_overlap(t.evaluation_product_groups,h.groups))
     GROUP BY ALL
    )
    SELECT p.dataset_version,p.feature_version,'store_subcategory_lifecycle_week1_blend_v1' baseline_version,
     'week1' forecast_stage,p.horizon,p.evaluation_mode,p.model_product_id_type,p.model_product_id,
     p.color_identity_type,p.color_identity_value,p.observed_entry_date,p.storeid,
     date_add(p.observed_entry_date,7) prediction_cutoff,p.product_seen_before_entry,
     p.business_line,p.category,p.subcategory,p.taxonomy_source,p.mix_source,p.rate_source,
     p.daily_rate,p.mix,p.daily_proxy,p.factor,p.calibration_source,p.eligible_rows,p.matching_rows,
     p.zero_targets,p.matching_zero_targets,p.latest_training_label prelaunch_latest_training_label,
     fit.early_factor,fit.early_training_rows,fit.latest_early_training_label,
     w.signal_complete week1_complete,w.week1_reported_sales,p.target_complete,p.reported_sales_target,
     p.recorded_store_days,p.healthy_store_days,p.missing_store_days,p.missing_measure_days,p.assumed_zero_days,
     p.store_forecast prelaunch_component,
     CASE WHEN w.signal_complete AND fit.early_factor IS NOT NULL
          THEN w.week1_reported_sales*fit.early_factor END early_component,
     CASE WHEN w.signal_complete AND fit.early_factor IS NOT NULL THEN 'blended'
          ELSE 'prelaunch_fallback' END update_source
    FROM {PRE} p LEFT JOIN lifecycle_week1_signal w
     ON w.dataset_version=p.dataset_version AND w.model_product_id_type=p.model_product_id_type
     AND w.model_product_id=p.model_product_id AND w.color_identity_type=p.color_identity_type
     AND w.color_identity_value <=> p.color_identity_value AND w.observed_entry_date=p.observed_entry_date
     AND w.storeid=p.storeid
    LEFT JOIN fit ON fit.prediction_cutoff=date_add(p.observed_entry_date,7)
     AND fit.evaluation_mode=p.evaluation_mode
    WHERE p.horizon='{horizon}'
    """

spark.sql(f"CREATE OR REPLACE TEMP VIEW {WEEK} AS {week_sql('8_28')}")
cache_view(WEEK)
week_check = spark.sql(f"""SELECT evaluation_mode,count(*) rows,
 count(DISTINCT to_json(named_struct('t',model_product_id_type,'p',model_product_id,'ct',color_identity_type,
 'cv',color_identity_value,'d',observed_entry_date,'s',storeid))) keys,
 count_if(latest_early_training_label>=prediction_cutoff OR prelaunch_latest_training_label>=observed_entry_date
 OR (update_source='blended' AND (NOT week1_complete OR early_component IS NULL))) invalid
 FROM {WEEK} GROUP BY ALL""")
assert week_check.count()==2 and week_check.filter("rows<>keys OR invalid>0").count()==0
model_query = "SELECT * FROM lifecycle_saved_models"

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW {STORE_RESULTS} AS
WITH models AS ({model_query}), baseline AS (
 SELECT dataset_version,feature_version,baseline_version,forecast_stage,horizon,evaluation_mode,
  model_product_id_type,model_product_id,color_identity_type,color_identity_value,observed_entry_date,storeid,
  prediction_cutoff,product_seen_before_entry,business_line,category,subcategory,taxonomy_source,mix_source,
  rate_source,daily_rate,mix,daily_proxy,factor,calibration_source,eligible_rows,matching_rows,zero_targets,
  matching_zero_targets,latest_training_label prelaunch_latest_training_label,
  CAST(NULL AS DOUBLE) early_factor,CAST(NULL AS BIGINT) early_training_rows,
  CAST(NULL AS DATE) latest_early_training_label,CAST(NULL AS STRING) update_source,
  target_complete,reported_sales_target,recorded_store_days,healthy_store_days,missing_store_days,
  missing_measure_days,assumed_zero_days,store_forecast prelaunch_component,CAST(NULL AS DOUBLE) early_component
 FROM {PRE}
 UNION ALL
 SELECT dataset_version,feature_version,baseline_version,forecast_stage,horizon,evaluation_mode,
  model_product_id_type,model_product_id,color_identity_type,color_identity_value,observed_entry_date,storeid,
  prediction_cutoff,product_seen_before_entry,business_line,category,subcategory,taxonomy_source,mix_source,
  rate_source,daily_rate,mix,daily_proxy,factor,calibration_source,eligible_rows,matching_rows,zero_targets,
  matching_zero_targets,prelaunch_latest_training_label,early_factor,early_training_rows,latest_early_training_label,
  update_source,target_complete,reported_sales_target,recorded_store_days,healthy_store_days,missing_store_days,
  missing_measure_days,assumed_zero_days,prelaunch_component,early_component
 FROM {WEEK}
)
SELECT b.*,w.signal_complete week1_complete,w.week1_reported_sales,
 f.state,f.latitude,f.longitude,f.product_names,f.sizes,f.observed_prices,f.reporting_group,
 CASE WHEN b.forecast_stage='week1' AND b.early_component IS NOT NULL
      THEN .75D*b.prelaunch_component+.25D*b.early_component ELSE b.prelaunch_component END baseline_weight_025,
 CASE WHEN b.forecast_stage='week1' AND b.early_component IS NOT NULL
      THEN .5D*b.prelaunch_component+.5D*b.early_component ELSE b.prelaunch_component END baseline_weight_050,
 CASE WHEN b.forecast_stage='week1' AND b.early_component IS NOT NULL
      THEN .25D*b.prelaunch_component+.75D*b.early_component ELSE b.prelaunch_component END baseline_weight_075,
 CASE WHEN b.forecast_stage='week1' AND b.early_component IS NOT NULL
      THEN b.early_component ELSE b.prelaunch_component END baseline_weight_100,
 m.store_forecast lightgbm_store_forecast,m.product_known lightgbm_product_known,m.store_known lightgbm_store_known,
 m.training_rows lightgbm_training_rows,m.latest_training_label lightgbm_latest_training_label,
 m.store_forecast IS NOT NULL lightgbm_available
FROM baseline b LEFT JOIN models m ON b.forecast_stage=m.forecast_stage AND b.horizon=m.horizon
 AND b.evaluation_mode=m.evaluation_mode AND b.prediction_cutoff=m.prediction_cutoff
 AND b.model_product_id_type=m.model_product_id_type AND b.model_product_id=m.model_product_id
 AND b.color_identity_type=m.color_identity_type AND b.color_identity_value <=> m.color_identity_value
 AND b.observed_entry_date=m.observed_entry_date AND b.storeid=m.storeid
LEFT JOIN lifecycle_week1_signal w
 ON w.dataset_version=b.dataset_version AND w.model_product_id_type=b.model_product_id_type
 AND w.model_product_id=b.model_product_id AND w.color_identity_type=b.color_identity_type
 AND w.color_identity_value <=> b.color_identity_value AND w.observed_entry_date=b.observed_entry_date AND w.storeid=b.storeid
LEFT JOIN lifecycle_saved_metadata f
 ON f.dataset_version=b.dataset_version AND f.feature_version=b.feature_version AND f.forecast_stage=b.forecast_stage
 AND f.model_product_id_type=b.model_product_id_type AND f.model_product_id=b.model_product_id
 AND f.color_identity_type=b.color_identity_type AND f.color_identity_value <=> b.color_identity_value
 AND f.observed_entry_date=b.observed_entry_date AND f.storeid=b.storeid
""")
assert cache_view(STORE_RESULTS)==10288720
validation = spark.sql(f"""
 SELECT count(*) rows,count_if(r.lightgbm_store_forecast IS NULL) missing,
 count_if(NOT (r.lightgbm_store_forecast <=> m.store_forecast)) changed
 FROM lifecycle_saved_models m LEFT JOIN {STORE_RESULTS} r
 ON r.forecast_stage=m.forecast_stage AND r.horizon=m.horizon AND r.evaluation_mode=m.evaluation_mode
 AND r.prediction_cutoff=m.prediction_cutoff AND r.model_product_id_type=m.model_product_id_type
 AND r.model_product_id=m.model_product_id AND r.color_identity_type=m.color_identity_type
 AND r.color_identity_value <=> m.color_identity_value AND r.observed_entry_date=m.observed_entry_date
 AND r.storeid=m.storeid
""").first()
assert validation.rows==7716540 and validation.missing==0 and validation.changed==0, validation
print("All saved LightGBM forecasts preserved", flush=True)

# COMMAND ----------
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW {MATCHED} AS
WITH window_coverage AS (
 SELECT forecast_stage,horizon,evaluation_mode,count(*) expected_rows,count_if(lightgbm_available) available_rows
 FROM {STORE_RESULTS}
 WHERE (forecast_stage='prelaunch' AND horizon IN ('1_14','15_28'))
    OR (forecast_stage='week1' AND horizon='8_28') GROUP BY ALL
), snapshot_status AS (
 SELECT CASE WHEN count(*)=6 AND count_if(expected_rows=available_rows)=6
             THEN 'complete_development_forecasts' ELSE 'partial_lightgbm_forecasts' END comparison_status
 FROM window_coverage
), pivoted AS (
 SELECT evaluation_mode,model_product_id_type,model_product_id,color_identity_type,color_identity_value,
 observed_entry_date,storeid,max(product_seen_before_entry) product_seen_before_entry,
 max(business_line) business_line,max(category) category,max(subcategory) subcategory,max(state) state,
 max(latitude) latitude,max(longitude) longitude,first(product_names,true) product_names,
 first(sizes,true) sizes,first(observed_prices,true) observed_prices,max(reporting_group) reporting_group,
 max(taxonomy_source) taxonomy_source,max(mix_source) mix_source,max(rate_source) rate_source,
 max(daily_rate) daily_rate,max(mix) mix,max(week1_complete) week1_complete,max(week1_reported_sales) week1_sales,
 max(prelaunch_component) FILTER (WHERE forecast_stage='prelaunch' AND horizon='1_14') c1,
 max(prelaunch_component) FILTER (WHERE forecast_stage='prelaunch' AND horizon='15_28') c2,
 max(prelaunch_component) FILTER (WHERE forecast_stage='prelaunch' AND horizon='29_56') c3,
 max(prelaunch_component) FILTER (WHERE forecast_stage='week1' AND horizon='8_28') cb8,
 max(early_component) FILTER (WHERE forecast_stage='week1' AND horizon='8_28') ce8,
 max(prelaunch_component) FILTER (WHERE forecast_stage='week1' AND horizon='29_56') cb29,
 max(early_component) FILTER (WHERE forecast_stage='week1' AND horizon='29_56') ce29,
 max(lightgbm_store_forecast) FILTER (WHERE forecast_stage='prelaunch' AND horizon='1_14') l1,
 max(lightgbm_store_forecast) FILTER (WHERE forecast_stage='prelaunch' AND horizon='15_28') l2,
 max(lightgbm_store_forecast) FILTER (WHERE forecast_stage='prelaunch' AND horizon='29_56') l3,
 max(lightgbm_store_forecast) FILTER (WHERE forecast_stage='week1' AND horizon='8_28') l8,
 max(lightgbm_store_forecast) FILTER (WHERE forecast_stage='week1' AND horizon='29_56') l29
 FROM {STORE_RESULTS} GROUP BY ALL
), labeled AS (
 SELECT p.*,t.horizon,t.reported_sales_target actual,
 CASE WHEN t.horizon='1_28' THEN c1+c2 ELSE c3 END category_prelaunch,
 CASE WHEN t.horizon='1_28' THEN cb8 ELSE cb29 END category_update_base,
 CASE WHEN t.horizon='1_28' THEN ce8 ELSE ce29 END category_update_early,
 CASE WHEN t.horizon='1_28' THEN l1+l2 ELSE l3 END lightgbm_prelaunch,
 CASE WHEN t.horizon='1_28' THEN week1_sales+l8 ELSE l29 END lightgbm_week1
 FROM pivoted p JOIN {SOURCE}.model_launch_target_name_merged_v1 t
 ON t.dataset_version='{DATASET}' AND t.horizon='1_28' AND t.target_complete
 AND p.model_product_id_type=t.model_product_id_type AND p.model_product_id=t.model_product_id
 AND p.color_identity_type=t.color_identity_type AND p.color_identity_value <=> t.color_identity_value
 AND p.observed_entry_date=t.observed_entry_date AND p.storeid=t.storeid WHERE p.week1_complete
)
SELECT labeled.*,snapshot_status.comparison_status FROM labeled CROSS JOIN snapshot_status
 WHERE category_prelaunch IS NOT NULL AND category_update_base IS NOT NULL
 AND lightgbm_prelaunch IS NOT NULL AND lightgbm_week1 IS NOT NULL
""")
spark.sql(f"""
CREATE OR REPLACE TEMP VIEW {METRICS} AS
WITH weights AS (SELECT * FROM VALUES (0D),(.25D),(.5D),(.75D),(1D) AS w(early_weight)),
long_form AS (
 SELECT p.*,w.early_weight,model,stage,forecast FROM {MATCHED} p CROSS JOIN weights w
 LATERAL VIEW stack(4,'subcategory','prelaunch',category_prelaunch,
  'subcategory','week1',CASE WHEN horizon='1_28' THEN week1_sales ELSE 0 END +
    CASE WHEN category_update_early IS NULL THEN category_update_base
         ELSE (1D-w.early_weight)*category_update_base+w.early_weight*category_update_early END,
  'lightgbm','prelaunch',lightgbm_prelaunch,'lightgbm','week1',lightgbm_week1) x AS model,stage,forecast
), events AS (
 SELECT evaluation_mode,model_product_id_type,model_product_id,color_identity_type,color_identity_value,
 observed_entry_date,product_seen_before_entry,horizon,early_weight,model,stage,comparison_status,count(*) stores,
 sum(actual) actual,sum(forecast) forecast,sum(abs(forecast-actual)) store_error FROM long_form GROUP BY ALL
)
SELECT evaluation_mode,horizon,early_weight,model,stage,first(comparison_status,true) comparison_status,
 CASE WHEN grouping(product_seen_before_entry)=1 THEN 'all' WHEN product_seen_before_entry THEN 'known_identity'
      WHEN product_seen_before_entry=false THEN 'new_identity' ELSE 'unknown' END identity_status,
 count(*) events,sum(stores) covered_store_rows,sum(actual) actual,sum(forecast) forecast,
 try_divide(sum(store_error),sum(actual)) store_wape,
 try_divide(sum(abs(forecast-actual)),sum(actual)) covered_network_wape,
 try_divide(sum(forecast-actual),sum(actual)) signed_bias
FROM events GROUP BY GROUPING SETS
 ((evaluation_mode,horizon,early_weight,model,stage),
  (evaluation_mode,horizon,early_weight,model,stage,product_seen_before_entry))
""")
matched_count = cache_view(MATCHED)
metrics_count = cache_view(METRICS)
assert matched_count == 2230978 and metrics_count > 0, (matched_count, metrics_count)
display(spark.sql(f"SELECT * FROM {METRICS}"))
print("VALIDATED; publishing replacements to existing results tables", flush=True)
for final,temp in [
 ("store_subcategory_feature_history_20261119_v1",BASE),
 ("store_subcategory_prelaunch_forecasts_20261119_v1",PRE),
 ("store_subcategory_week1_forecasts_20261119_v1",WEEK),
 ("store_level_forecast_results_20261119_v1",STORE_RESULTS),
 ("store_level_matched_comparison_20261119_v1",MATCHED),
 ("store_level_matched_metrics_20261119_v1",METRICS),
]:
    spark.sql(f"CREATE OR REPLACE TABLE {RESULTS}.{final} USING DELTA AS SELECT * FROM {temp}")
    print(f"Replaced existing {final}", flush=True)
print("LIFECYCLE_REFRESH_COMPLETED", flush=True)
