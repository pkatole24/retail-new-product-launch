CREATE TABLE `msba-capstone-team13`.modelling.model_launch_target_name_merged_v1 USING DELTA AS
-- Reconciled store/window outcomes without persisting the daily Cartesian grid.
-- All membership is fixed at entry. Partial outcomes remain distinct from labels.
WITH horizons AS (
 SELECT * FROM VALUES ('1_14',1,14),('15_28',15,28),('1_28',1,28),
   ('8_28',8,28),('29_56',29,56),('8_56',8,56) AS h(horizon,first_day,last_day)
), windows AS (
 SELECT c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,
        c.observed_entry_date,c.reporting_group,h.horizon,h.first_day,h.last_day,
        date_add(c.observed_entry_date,h.first_day-1) window_start,
        date_add(c.observed_entry_date,h.last_day-1) window_end
 FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1 VERSION AS OF 0 c CROSS JOIN horizons h
 WHERE c.scope_status='preliminary_in_scope'
   AND date_add(c.observed_entry_date,h.last_day-1)<=DATE '2026-05-31'
), date_windows AS (
 SELECT DISTINCT observed_entry_date,horizon,first_day,last_day,window_start,window_end FROM windows
), entry_stores AS (
 SELECT DISTINCT observed_entry_date,storeid
 FROM `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1
 WHERE observed_entry_date IN (SELECT DISTINCT observed_entry_date FROM windows)
), coverage AS (
 SELECT w.observed_entry_date,w.horizon,w.first_day,w.last_day,es.storeid,
        count(s.storeid) AS recorded_store_days,
        count_if(s.network_health_status='healthy') AS healthy_store_days,
        count_if(s.storeid IS NOT NULL AND (s.network_health_status IS NULL OR s.network_health_status<>'healthy')) AS unhealthy_store_days
 FROM date_windows w JOIN entry_stores es ON w.observed_entry_date=es.observed_entry_date
 LEFT JOIN `msba-capstone-team13`.modelling.model_store_day VERSION AS OF 1 s
   ON s.dataset_version='stage1_20261112_v1' AND s.storeid=es.storeid AND s.dt BETWEEN w.window_start AND w.window_end
 GROUP BY ALL
), observed AS (
 SELECT w.model_product_id_type,w.model_product_id,w.color_identity_type,w.color_identity_value,
        w.observed_entry_date,w.horizon,p.storeid,
        count_if(s.network_health_status='healthy') AS observed_healthy_product_days,
        count_if(s.network_health_status='healthy' AND (p.missing_sales_observations>0 OR p.reported_sales IS NULL)) AS missing_measure_days,
        sum(CASE WHEN s.network_health_status='healthy' THEN p.reported_sales END) AS observed_healthy_sales
 FROM windows w JOIN `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0 p
   ON w.model_product_id_type=p.model_product_id_type AND w.model_product_id=p.model_product_id
  AND w.color_identity_type=p.color_identity_type AND w.color_identity_value <=> p.color_identity_value
  AND p.currency='USD' AND p.dt BETWEEN w.window_start AND w.window_end
 LEFT JOIN `msba-capstone-team13`.modelling.model_store_day VERSION AS OF 1 s
   ON s.dataset_version='stage1_20261112_v1' AND s.dt=p.dt AND s.storeid=p.storeid
 GROUP BY ALL
), assembled AS (
 SELECT w.*,m.storeid,c.recorded_store_days,c.healthy_store_days,c.unhealthy_store_days,
        w.last_day-w.first_day+1-c.recorded_store_days AS missing_store_days,
        coalesce(o.observed_healthy_product_days,0) AS observed_healthy_product_days,
        coalesce(o.missing_measure_days,0) AS missing_measure_days,
        coalesce(o.observed_healthy_sales,CAST(0 AS BIGINT)) AS partial_healthy_sales
 FROM windows w JOIN `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1 m
  ON w.model_product_id_type=m.model_product_id_type AND w.model_product_id=m.model_product_id
 AND w.color_identity_type=m.color_identity_type AND w.color_identity_value <=> m.color_identity_value
 AND w.observed_entry_date=m.observed_entry_date
 JOIN coverage c ON c.observed_entry_date=w.observed_entry_date AND c.horizon=w.horizon AND c.storeid=m.storeid
 LEFT JOIN observed o ON o.model_product_id_type=w.model_product_id_type AND o.model_product_id=w.model_product_id
  AND o.color_identity_type=w.color_identity_type AND o.color_identity_value <=> w.color_identity_value
  AND o.observed_entry_date=w.observed_entry_date AND o.horizon=w.horizon AND o.storeid=m.storeid
)
SELECT 'product_color_name_merged_stage2_20261116_v1' AS dataset_version,*,
       healthy_store_days-observed_healthy_product_days AS assumed_zero_days,
       healthy_store_days=last_day-first_day+1 AND missing_measure_days=0 AS target_complete,
       CASE WHEN healthy_store_days=last_day-first_day+1 AND missing_measure_days=0
            THEN partial_healthy_sales END AS reported_sales_target
FROM assembled;
