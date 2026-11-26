-- Seven-day signal at entry-store grain. Incomplete observed windows stay null;
-- the partial amount and coverage are available separately for downstream masks.
CREATE OR REPLACE VIEW `msba-capstone-team13`.modelling.model_week1_signal_name_merged_v1 AS
WITH entries AS (
 SELECT * FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1
 WHERE scope_status='preliminary_in_scope' AND date_add(observed_entry_date,6)<=DATE '2026-05-31'
), entry_stores AS (
 SELECT DISTINCT observed_entry_date,storeid
 FROM `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1
 WHERE observed_entry_date IN (SELECT DISTINCT observed_entry_date FROM entries)
), coverage AS (
 SELECT e.observed_entry_date,e.storeid,count(s.dt) recorded_days,
        count_if(s.network_health_status='healthy') healthy_days
 FROM entry_stores e LEFT JOIN `msba-capstone-team13`.modelling.model_store_day s
   ON s.dataset_version='stage1_20261112_v1' AND s.storeid=e.storeid
  AND s.dt BETWEEN e.observed_entry_date AND date_add(e.observed_entry_date,6)
 GROUP BY ALL
), observations AS (
 SELECT c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,
        c.observed_entry_date,p.storeid,count_if(s.network_health_status='healthy') observed_healthy_days,
        count_if(s.network_health_status='healthy' AND (p.missing_sales_observations>0 OR p.reported_sales IS NULL)) missing_measure_days,
        sum(CASE WHEN s.network_health_status='healthy' THEN p.reported_sales END) partial_healthy_sales
 FROM entries c JOIN `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 p
   ON c.model_product_id_type=p.model_product_id_type AND c.model_product_id=p.model_product_id
  AND c.color_identity_type=p.color_identity_type AND c.color_identity_value <=> p.color_identity_value
  AND p.currency='USD' AND p.dt BETWEEN c.observed_entry_date AND date_add(c.observed_entry_date,6)
 LEFT JOIN `msba-capstone-team13`.modelling.model_store_day s
   ON s.dataset_version='stage1_20261112_v1' AND s.storeid=p.storeid AND s.dt=p.dt
 GROUP BY ALL
), assembled AS (
 SELECT c.dataset_version,c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,
        c.observed_entry_date,m.storeid,v.recorded_days,v.healthy_days,7-v.recorded_days missing_store_days,
        coalesce(o.observed_healthy_days,0) observed_healthy_days,coalesce(o.missing_measure_days,0) missing_measure_days,
        coalesce(o.partial_healthy_sales,CAST(0 AS BIGINT)) partial_healthy_sales
 FROM entries c JOIN `msba-capstone-team13`.modelling.model_launch_store_name_merged_v1 m
   ON c.model_product_id_type=m.model_product_id_type AND c.model_product_id=m.model_product_id
  AND c.color_identity_type=m.color_identity_type AND c.color_identity_value <=> m.color_identity_value
  AND c.observed_entry_date=m.observed_entry_date
 JOIN coverage v ON v.observed_entry_date=c.observed_entry_date AND v.storeid=m.storeid
 LEFT JOIN observations o ON o.model_product_id_type=c.model_product_id_type AND o.model_product_id=c.model_product_id
  AND o.color_identity_type=c.color_identity_type AND o.color_identity_value <=> c.color_identity_value
  AND o.observed_entry_date=c.observed_entry_date AND o.storeid=m.storeid
)
SELECT *,healthy_days-observed_healthy_days assumed_zero_days,
       healthy_days=7 AND missing_measure_days=0 AS signal_complete,
       CASE WHEN healthy_days=7 AND missing_measure_days=0 THEN partial_healthy_sales END AS week1_reported_sales,
       date_add(observed_entry_date,7) prediction_cutoff
FROM assembled;
