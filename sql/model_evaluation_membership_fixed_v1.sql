-- Provisional time roles and source-product lineage for both evaluation modes.
CREATE TABLE `msba-capstone-team13`.modelling.model_evaluation_membership_fixed_v1 USING DELTA AS
WITH product_first AS (
 SELECT coalesce(k.evaluation_product_group,to_json(named_struct('type',c.source_model_product_id_type,'id',c.source_model_product_id))) evaluation_product_group,
        min(c.source_color_entry_date) first_observed
 FROM `msba-capstone-team13`.modelling.model_colorway_source_product_name_merged_v1 c
 LEFT JOIN `msba-capstone-team13`.modelling.model_evaluation_product_link_components_v1 k
  ON c.source_model_product_id_type=k.source_model_product_id_type AND c.source_model_product_id=k.source_model_product_id
 GROUP BY ALL
), lineage AS (
 SELECT l.model_product_id_type,l.model_product_id,l.color_identity_type,l.color_identity_value,
        array_sort(collect_set(named_struct('type',l.source_model_product_id_type,'id',l.source_model_product_id))) source_products,
        array_sort(collect_set(coalesce(k.evaluation_product_group,
          to_json(named_struct('type',l.source_model_product_id_type,'id',l.source_model_product_id))))) evaluation_product_groups,
        min(p.first_observed) first_product_observed_date
 FROM `msba-capstone-team13`.modelling.model_colorway_source_product_name_merged_v1 l
 LEFT JOIN `msba-capstone-team13`.modelling.model_evaluation_product_link_components_v1 k
  ON l.source_model_product_id_type=k.source_model_product_id_type AND l.source_model_product_id=k.source_model_product_id
 JOIN product_first p ON p.evaluation_product_group=coalesce(k.evaluation_product_group,
          to_json(named_struct('type',l.source_model_product_id_type,'id',l.source_model_product_id)))
 GROUP BY ALL
), stages AS (
 SELECT * FROM VALUES ('prelaunch','1_14',1,14),('prelaunch','15_28',15,28),('prelaunch','1_28',1,28),
  ('prelaunch','8_28',8,28),('prelaunch','8_56',8,56),('week1','8_28',8,28),('week1','29_56',29,56),('week1','8_56',8,56)
 AS h(forecast_stage,horizon,first_day,last_day)
)
SELECT c.dataset_version,c.model_product_id_type,c.model_product_id,c.color_identity_type,c.color_identity_value,
       c.observed_entry_date,c.reporting_group,c.scope_status,c.family_status,l.source_products,l.evaluation_product_groups,l.first_product_observed_date,
       l.first_product_observed_date<c.observed_entry_date product_seen_before_entry,
       s.forecast_stage,s.horizon,s.first_day,s.last_day,
       CASE WHEN s.forecast_stage='prelaunch' THEN c.observed_entry_date ELSE date_add(c.observed_entry_date,7) END prediction_cutoff,
       date_add(c.observed_entry_date,s.first_day-1) window_start,date_add(c.observed_entry_date,s.last_day-1) window_end,
       CASE WHEN c.observed_entry_date BETWEEN DATE '2025-09-01' AND DATE '2025-11-30' THEN 'development'
            WHEN c.observed_entry_date BETWEEN DATE '2025-12-01' AND DATE '2026-02-28' THEN 'diagnostic'
            WHEN c.observed_entry_date BETWEEN DATE '2026-03-01' AND DATE '2026-04-30' THEN 'locked_test'
            ELSE 'outside_evaluation_periods' END evaluation_period,
       date_add(c.observed_entry_date,s.last_day-1)<=DATE '2026-05-31' calendar_mature,
       'preliminary_not_certified' membership_status
FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1 VERSION AS OF 0 c
JOIN lineage l ON l.model_product_id_type=c.model_product_id_type AND l.model_product_id=c.model_product_id
 AND l.color_identity_type=c.color_identity_type AND l.color_identity_value <=> c.color_identity_value
CROSS JOIN stages s;
