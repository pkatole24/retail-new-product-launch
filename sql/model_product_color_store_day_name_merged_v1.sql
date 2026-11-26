CREATE TABLE `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 USING DELTA AS
-- Rebuild the released daily table directly from pinned source observations.
-- Preserve the original aggregation stages so measures and null semantics match.
WITH ids AS (
 SELECT 'style_code' typ,style_code id,model_product_group_id grp
 FROM `msba-capstone-team13`.modelling.model_product_identity_resolution_20261116 VERSION AS OF 0
 UNION SELECT 'prod_id',old_product_id,model_product_group_id
 FROM `msba-capstone-team13`.modelling.model_product_identity_resolution_20261116 VERSION AS OF 0
), source AS (
 SELECT dt,storeid,product_id_type,product_id,currency,color,
        nullif(regexp_extract(url,'[?&]color=[A-Z0-9]+-([A-Z0-9]+)(?:[&#]|$)',1),'') code,
        sales,revenue,price
 FROM `msba-capstone-team13`.capstone.sales_fact VERSION AS OF 29
), observation AS (
 SELECT dt,storeid,product_id_type,product_id,currency,color,code,count(*) source_observation_count,
        sum(sales) reported_sales,sum(revenue) reported_revenue,count_if(sales IS NULL) missing_sales_observations,
        count_if(revenue IS NULL) missing_revenue_observations,min(price) min_observed_price,max(price) max_observed_price
 FROM source GROUP BY ALL
), keyed AS (
 SELECT *,CASE WHEN color IS NOT NULL AND trim(color)<>'' THEN 'color_name'
               WHEN code IS NOT NULL THEN 'color_code_only' ELSE 'unresolved' END color_identity_type,
          CASE WHEN color IS NOT NULL AND trim(color)<>'' THEN color ELSE code END color_identity_value
 FROM observation
), source_daily AS (
 SELECT dt,storeid,product_id_type,product_id,currency,color_identity_type,color_identity_value,
        sum(source_observation_count) source_observation_count,sum(reported_sales) reported_sales,
        sum(reported_revenue) reported_revenue,sum(missing_sales_observations) missing_sales_observations,
        sum(missing_revenue_observations) missing_revenue_observations,
        min(min_observed_price) min_observed_price,max(max_observed_price) max_observed_price
 FROM keyed GROUP BY ALL
), resolved_rows AS (
 SELECT d.*,CASE WHEN i.id IS NOT NULL THEN 'resolved_group' ELSE d.product_id_type END model_type,
        coalesce(i.grp,d.product_id) model_id,i.id IS NOT NULL resolution_applied
 FROM source_daily d LEFT JOIN ids i ON d.product_id_type=i.typ AND d.product_id=i.id
), resolved AS (
 SELECT dt,storeid,model_type,model_id,currency,color_identity_type,color_identity_value,
        array_sort(collect_set(named_struct('type',product_id_type,'id',product_id))) source_product_ids,
        max(CASE WHEN resolution_applied THEN 1 ELSE 0 END)=1 resolution_applied,
        sum(source_observation_count) source_observation_count,sum(reported_sales) reported_sales,
        sum(reported_revenue) reported_revenue,sum(missing_sales_observations) missing_sales_observations,
        sum(missing_revenue_observations) missing_revenue_observations,
        min(min_observed_price) min_observed_price,max(max_observed_price) max_observed_price
 FROM resolved_rows GROUP BY ALL
), mapped AS (
 SELECT r.*,coalesce(m.model_product_id_type,r.model_type) new_type,
        coalesce(m.model_product_id,r.model_id) new_id,m.model_product_id IS NOT NULL name_merge_applied
 FROM resolved r LEFT JOIN `msba-capstone-team13`.modelling.model_product_color_name_merge_map_v1 VERSION AS OF 0 m
  ON r.model_type=m.source_model_product_id_type AND r.model_id=m.source_model_product_id
 AND r.color_identity_type='color_name' AND r.color_identity_value=m.color_identity_value
)
SELECT 'product_color_name_merged_20261116_v1' draft_version,dt,storeid,new_type model_product_id_type,
       new_id model_product_id,currency,color_identity_type,color_identity_value,
       array_sort(array_distinct(flatten(collect_list(source_product_ids)))) source_product_ids,
       max(CASE WHEN resolution_applied OR name_merge_applied THEN 1 ELSE 0 END)=1 resolution_applied,
       sum(source_observation_count) source_observation_count,sum(reported_sales) reported_sales,
       sum(reported_revenue) reported_revenue,sum(missing_sales_observations) missing_sales_observations,
       sum(missing_revenue_observations) missing_revenue_observations,
       min(min_observed_price) min_observed_price,max(max_observed_price) max_observed_price
FROM mapped GROUP BY ALL;
