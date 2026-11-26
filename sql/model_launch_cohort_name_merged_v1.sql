-- Recompute colorway entry and entry-day taxonomy after exact-name merging.
CREATE TABLE `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1 USING DELTA AS
WITH entries AS (
 SELECT model_product_id_type,model_product_id,color_identity_type,color_identity_value,min(dt) observed_entry_date
 FROM `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0 GROUP BY ALL
), entry_sources AS (
 SELECT DISTINCT d.model_product_id_type,d.model_product_id,d.color_identity_type,d.color_identity_value,
        d.dt observed_entry_date,d.storeid,d.currency,src.type source_type,src.id source_id
 FROM `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0 d
 JOIN entries e ON d.model_product_id_type=e.model_product_id_type AND d.model_product_id=e.model_product_id
  AND d.color_identity_type=e.color_identity_type AND d.color_identity_value <=> e.color_identity_value
  AND d.dt=e.observed_entry_date
 LATERAL VIEW explode(d.source_product_ids) s AS src
), paths AS (
 SELECT DISTINCT e.model_product_id_type,e.model_product_id,e.color_identity_type,e.color_identity_value,
        e.observed_entry_date,f.breadcrumbs,
        CASE WHEN f.breadcrumbs RLIKE '^Women.s Clothes > ' THEN 'women'
             WHEN f.breadcrumbs RLIKE '^Men.s Clothes > ' THEN 'men'
             WHEN f.breadcrumbs RLIKE '^Accessories > ' THEN 'accessories'
             WHEN f.breadcrumbs='Shoes' THEN 'footwear' ELSE 'unknown' END line,
        CASE WHEN f.breadcrumbs RLIKE '^(Women.s Clothes|Men.s Clothes) > Sports Bras$' THEN 'Bras & Underwear'
             WHEN f.breadcrumbs RLIKE '^Women.s Clothes > Underwear$' THEN 'Bras & Underwear'
             WHEN f.breadcrumbs RLIKE '^(Women.s Clothes|Men.s Clothes|Accessories) > '
                  THEN try_element_at(split(f.breadcrumbs,' > '),2) END category
 FROM entry_sources e JOIN `msba-capstone-team13`.capstone.sales_fact VERSION AS OF 29 f
  ON f.dt=e.observed_entry_date AND f.storeid=e.storeid AND f.currency=e.currency
 AND f.product_id_type=e.source_type AND f.product_id=e.source_id
 AND CASE WHEN f.color IS NOT NULL AND trim(f.color)<>'' THEN 'color_name'
          WHEN nullif(regexp_extract(f.url,'[?&]color=[A-Z0-9]+-([A-Z0-9]+)(?:[&#]|$)',1),'') IS NOT NULL THEN 'color_code_only'
          ELSE 'unresolved' END=e.color_identity_type
 AND CASE WHEN f.color IS NOT NULL AND trim(f.color)<>'' THEN f.color
          ELSE nullif(regexp_extract(f.url,'[?&]color=[A-Z0-9]+-([A-Z0-9]+)(?:[&#]|$)',1),'') END <=> e.color_identity_value
), taxonomy AS (
 SELECT model_product_id_type,model_product_id,color_identity_type,color_identity_value,observed_entry_date,
        count(DISTINCT CASE WHEN line<>'unknown' THEN line END) known_line_count,
        count_if(line='unknown') unknown_line_paths,max(CASE WHEN line<>'unknown' THEN line END) known_line,
        count(DISTINCT category) known_category_count,count_if(category IS NULL) missing_category_paths,
        max(category) known_category
 FROM paths GROUP BY ALL
), stores AS (
 SELECT storeid,min(dt) first_date FROM `msba-capstone-team13`.modelling.model_store_day VERSION AS OF 1
 WHERE dataset_version='stage1_20261112_v1' GROUP BY storeid
), store_counts AS (
 SELECT e.observed_entry_date,count(*) entry_store_count FROM (SELECT DISTINCT observed_entry_date FROM entries) e
 JOIN stores s ON s.first_date<=e.observed_entry_date GROUP BY e.observed_entry_date
), classified AS (
 SELECT e.*,datediff(e.observed_entry_date,DATE '2024-11-10') preceding_dataset_calendar_days,
        datediff(e.observed_entry_date,DATE '2024-11-10')>=90 has_90_day_dataset_lookback,
        date_add(e.observed_entry_date,55) day_56,date_add(e.observed_entry_date,55)<=DATE '2026-05-31' has_calendar_56_day_followup,
        CASE WHEN e.color_identity_type='unresolved' THEN 'unresolved_color' WHEN e.color_identity_type='color_code_only' THEN 'code_only_color' ELSE 'named_color' END evidence_class,
        CASE WHEN t.known_line_count=1 THEN t.known_line ELSE 'unknown' END business_line,
        CASE WHEN t.known_line_count=1 AND t.known_category_count=1 AND t.missing_category_paths=0 THEN t.known_category ELSE 'unknown' END category,
        t.known_line_count,t.unknown_line_paths,t.known_category_count,t.missing_category_paths,s.entry_store_count
 FROM entries e JOIN taxonomy t ON e.model_product_id_type=t.model_product_id_type AND e.model_product_id=t.model_product_id
  AND e.color_identity_type=t.color_identity_type AND e.color_identity_value <=> t.color_identity_value
  AND e.observed_entry_date=t.observed_entry_date
 LEFT JOIN store_counts s ON e.observed_entry_date=s.observed_entry_date
), reporting AS (
 SELECT *,CASE WHEN color_identity_type='unresolved' THEN 'unresolved_color'
      WHEN business_line='unknown' THEN 'unknown_business_line'
      WHEN business_line IN ('women','men') AND color_identity_type='color_code_only' THEN 'apparel_code_only'
      WHEN business_line IN ('women','men') AND color_identity_type='color_name' THEN 'apparel_named_color'
      ELSE 'outside_apparel' END reporting_group FROM classified
)
SELECT 'product_color_name_merged_stage2_20261116_v1' dataset_version,*,
       CASE WHEN color_identity_type='unresolved' THEN 'unresolved_color' WHEN reporting_group='outside_apparel' THEN 'outside_apparel'
            WHEN NOT has_90_day_dataset_lookback THEN 'insufficient_dataset_history' ELSE 'preliminary_in_scope' END scope_status,
       'family_group_not_certified' family_status,'retrospective_identity_and_color_proxy' historical_availability_status
FROM reporting;
