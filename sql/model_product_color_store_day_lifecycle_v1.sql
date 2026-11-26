-- For the pilot, run the SELECT with first_day = 2024-11-10 and last_day = 2024-11-12.
-- For publication, retain the released bounds and prepend the CTAS below.
-- CREATE TABLE `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1
-- USING DELTA
-- CLUSTER BY (dt)
-- COMMENT 'Sparse released store-day product-color sales with dated product evidence and entry-day taxonomy.'
-- AS
--
-- Stage 2's sales_fact v29 is outside Delta's time-travel retention. Versions
-- 30–32 were one OPTIMIZE and two metadata-only SET TBLPROPERTIES commits.
-- Read v32 for prices and the attribute evidence match. Sales always come from
-- the released v0 daily table.
-- Names and sizes reflect same-day evidence across stores for the exact
-- merged colorway. Conflicting names and unsupported sizes remain NULL.
-- Entry-day subcategory requires one supported label under the released
-- business line and category. Identity and lookup-filled colors are retrospective.
-- For forecasting, restrict historical features to evidence available at
-- the cutoff. A target day's sales are the label, and its price and size
-- require independent cutoff-time evidence for use as inputs. Hypothetical
-- colorways take attributes from the forecast request. Join separate store
-- snapshots for store features and use model_store_day for coverage when
-- constructing targets for absent pairs. Absence here does not imply zero units.
WITH bounds AS (
  SELECT DATE '2024-11-10' first_day, DATE '2026-05-31' last_day
),
daily AS (
  SELECT d.* FROM `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0 d
  CROSS JOIN bounds WHERE d.dt BETWEEN first_day AND last_day
),
cohort AS (
  SELECT * FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1 VERSION AS OF 0
),
reviewed_ids AS (
  SELECT 'style_code' source_type, style_code source_id, model_product_group_id
  FROM `msba-capstone-team13`.modelling.model_product_identity_resolution_20261116 VERSION AS OF 0
  UNION
  SELECT 'prod_id', old_product_id, model_product_group_id
  FROM `msba-capstone-team13`.modelling.model_product_identity_resolution_20261116 VERSION AS OF 0
),
fact AS (
  SELECT f.*, CASE WHEN trim(f.color) <> '' THEN f.color END named_color,
    nullif(regexp_extract(f.url, '[?&]color=[A-Z0-9]+-([A-Z0-9]+)(?:[&#]|$)', 1), '') color_code
  FROM `msba-capstone-team13`.capstone.sales_fact VERSION AS OF 32 f
  CROSS JOIN bounds WHERE f.dt <= last_day
),
resolved_fact AS (
  SELECT /*+ BROADCAST(i) */ f.*,
    CASE WHEN i.source_id IS NOT NULL THEN 'resolved_group' ELSE f.product_id_type END resolved_type,
    coalesce(i.model_product_group_id, f.product_id) resolved_id,
    CASE WHEN f.named_color IS NOT NULL THEN 'color_name'
         WHEN f.color_code IS NOT NULL THEN 'color_code_only' ELSE 'unresolved' END color_identity_type,
    coalesce(f.named_color, f.color_code) color_identity_value
  FROM fact f LEFT JOIN reviewed_ids i
    ON f.product_id_type = i.source_type AND f.product_id = i.source_id
),
mapped_fact AS (
  SELECT /*+ BROADCAST(m) */ f.*,
    coalesce(m.model_product_id_type, f.resolved_type) model_product_id_type,
    coalesce(m.model_product_id, f.resolved_id) model_product_id
  FROM resolved_fact f
  LEFT JOIN `msba-capstone-team13`.modelling.model_product_color_name_merge_map_v1 VERSION AS OF 0 m
    ON f.resolved_type = m.source_model_product_id_type AND f.resolved_id = m.source_model_product_id
    AND f.color_identity_type = 'color_name' AND f.color_identity_value = m.color_identity_value
),
attribute_evidence AS (
  SELECT evidence_date, product_id_type, product_id, currency,
    struct(title, varianttitle, product_name_evidence, size, size_status,
           size_unresolved, color, breadcrumbs, observed_price) observation,
    CASE WHEN trim(product_name_evidence) <> '' THEN product_name_evidence END product_name,
    CASE WHEN size_status = 'extracted_title' AND NOT size_unresolved THEN nullif(trim(size), '') END size_label,
    business_line, category, subcategory
  FROM `msba-capstone-team13`.modelling.model_product_attribute_evidence VERSION AS OF 1
  WHERE dataset_version = 'stage1_20261112_v1' AND attribute_version = 'attributes_20261115_v1'
),
described_observations AS (
  -- The fact supplies store and URL-derived color, which the attribute release omits.
  -- Match the complete distinct evidence projection with null-safe equality.
  SELECT f.dt, f.model_product_id_type, f.model_product_id, f.color_identity_type, f.color_identity_value,
    a.product_name, a.size_label, a.business_line, a.category, a.subcategory,
    a.evidence_date IS NOT NULL has_attribute_evidence
  FROM mapped_fact f LEFT JOIN attribute_evidence a
    ON f.dt = a.evidence_date AND f.product_id_type = a.product_id_type AND f.product_id = a.product_id
    AND f.currency <=> a.currency
    AND struct(f.title, f.varianttitle, f.product_title AS product_name_evidence, f.size, f.size_status,
               f.size_unresolved, f.color, f.breadcrumbs, f.price AS observed_price) <=> a.observation
),
day_attributes AS (
  SELECT dt, model_product_id_type, model_product_id, color_identity_type, color_identity_value,
    CASE WHEN count(DISTINCT product_name) = 1 THEN max(product_name) END product_name,
    CASE WHEN count(size_label) > 0 THEN array_sort(collect_set(size_label)) END size_labels,
    count(DISTINCT product_name) name_count, count(*) described_count,
    count_if(NOT has_attribute_evidence) unmatched_attribute_observations
  FROM described_observations CROSS JOIN bounds WHERE dt BETWEEN first_day AND last_day
  GROUP BY dt, model_product_id_type, model_product_id, color_identity_type, color_identity_value
),
entry_subcategory AS (
  SELECT c.model_product_id_type, c.model_product_id, c.color_identity_type, c.color_identity_value,
    CASE WHEN count(DISTINCT a.subcategory) = 1 THEN max(a.subcategory) END subcategory
  FROM cohort c JOIN described_observations a
    ON c.observed_entry_date = a.dt AND c.model_product_id_type = a.model_product_id_type
    AND c.model_product_id = a.model_product_id AND c.color_identity_type = a.color_identity_type
    AND c.color_identity_value <=> a.color_identity_value
    AND c.business_line = a.business_line AND c.category = a.category
  GROUP BY c.model_product_id_type, c.model_product_id, c.color_identity_type, c.color_identity_value
),
daily_price AS (
  -- Each retained fact observation contributes once, without sales weighting.
  SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency,
    median(CASE WHEN price >= 0 AND price < double('Infinity') THEN price END) representative_observed_price,
    count(*) price_source_observations,
    array_sort(collect_set(named_struct('type', product_id_type, 'id', product_id))) price_source_ids,
    sum(sales) fact_reported_sales, count_if(sales IS NULL) fact_missing_sales,
    min(price) fact_min_price, max(price) fact_max_price
  FROM mapped_fact CROSS JOIN bounds WHERE dt BETWEEN first_day AND last_day
  GROUP BY dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency
),
candidate AS (
  SELECT d.dt, d.storeid, d.model_product_id_type, d.model_product_id, d.color_identity_type, d.color_identity_value, d.currency,
    c.observed_entry_date, datediff(d.dt, c.observed_entry_date) days_since_observed_entry,
    datediff(d.dt, c.observed_entry_date) + 1 lifecycle_day,
    a.product_name, a.size_labels, c.business_line, c.category, e.subcategory,
    CASE WHEN p.price_source_observations = d.source_observation_count AND p.price_source_ids = d.source_product_ids
         THEN p.representative_observed_price END representative_observed_price,
    c.reporting_group, c.scope_status, d.reported_sales,
    coalesce(d.missing_sales_observations = 0 AND d.reported_sales IS NOT NULL, false) sales_label_complete,
    d.source_product_ids
  FROM daily d
  LEFT JOIN cohort c ON d.model_product_id_type = c.model_product_id_type
    AND d.model_product_id = c.model_product_id AND d.color_identity_type = c.color_identity_type
    AND d.color_identity_value <=> c.color_identity_value
  LEFT JOIN day_attributes a ON d.dt = a.dt AND d.model_product_id_type = a.model_product_id_type
    AND d.model_product_id = a.model_product_id AND d.color_identity_type = a.color_identity_type
    AND d.color_identity_value <=> a.color_identity_value
  LEFT JOIN entry_subcategory e ON d.model_product_id_type = e.model_product_id_type
    AND d.model_product_id = e.model_product_id AND d.color_identity_type = e.color_identity_type
    AND d.color_identity_value <=> e.color_identity_value
  LEFT JOIN daily_price p ON d.dt = p.dt AND d.storeid = p.storeid AND d.currency <=> p.currency
    AND d.model_product_id_type = p.model_product_id_type AND d.model_product_id = p.model_product_id
    AND d.color_identity_type = p.color_identity_type AND d.color_identity_value <=> p.color_identity_value
)
SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency,
  observed_entry_date, days_since_observed_entry, lifecycle_day,
  product_name, size_labels, business_line, category, subcategory,
  representative_observed_price, representative_observed_price IS NULL price_missing,
  reporting_group, scope_status, reported_sales, sales_label_complete, source_product_ids
FROM candidate;
