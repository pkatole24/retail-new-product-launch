-- For a read-only pilot or full candidate, replace the first CTE's SELECT
-- with the corresponding build SELECT (including its WITH clause).
-- For a pilot, give released the same date bounds.
--
-- Prepublication Databricks statement IDs from the 2026-11-21 build:
-- Nov 10–12 pilot (302,575 matched rows): 01f1e5ec-8bd6-1250-bab6-331a5c314ee0
-- Full candidate (53,380,882 matched rows): 01f1e5ec-b26b-1723-bccb-b9ebe806df7b
-- Cohort entries (25,902, exact first-day match): 01f1e5ec-94a6-1b34-b120-1c50a86cf2a8
-- Fact lineage (all keys reconciled): 01f1e5ec-bb2c-17ac-9d2d-41c9f09cfda4
-- Descriptive evidence (no unmatched observations or expanded counts): 01f1e5ec-cb9a-15a8-b3a8-4a27b23d812d
-- Name, size and exact median spot checks: 01f1e5ec-c381-13fc-808a-8716c4aca6b2, 01f1e5ec-ec18-172f-be21-3968f0a352b2
-- Missing-label cases: 01f1e5ec-c462-1586-af51-3e1ad6d6988b
--
-- Managed Delta v0 committed at 2026-11-21 20:57:59.575 UTC, about 4 minutes
-- after CTAS submission (01f1e5ed-b0f1-1854-a21c-05fb746aa048).
-- Post-publication reconciliation (01f1e5ee-4b6d-1abb-8db7-846736997f5b):
-- 53,380,882 rows, 568 dates (2024-11-10 through 2026-05-31), 79,096,957 USD reported units,
-- and 53,380,882 supported medians. All key, sales, lineage, lifecycle, and date/currency checks passed.
WITH output AS (
  SELECT * FROM `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1 VERSION AS OF 0
),
released AS (
  SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency, reported_sales,
    coalesce(missing_sales_observations = 0 AND reported_sales IS NOT NULL, false) sales_label_complete,
    source_product_ids
  FROM `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0
),
combined AS (
  SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency, reported_sales, sales_label_complete, source_product_ids, 0 old_rows, 1 new_rows FROM output
  UNION ALL
  SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency, reported_sales, sales_label_complete, source_product_ids, 1 old_rows, 0 new_rows FROM released
),
key_comparison AS (
  SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency,
    sum(old_rows) old_rows, sum(new_rows) new_rows,
    count(*) distinct_payloads, sum(old_units) old_units, sum(new_units) new_units
  FROM (
    SELECT dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency, reported_sales, sales_label_complete, source_product_ids, sum(old_rows) old_rows, sum(new_rows) new_rows,
      sum(CASE WHEN old_rows = 1 THEN reported_sales END) old_units,
      sum(CASE WHEN new_rows = 1 THEN reported_sales END) new_units
    FROM combined GROUP BY ALL
  )
  GROUP BY dt, storeid, model_product_id_type, model_product_id, color_identity_type, color_identity_value, currency
),
date_currency AS (
  SELECT dt, currency, sum(old_rows) old_rows, sum(new_rows) new_rows,
    sum(old_units) old_units, sum(new_units) new_units,
    count_if(old_rows <> 1 OR new_rows <> 1 OR distinct_payloads <> 1) key_or_payload_errors,
    count_if(new_rows > 1) duplicate_output_keys,
    count_if(old_rows = 0) extra_output_keys, count_if(new_rows = 0) missing_output_keys
  FROM key_comparison GROUP BY dt, currency
),
reconciliation AS (
  SELECT sum(old_rows) released_rows, sum(new_rows) output_rows,
    sum(key_or_payload_errors) key_or_payload_errors, sum(duplicate_output_keys) duplicate_output_keys,
    sum(extra_output_keys) extra_output_keys, sum(missing_output_keys) missing_output_keys,
    count_if(old_rows <> new_rows OR NOT (old_units <=> new_units)) date_currency_errors
  FROM date_currency
),
attributes AS (
  SELECT min(dt) first_day, max(dt) last_day, count(DISTINCT dt) observed_dates,
    count_if(observed_entry_date IS NULL) unmatched_cohort_rows,
    count_if(days_since_observed_entry < 0) negative_ages,
    count_if(lifecycle_day <> days_since_observed_entry + 1) invalid_lifecycle_days,
    count_if(price_missing <> (representative_observed_price IS NULL)) invalid_price_flags,
    count_if(NOT price_missing) priced_rows, count_if(price_missing) unpriced_rows,
    count_if(product_name IS NULL) missing_name_rows, count_if(size_labels IS NULL) unknown_size_rows,
    count_if(subcategory IS NULL) unknown_subcategory_rows,
    count_if(NOT sales_label_complete) incomplete_labels, count_if(reported_sales IS NULL) null_sales_rows,
    count_if(reported_sales = 0) recorded_zero_rows, count_if(color_identity_value IS NULL) null_color_rows,
    count_if(color_identity_type = 'color_code_only') code_only_rows,
    count_if(color_identity_type = 'color_name') named_color_rows, sum(reported_sales) reported_units
  FROM output
)
SELECT * FROM reconciliation CROSS JOIN attributes;

-- Independently verify the cohort's full-history minima and cardinality.
WITH observed AS (
 SELECT model_product_id_type, model_product_id, color_identity_type, color_identity_value, min(dt) AS first_day
 FROM `msba-capstone-team13`.modelling.model_product_color_store_day_name_merged_v1 VERSION AS OF 0 GROUP BY ALL
), c AS (
 SELECT model_product_id_type, model_product_id, color_identity_type, color_identity_value,
        min(observed_entry_date) AS first_day, count(*) AS n
 FROM `msba-capstone-team13`.modelling.model_launch_cohort_name_merged_v1 VERSION AS OF 0 GROUP BY 1, 2, 3, 4
)
SELECT count(*) key_count, count_if(c.n > 1) duplicate_cohort_keys,
 count_if(c.first_day IS NULL) missing_cohort_keys, count_if(o.first_day IS NULL) extra_cohort_keys,
 count_if(NOT (c.first_day <=> o.first_day)) incorrect_first_days
FROM observed o FULL JOIN c ON o.model_product_id_type = c.model_product_id_type AND o.model_product_id = c.model_product_id
 AND o.color_identity_type = c.color_identity_type AND o.color_identity_value <=> c.color_identity_value;

-- Released rows can lack these cases, so exercise the label rule explicitly.
SELECT test_case, missing_sales_observations, reported_sales,
 coalesce(missing_sales_observations = 0 AND reported_sales IS NOT NULL, false) AS sales_label_complete
FROM VALUES ('complete_positive',0,7), ('explicit_zero',0,0), ('partial_sum',1,7),
 ('all_measures_missing',2,CAST(NULL AS BIGINT)), ('unknown_missing_count',CAST(NULL AS BIGINT),7)
AS examples(test_case, missing_sales_observations, reported_sales);

-- Spot-check null-safe unresolved and code-only color matching on their first days.
SELECT
 count_if(dt=DATE '2025-02-12' AND color_identity_value IS NULL
   AND color_identity_type='unresolved' AND observed_entry_date IS NOT NULL
   AND lifecycle_day=days_since_observed_entry+1) AS validated_unresolved_rows,
 count_if(dt=DATE '2025-11-17' AND color_identity_type='color_code_only'
   AND color_identity_value IS NOT NULL AND observed_entry_date IS NOT NULL) AS validated_code_rows,
 count_if(dt=DATE '2025-11-17' AND size_labels IS NULL) AS unknown_size_rows,
 count_if(dt=DATE '2025-11-17' AND cardinality(size_labels) = 0) AS imputed_empty_size_rows
FROM `msba-capstone-team13`.modelling.model_product_color_store_day_lifecycle_v1 VERSION AS OF 0
WHERE dt IN (DATE '2025-02-12', DATE '2025-11-17');

/*
To reconcile fact-lineage recovery, retain the build CTEs through candidate and replace their final SELECT with:

SELECT count(*) AS compared_keys,
  count_if(d.dt IS NULL) AS extra_fact_keys,
  count_if(p.dt IS NULL) AS missing_fact_keys,
  count_if(NOT (d.source_observation_count <=> p.price_source_observations)
        OR NOT (d.source_product_ids <=> p.price_source_ids)
        OR NOT (d.reported_sales <=> p.fact_reported_sales)
        OR NOT (d.missing_sales_observations <=> p.fact_missing_sales)
        OR NOT (d.min_observed_price <=> p.fact_min_price)
        OR NOT (d.max_observed_price <=> p.fact_max_price)) AS fact_lineage_errors
FROM daily d FULL JOIN daily_price p ON d.dt = p.dt AND d.storeid = p.storeid AND d.currency <=> p.currency
  AND d.model_product_id_type = p.model_product_id_type AND d.model_product_id = p.model_product_id
  AND d.color_identity_type = p.color_identity_type AND d.color_identity_value <=> p.color_identity_value;

To reconcile the descriptive join, retain the same CTEs through candidate and replace their final SELECT with the following (including the leading comma):

,
released_attribute_days AS (
 SELECT dt, model_product_id_type, model_product_id, color_identity_type, color_identity_value,
   sum(source_observation_count) AS released_observations
 FROM daily GROUP BY ALL
)
SELECT count(*) AS compared_colorway_dates,
  count_if(NOT (d.released_observations <=> a.described_count)) AS description_count_errors,
  coalesce(sum(a.unmatched_attribute_observations),0) AS unmatched_attribute_observations,
  count_if(a.name_count > 1) AS conflicting_name_colorway_dates
FROM released_attribute_days d FULL JOIN day_attributes a
  ON d.dt = a.dt AND d.model_product_id_type = a.model_product_id_type AND d.model_product_id = a.model_product_id
  AND d.color_identity_type = a.color_identity_type AND d.color_identity_value <=> a.color_identity_value;
*/
