-- Input is the sparse product/store/day aggregate for one release. Missing
-- calendar dates enter the health baseline as zero network activity, never as
-- synthetic store observations. The current date is excluded from its baseline.
WITH store_day AS (
  SELECT dataset_version, dt, storeid, sum(reported_sales) AS reported_sales,
         sum(source_observation_count) AS source_observation_count,
         sum(missing_sales_observations) AS missing_sales_observations
  FROM `msba-capstone-team13`.modelling.model_product_store_day
  WHERE dataset_version = :dataset_version
  GROUP BY dataset_version, dt, storeid
), bounds AS (
  SELECT min(dt) AS first_date, max(dt) AS last_date FROM store_day
), calendar AS (
  SELECT explode(sequence(first_date, last_date, interval 1 day)) AS dt
  FROM bounds
), network AS (
  SELECT c.dt, count(s.storeid) AS active_stores,
         coalesce(sum(s.source_observation_count), 0) AS observation_count
  FROM calendar c LEFT JOIN store_day s ON c.dt = s.dt
  GROUP BY c.dt
), history AS (
  SELECT *, count(*) OVER prior AS prior_calendar_days,
         percentile(active_stores, 0.5) OVER prior AS median_active_stores,
         percentile(observation_count, 0.5) OVER prior AS median_observations
  FROM network
  WINDOW prior AS (ORDER BY dt ROWS BETWEEN 28 PRECEDING AND 1 PRECEDING)
), health AS (
  SELECT *, CASE WHEN prior_calendar_days < 28 THEN 'insufficient_history'
                 WHEN median_active_stores = 0 OR median_observations = 0 THEN 'insufficient_baseline'
                 WHEN active_stores < 0.5 * median_active_stores
                   OR observation_count < 0.5 * median_observations THEN 'unhealthy'
                 ELSE 'healthy' END AS network_health_status
  FROM history
)
SELECT s.*, true AS store_present, h.network_health_status,
       h.prior_calendar_days, h.active_stores, h.observation_count AS network_observation_count,
       h.median_active_stores, h.median_observations
FROM store_day s JOIN health h ON s.dt = h.dt;
