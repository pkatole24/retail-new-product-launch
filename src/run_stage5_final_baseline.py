"""Freeze the released categorical calibrators and issue final store forecasts."""
import argparse
import ast

from audit_stage5_final_inputs import CUTOFF, LAST_ENTRY, OUTPUT, RELEASED_TARGET
from fetch_weekly_lightgbm_sources import ROOT, SCHEMA, fetch
from run_weekly_lightgbm_1_14 import check, write_json


def released_sql():
    path = ROOT / "notebooks/store_level_baseline_results.py"
    source = ast.parse(path.read_text(encoding="utf-8"))
    history, week = None, None
    for node in source.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "HISTORY_QUERY" for target in node.targets
        ):
            history = ast.literal_eval(node.value)
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and node.value.args:
            arg = node.value.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                marker = "CREATE OR REPLACE TEMP VIEW lifecycle_week1_signal AS\n"
                if marker in arg.value:
                    week = arg.value.split(marker, 1)[1].strip().rstrip(";")
    check(history and week, "Released categorical feature SQL is missing")
    old_scope = "AND evaluation_period<>'locked_test' AND observed_entry_date<=DATE '2025-11-30'"
    check(old_scope in history, "Released baseline history scope changed")
    history = history.replace(old_scope, f"AND observed_entry_date<=DATE '{LAST_ENTRY}'")
    return history, week


def query(name, horizon, days, week_one, strict):
    history, week = released_sql()
    delay = 7 if week_one else 0
    target_start = 14 if horizon == "15_28" else delay
    fitting_day = f"date_add(b.observed_entry_date,{delay})" if strict else f"DATE '{CUTOFF}'"
    admissible = "NOT arrays_overlap(t.groups,d.excluded)" if strict else "true"
    early = "w.week1_reported_sales" if week_one else "CAST(NULL AS BIGINT)"
    early_join = """
      LEFT JOIN early w ON w.model_product_id_type=b.model_product_id_type
        AND w.model_product_id=b.model_product_id AND w.color_identity_type=b.color_identity_type
        AND w.color_identity_value <=> b.color_identity_value AND w.observed_entry_date=b.observed_entry_date
        AND w.storeid=b.storeid
    """ if week_one else ""
    early_cte = f"early AS ({week})," if week_one else ""
    return f"""
    WITH {early_cte} proxies AS ({history}), labeled AS (
     SELECT b.model_product_id_type,b.model_product_id,b.color_identity_type,b.color_identity_value,
       b.observed_entry_date,b.storeid,{fitting_day} AS fit_group,b.evaluation_product_groups AS groups,
       named_struct('line',b.business_line,'category',b.category,
                    'subcategory',b.subcategory,'source',b.taxonomy_source) AS taxonomy,
       t.window_end,t.reported_sales_target AS actual_units,
       b.daily_proxy*{days}D AS proxy,{early} AS first_week_sales
     FROM proxies b JOIN {RELEASED_TARGET} t
       ON t.model_product_id_type=b.model_product_id_type AND t.model_product_id=b.model_product_id
       AND t.color_identity_type=b.color_identity_type AND t.color_identity_value <=> b.color_identity_value
       AND t.observed_entry_date=b.observed_entry_date AND t.storeid=b.storeid AND t.horizon='{horizon}'
     {early_join}
     WHERE t.window_start=date_add(b.observed_entry_date,{target_start})
    ), train AS (
     SELECT groups,taxonomy,max(window_end) AS latest_outcome,
       sum(actual_units) FILTER (WHERE proxy>0) AS sales,
       sum(proxy) FILTER (WHERE proxy>0) AS proxy,
       sum(actual_units) FILTER (WHERE first_week_sales IS NOT NULL) AS later_sales,
       sum(first_week_sales) AS early_sales
     FROM labeled WHERE window_end<DATE '{CUTOFF}' AND actual_units IS NOT NULL
     GROUP BY groups,taxonomy
    ), heldout AS (
     SELECT fit_group,array_distinct(flatten(collect_set(groups))) AS excluded
     FROM (SELECT DISTINCT fit_group,groups FROM labeled WHERE observed_entry_date>=DATE '{CUTOFF}')
     GROUP BY fit_group
    ), qualifying AS (
     SELECT d.fit_group,t.* FROM heldout d CROSS JOIN train t WHERE {admissible}
    ), pooled AS (
     SELECT fit_group,try_divide(sum(sales),sum(proxy)) AS pooled_factor,
       try_divide(sum(later_sales),sum(early_sales)) AS early_factor,max(latest_outcome) AS latest_outcome
     FROM qualifying GROUP BY fit_group
    ), factors AS (
     SELECT fit_group,taxonomy,try_divide(sum(sales),sum(proxy)) AS factor
     FROM qualifying GROUP BY fit_group,taxonomy
    )
    SELECT r.model_product_id_type,r.model_product_id,r.color_identity_type,r.color_identity_value,
      r.observed_entry_date,r.storeid,date_add(r.observed_entry_date,{delay}) AS forecast_date,
      r.actual_units,r.proxy*coalesce(f.factor,p.pooled_factor) AS prelaunch_component,
      r.first_week_sales*p.early_factor AS early_component,p.latest_outcome AS latest_training_outcome,
      '{name}' AS forecast_window
    FROM labeled r JOIN pooled p ON p.fit_group=r.fit_group
    LEFT JOIN factors f ON r.fit_group=f.fit_group AND r.taxonomy=f.taxonomy
    WHERE r.observed_entry_date BETWEEN DATE '{CUTOFF}' AND DATE '{LAST_ENTRY}'
    """


def run(mode):
    strict = mode == "strict"
    destination = OUTPUT / "baseline" / mode
    windows = [("prelaunch_1_14", "1_14", 14, False),
               ("prelaunch_15_28", "15_28", 14, False), ("week1_8_28", "8_28", 21, True)]
    queries = {name: query(name, horizon, days, week_one, strict)
               for name, horizon, days, week_one in windows}
    destination.mkdir(parents=True, exist_ok=True)
    for name, statement in queries.items():
        (destination / f"{name}.sql").write_text(statement, encoding="utf-8")
    fetch(queries, destination)
    write_json(OUTPUT / f"baseline_{mode}.done.json", {"mode": mode, "fit_cutoff": CUTOFF})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("main", "strict"))
    run(parser.parse_args().mode)
