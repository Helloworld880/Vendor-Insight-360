-- ============================================================================
-- Analytical SQL — Vendor Insight360
-- Run against Data layer/vendors.db (SQLite ≥ 3.25 for window functions).
-- Each query answers a concrete business question; results power the
-- dashboard pages and the case-study writeup.
-- ============================================================================

-- 1. Quarter-over-quarter spend change per vendor (LAG window function)
--    Business question: whose spend is accelerating fastest?
SELECT
    vendor_id,
    period,
    total_spend,
    LAG(total_spend) OVER (PARTITION BY vendor_id ORDER BY period)        AS prev_spend,
    ROUND(
        100.0 * (total_spend - LAG(total_spend) OVER (PARTITION BY vendor_id ORDER BY period))
        / NULLIF(LAG(total_spend) OVER (PARTITION BY vendor_id ORDER BY period), 0), 1
    )                                                                      AS qoq_spend_change_pct
FROM financial_metrics
ORDER BY vendor_id, period;

-- 2. Vendor performance ranking within category (RANK + PERCENT_RANK)
--    Business question: who is best-in-class per category?
SELECT
    v.name,
    v.category,
    ROUND(AVG(pm.overall_score), 1)                                        AS avg_score,
    RANK()         OVER (PARTITION BY v.category ORDER BY AVG(pm.overall_score) DESC) AS rank_in_category,
    ROUND(PERCENT_RANK() OVER (PARTITION BY v.category ORDER BY AVG(pm.overall_score)), 2) AS pctile_in_category
FROM vendors v
JOIN performance_metrics pm ON pm.vendor_id = v.id
GROUP BY v.id, v.name, v.category
ORDER BY v.category, rank_in_category;

-- 3. Three-month rolling average performance (frame-based window)
--    Business question: smooth out monthly noise to see real trends.
SELECT
    vendor_id,
    metric_date,
    overall_score,
    ROUND(AVG(overall_score) OVER (
        PARTITION BY vendor_id
        ORDER BY metric_date
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 1)                                                                  AS rolling_3mo_score
FROM performance_metrics
ORDER BY vendor_id, metric_date;

-- 4. Spend concentration: cumulative share of top vendors (CTE + window SUM)
--    Business question: how exposed are we to our biggest vendors?
WITH vendor_spend AS (
    SELECT vendor_id, SUM(total_spend) AS spend
    FROM financial_metrics
    GROUP BY vendor_id
)
SELECT
    vendor_id,
    spend,
    ROUND(100.0 * spend / SUM(spend) OVER (), 2)                           AS pct_of_total,
    ROUND(100.0 * SUM(spend) OVER (ORDER BY spend DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
          / SUM(spend) OVER (), 1)                                         AS cumulative_pct
FROM vendor_spend
ORDER BY spend DESC;

-- 5. Risk-level transitions between consecutive assessments (LAG + CASE)
--    Business question: which vendors are deteriorating right now?
SELECT
    vendor_id,
    assessment_date,
    risk_level,
    LAG(risk_level) OVER (PARTITION BY vendor_id ORDER BY assessment_date) AS prev_level,
    CASE
        WHEN risk_level = 'High'
         AND LAG(risk_level) OVER (PARTITION BY vendor_id ORDER BY assessment_date) IN ('Low','Medium')
        THEN 'ESCALATED'
        WHEN risk_level = 'Low'
         AND LAG(risk_level) OVER (PARTITION BY vendor_id ORDER BY assessment_date) IN ('High','Medium')
        THEN 'IMPROVED'
        ELSE 'STABLE'
    END                                                                    AS transition
FROM risk_assessments
ORDER BY vendor_id, assessment_date;

-- 6. Performance vs category benchmark gap (CTE join)
--    Business question: who under-delivers against their peer benchmark?
WITH category_avg AS (
    SELECT v.category, AVG(pm.overall_score) AS category_score
    FROM vendors v
    JOIN performance_metrics pm ON pm.vendor_id = v.id
    GROUP BY v.category
)
SELECT
    v.name,
    v.category,
    ROUND(AVG(pm.overall_score), 1)                                        AS vendor_score,
    ROUND(ca.category_score, 1)                                            AS category_avg,
    ROUND(AVG(pm.overall_score) - ca.category_score, 1)                    AS gap_vs_peers
FROM vendors v
JOIN performance_metrics pm ON pm.vendor_id = v.id
JOIN category_avg ca        ON ca.category = v.category
GROUP BY v.id, v.name, v.category, ca.category_score
HAVING gap_vs_peers < 0
ORDER BY gap_vs_peers ASC;

-- 7. Month-over-month portfolio health (aggregate + LAG on aggregate)
--    Business question: is the whole portfolio improving or degrading?
WITH monthly AS (
    SELECT
        strftime('%Y-%m', metric_date)  AS month,
        AVG(overall_score)              AS avg_score,
        AVG(defect_rate_pct)            AS avg_defect
    FROM performance_metrics
    GROUP BY strftime('%Y-%m', metric_date)
)
SELECT
    month,
    ROUND(avg_score, 1)                                                    AS avg_score,
    ROUND(avg_score - LAG(avg_score) OVER (ORDER BY month), 2)             AS mom_change,
    ROUND(avg_defect, 2)                                                   AS avg_defect_rate
FROM monthly
ORDER BY month;

-- 8. Vendors overdue for a risk reassessment (date arithmetic + anti-pattern guard)
--    Business question: where are our blind spots?
SELECT
    v.name,
    MAX(ra.assessment_date)                                                AS last_assessed,
    CAST(julianday('now') - julianday(MAX(ra.assessment_date)) AS INT)     AS days_since_assessment
FROM vendors v
LEFT JOIN risk_assessments ra ON ra.vendor_id = v.id
GROUP BY v.id, v.name
HAVING days_since_assessment > 90 OR last_assessed IS NULL
ORDER BY days_since_assessment DESC;

-- 9. Cost-savings realization by category (conditional aggregation)
--    Business question: which categories actually deliver promised savings?
SELECT
    v.category,
    COUNT(DISTINCT v.id)                                                   AS vendors,
    ROUND(SUM(fm.cost_savings), 0)                                         AS total_savings,
    ROUND(SUM(fm.total_spend), 0)                                          AS total_spend,
    ROUND(100.0 * SUM(fm.cost_savings) / NULLIF(SUM(fm.total_spend), 0), 2) AS savings_rate_pct,
    SUM(CASE WHEN fm.cost_savings > 0 THEN 1 ELSE 0 END)                   AS quarters_with_savings
FROM vendors v
JOIN financial_metrics fm ON fm.vendor_id = v.id
GROUP BY v.category
ORDER BY savings_rate_pct DESC;

-- 10. NTILE-based spend segmentation (quartile buckets in pure SQL)
--     Business question: quick spend-tier segmentation without Python.
WITH vendor_spend AS (
    SELECT vendor_id, SUM(total_spend) AS spend
    FROM financial_metrics
    GROUP BY vendor_id
)
SELECT
    vendor_id,
    spend,
    NTILE(4) OVER (ORDER BY spend)                                         AS spend_quartile,
    CASE NTILE(4) OVER (ORDER BY spend)
        WHEN 4 THEN 'Strategic'
        WHEN 3 THEN 'Core'
        WHEN 2 THEN 'Transactional'
        ELSE 'Tail'
    END                                                                    AS spend_tier
FROM vendor_spend
ORDER BY spend DESC;
