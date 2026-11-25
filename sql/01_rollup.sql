-- 上卷：月 -> 年 -> 总计。SQLite 不支持原生 ROLLUP，以 UNION ALL 实现同义聚合。
SELECT 'month' AS level, year, month, COUNT(*) AS observed_hours,
 SUM(cnt) AS rentals, ROUND(AVG(cnt),2) AS mean_per_observed_hour
FROM rental_cube_base GROUP BY year, month
UNION ALL
SELECT 'year', year, NULL, COUNT(*), SUM(cnt), ROUND(AVG(cnt),2)
FROM rental_cube_base GROUP BY year
UNION ALL
SELECT 'all', NULL, NULL, COUNT(*), SUM(cnt), ROUND(AVG(cnt),2) FROM rental_cube_base;
