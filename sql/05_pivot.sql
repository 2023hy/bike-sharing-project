-- 旋转：日类型由行转列；按已观测小时的均值比较。
SELECT hour_key,
 ROUND(AVG(CASE WHEN daytype_key=1 THEN cnt END),2) AS workingday,
 ROUND(AVG(CASE WHEN daytype_key=2 THEN cnt END),2) AS weekend,
 ROUND(AVG(CASE WHEN daytype_key=3 THEN cnt END),2) AS holiday
FROM rental_cube_base GROUP BY hour_key ORDER BY hour_key;
