-- 切片：固定年为 2012，比较日类型和小时。
SELECT daytype_name, hour_key, COUNT(*) AS observed_hours,
 ROUND(AVG(cnt),2) AS mean_rentals FROM rental_cube_base
WHERE year=2012 GROUP BY daytype_name, hour_key ORDER BY daytype_name, hour_key;
