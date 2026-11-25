-- 切块：2012 年 7~9 月、工作日、晴/薄雾天气、早晚通勤时段。
SELECT month, weather_name, period, COUNT(*) AS observed_hours,
 SUM(cnt) AS rentals, ROUND(AVG(cnt),2) AS mean_rentals
FROM rental_cube_base WHERE year=2012 AND month BETWEEN 7 AND 9
AND daytype_key=1 AND weather_key IN (1,2) AND hour_key IN (7,8,9,16,17,18,19)
GROUP BY month, weather_name, period ORDER BY month, weather_name, period;
