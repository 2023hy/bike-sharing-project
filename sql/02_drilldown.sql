-- 下钻：选取 2012 年 9 月，查看日期与小时的细节。
SELECT date_key, hour_key, daytype_name, cnt AS rentals
FROM rental_cube_base WHERE year=2012 AND month=9 ORDER BY date_key, hour_key;
