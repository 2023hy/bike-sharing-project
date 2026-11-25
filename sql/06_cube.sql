-- 二维 CUBE：季节编码 × 日类型，包含全部四种 grouping sets。
SELECT season_name, daytype_name, 0 AS season_grouped, 0 AS daytype_grouped,
 COUNT(*) AS observed_hours, SUM(cnt) AS rentals FROM rental_cube_base GROUP BY season_name,daytype_name
UNION ALL
SELECT season_name,NULL,0,1,COUNT(*),SUM(cnt) FROM rental_cube_base GROUP BY season_name
UNION ALL
SELECT NULL,daytype_name,1,0,COUNT(*),SUM(cnt) FROM rental_cube_base GROUP BY daytype_name
UNION ALL
SELECT NULL,NULL,1,1,COUNT(*),SUM(cnt) FROM rental_cube_base;
