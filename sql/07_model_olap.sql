-- 挖掘结果回仓：按模型、月份、日类型分析测试误差。
SELECT p.model_key, d.year, d.month, t.daytype_name, COUNT(*) AS predictions,
 ROUND(AVG(p.absolute_error),3) AS mae
FROM fact_predictions p JOIN fact_rentals f USING(observation_id)
JOIN dim_date d USING(date_key) JOIN dim_daytype t USING(daytype_key)
WHERE p.split='test' GROUP BY p.model_key,d.year,d.month,t.daytype_name
ORDER BY p.model_key,d.year,d.month,t.daytype_name;
