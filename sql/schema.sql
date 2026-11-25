PRAGMA foreign_keys = ON;
CREATE TABLE dim_date (
 date_key TEXT PRIMARY KEY, year INTEGER NOT NULL, quarter INTEGER NOT NULL,
 month INTEGER NOT NULL, day INTEGER NOT NULL, weekday INTEGER NOT NULL
);
CREATE TABLE dim_hour (
 hour_key INTEGER PRIMARY KEY CHECK(hour_key BETWEEN 0 AND 23), period TEXT NOT NULL
);
CREATE TABLE dim_weather (weather_key INTEGER PRIMARY KEY, weather_name TEXT NOT NULL);
CREATE TABLE dim_season (season_key INTEGER PRIMARY KEY, season_name TEXT NOT NULL);
CREATE TABLE dim_daytype (daytype_key INTEGER PRIMARY KEY, daytype_name TEXT NOT NULL);
CREATE TABLE fact_rentals (
 observation_id INTEGER PRIMARY KEY, timestamp TEXT NOT NULL UNIQUE,
 date_key TEXT NOT NULL REFERENCES dim_date(date_key),
 hour_key INTEGER NOT NULL REFERENCES dim_hour(hour_key),
 weather_key INTEGER NOT NULL REFERENCES dim_weather(weather_key),
 season_key INTEGER NOT NULL REFERENCES dim_season(season_key),
 daytype_key INTEGER NOT NULL REFERENCES dim_daytype(daytype_key),
 temp REAL NOT NULL, atemp REAL NOT NULL, hum REAL NOT NULL, windspeed REAL NOT NULL,
 casual INTEGER NOT NULL CHECK(casual >= 0),
 registered INTEGER NOT NULL CHECK(registered >= 0),
 cnt INTEGER NOT NULL CHECK(cnt >= 0 AND cnt = casual + registered)
);
CREATE INDEX idx_rentals_date ON fact_rentals(date_key);
CREATE INDEX idx_rentals_dimensions ON fact_rentals(season_key, daytype_key, weather_key);
CREATE TABLE etl_log (
 run_id INTEGER PRIMARY KEY, completed_utc TEXT NOT NULL, source_sha256 TEXT NOT NULL,
 source_rows INTEGER NOT NULL, loaded_rows INTEGER NOT NULL, total_rentals INTEGER NOT NULL
);
CREATE TABLE dim_model (model_key TEXT PRIMARY KEY, task TEXT NOT NULL, selected_on_validation INTEGER NOT NULL);
CREATE TABLE fact_predictions (
 model_key TEXT NOT NULL REFERENCES dim_model(model_key),
 observation_id INTEGER NOT NULL REFERENCES fact_rentals(observation_id),
 split TEXT NOT NULL, actual REAL NOT NULL, prediction REAL NOT NULL,
 absolute_error REAL NOT NULL, PRIMARY KEY(model_key, observation_id, split)
);
CREATE VIEW rental_cube_base AS
SELECT f.*, d.year, d.quarter, d.month, d.weekday, h.period,
 w.weather_name, s.season_name, t.daytype_name
FROM fact_rentals f JOIN dim_date d USING(date_key)
JOIN dim_hour h ON f.hour_key=h.hour_key
JOIN dim_weather w USING(weather_key)
JOIN dim_season s USING(season_key)
JOIN dim_daytype t USING(daytype_key);
