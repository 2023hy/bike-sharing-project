"""共享单车完整数据挖掘流程。CLI 与 Notebook 共用；所有路径相对于本文件。"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import platform
from pathlib import Path
import sqlite3
import time
import urllib.request
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
    precision_score, r2_score, recall_score, roc_auc_score, silhouette_score)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'data/raw'
PROCESSED = ROOT / 'data/processed'
RESULTS = ROOT / 'results'
FIGURES = ROOT / 'reports/figures'
DB = ROOT / 'warehouse.sqlite'
SEED = 42
URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'
SHA256 = 'b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401'
TRAIN_END = pd.Timestamp('2012-07-01')
VALID_END = pd.Timestamp('2012-10-01')
CAT = ['hour', 'weekday', 'month', 'season', 'daytype']
NUM = ['trend_days', 'hour_sin', 'hour_cos', 'lag_1', 'lag_2', 'lag_24', 'lag_168',
       'roll_mean_24', 'roll_mean_168', 'roll_std_24', 'observed_past_24',
       'temp_prev', 'hum_prev', 'windspeed_prev', 'weather_prev']
FEATURES = CAT + NUM


def init_dirs():
    for p in [RAW, PROCESSED, RESULTS, FIGURES]:
        p.mkdir(parents=True, exist_ok=True)


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')


def csv_save(frame, name):
    frame.to_csv(RESULTS / f'{name}.csv', index=False, encoding='utf-8-sig')


def collect():
    """下载并核验官方压缩包；已含原始包时完全离线。只解出白名单文件。"""
    init_dirs()
    archive = RAW / 'bike_sharing.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(URL, timeout=60) as src:
            temporary.write_bytes(src.read())
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != SHA256:
            raise ValueError('下载包 SHA256 不匹配，请核验上游版本，禁止静默使用不同数据。')
        temporary.replace(archive)
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == SHA256
    with zipfile.ZipFile(archive) as z:
        for name in ['hour.csv', 'day.csv', 'Readme.txt']:
            (RAW / name).write_bytes(z.read(name))
    manifest = {
        'dataset': 'UCI Bike Sharing', 'doi': '10.24432/C5W894', 'url': URL,
        'source_page': 'https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset',
        'license': 'CC BY 4.0 (UCI dataset page)',
        'sha256_zip': SHA256,
        'files': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [RAW / 'hour.csv', RAW / 'day.csv', RAW / 'Readme.txt']},
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'original_notebook': 'forecast/plot_time_series_lagged_features.ipynb',
        'adaptation': '沿用滞后特征与时间验证思路，独立实现；使用含真实日期的 UCI 原始 CSV。',
    }
    save_json(RAW / 'manifest.json', manifest)
    return manifest


def clean_data():
    """验证而非制造脏数据；只删除完全重复行，异常业务键直接失败。"""
    raw = pd.read_csv(RAW / 'hour.csv')
    source_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    df = raw.drop_duplicates().copy()
    required = ['instant','dteday','season','yr','mnth','hr','holiday','weekday',
                'workingday','weathersit','temp','atemp','hum','windspeed','casual','registered','cnt']
    assert set(required).issubset(df.columns)
    assert df[required].isna().sum().sum() == 0, '原始字段缺失，需人工核验'
    df['dteday'] = pd.to_datetime(df.dteday, format='%Y-%m-%d')
    df['timestamp'] = df.dteday + pd.to_timedelta(df.hr, unit='h')
    df = df.sort_values('timestamp').reset_index(drop=True)
    assert df.timestamp.is_unique and df.instant.is_unique
    assert (df.cnt == df.casual + df.registered).all()
    assert (df[['casual','registered','cnt']] >= 0).all().all()
    assert df.hr.between(0,23).all() and df.weathersit.between(1,4).all()
    assert df.season.between(1,4).all()
    assert (df[['temp','atemp','hum','windspeed']].ge(0) & df[['temp','atemp','hum','windspeed']].le(1)).all().all()
    assert (df.dteday.dt.year == df.yr+2011).all() and (df.dteday.dt.month == df.mnth).all()
    assert ((df.dteday.dt.day_of_week+1)%7 == df.weekday).all()
    expected_work = ((df.weekday.isin([0,6]) == False) & (df.holiday == 0)).astype(int)
    assert (expected_work == df.workingday).all()
    df['daytype'] = np.where(df.holiday == 1, 3, np.where(df.workingday == 1, 1, 2))
    # season 的页面与压缩包说明存在命名冲突，始终保留原始编码，不擅自命名春夏秋冬。
    calendar = pd.date_range(df.timestamp.min(), df.timestamp.max(), freq='h')
    missing = calendar.difference(df.timestamp)
    pd.DataFrame({'timestamp': missing}).to_csv(PROCESSED / 'missing_hours.csv', index=False)
    days = pd.read_csv(RAW / 'day.csv', parse_dates=['dteday'])
    summed = df.groupby('dteday')[['casual','registered','cnt']].sum()
    daily_check = summed.join(days.set_index('dteday')[['casual','registered','cnt']].add_suffix('_daily'))
    assert len(daily_check) == len(days)
    assert all((daily_check[c] == daily_check[f'{c}_daily']).all() for c in ['cnt','casual','registered'])
    df.to_csv(PROCESSED / 'hour_clean.csv', index=False)
    quality = {
        'source_rows': source_rows, 'clean_rows': len(df), 'exact_duplicates_removed': duplicate_rows,
        'missing_cells': int(raw.isna().sum().sum()), 'calendar_hours': len(calendar),
        'missing_hours': len(missing), 'date_count': int(df.dteday.nunique()),
        'total_rentals': int(df.cnt.sum()), 'min_timestamp': str(df.timestamp.min()),
        'max_timestamp': str(df.timestamp.max()), 'daily_totals_reconciled': True,
        'complete_24h_days': int((df.groupby('dteday').size() == 24).sum()),
        'policy': '缺行不是零租赁；仓库只记录真实观测，监督学习不填造标签；保留极端真实高峰。',
        'metadata_note': 'UCI 网页样本数与 season/温度说明和原始文件存在差异；以实际 CSV 行数为准，season 保留编码，温度保留原始归一化值。',
    }
    save_json(RESULTS / 'data_quality.json', quality)
    coverage = pd.DataFrame({'timestamp': calendar})
    coverage['observed'] = coverage.timestamp.isin(df.timestamp).astype(int)
    coverage['month'] = coverage.timestamp.dt.strftime('%Y-%m')
    coverage = coverage.groupby('month')['observed'].agg(['sum','size']).reset_index()
    coverage.columns = ['month','observed_hours','calendar_hours']
    coverage['coverage'] = coverage.observed_hours / coverage.calendar_hours
    csv_save(coverage, 'monthly_coverage')
    return df, quality


def hour_period(hour):
    return ('night' if hour < 6 else 'morning_peak' if hour < 10 else
            'daytime' if hour < 16 else 'evening_peak' if hour < 20 else 'evening')


def build_warehouse(df):
    """原始层 -> 清洗层 -> 星型仓库 -> 分析结果层；临时库完成后原子替换。"""
    temp = DB.with_suffix('.building.sqlite')
    if temp.exists():
        temp.unlink()
    with sqlite3.connect(temp) as con:
        con.executescript((ROOT / 'sql/schema.sql').read_text())
        dates = pd.date_range(df.dteday.min(), df.dteday.max(), freq='D')
        con.executemany('INSERT INTO dim_date VALUES (?,?,?,?,?,?)',
            [(d.strftime('%Y-%m-%d'),d.year,d.quarter,d.month,d.day,(d.day_of_week+1)%7) for d in dates])
        con.executemany('INSERT INTO dim_hour VALUES (?,?)', [(h,hour_period(h)) for h in range(24)])
        con.executemany('INSERT INTO dim_weather VALUES (?,?)',
            [(1,'clear'),(2,'mist_cloudy'),(3,'light_rain_snow'),(4,'severe_weather')])
        con.executemany('INSERT INTO dim_season VALUES (?,?)', [(i,f'season_{i}') for i in range(1,5)])
        con.executemany('INSERT INTO dim_daytype VALUES (?,?)', [(1,'workingday'),(2,'weekend'),(3,'holiday')])
        records = []
        for r in df.itertuples():
            records.append((r.instant,str(r.timestamp),r.dteday.strftime('%Y-%m-%d'),r.hr,
                r.weathersit,r.season,int(r.daytype),r.temp,r.atemp,r.hum,r.windspeed,r.casual,r.registered,r.cnt))
        con.executemany('INSERT INTO fact_rentals VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', records)
        con.execute('INSERT INTO etl_log VALUES (1,?,?,?,?,?)',
            (datetime.now(timezone.utc).isoformat(),SHA256,len(df),len(records),int(df.cnt.sum())))
        assert con.execute('PRAGMA foreign_key_check').fetchall() == []
        assert con.execute('SELECT COUNT(*),SUM(cnt) FROM fact_rentals').fetchone() == (len(df),int(df.cnt.sum()))
        assert con.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    temp.replace(DB)
    return pd.DataFrame({'table':['dim_date','dim_hour','dim_weather','dim_season','dim_daytype','fact_rentals'],
                         'rows':[len(dates),24,4,4,3,len(df)]})


def run_olap():
    tables = {}
    with sqlite3.connect(DB) as con:
        for p in sorted((ROOT / 'sql').glob('0[1-6]_*.sql')):
            frame = pd.read_sql_query(p.read_text(), con)
            csv_save(frame, p.stem)
            tables[p.stem] = frame
    fig, ax = plt.subplots(figsize=(10,4))
    pivot = tables['05_pivot'].set_index('hour_key')
    pivot.plot(ax=ax, marker='o', linewidth=1.8)
    ax.set(xlabel='Hour of day', ylabel='Mean rentals per observed hour', title='OLAP: hourly demand by day type')
    save_fig(fig, 'olap_daytype')
    fig, ax = plt.subplots(figsize=(10,4))
    m = tables['01_rollup'].query("level == 'month'").copy()
    labels = [f'{int(y)}-{int(mo):02}' for y,mo in zip(m.year,m.month)]
    ax.bar(labels,m.rentals,color='#187f8c'); ax.tick_params(axis='x',rotation=60)
    ax.set(ylabel='Observed rental total', title='Monthly observed rentals (no zero filling)')
    save_fig(fig,'monthly_rentals')
    return tables


def save_fig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def from_warehouse():
    with sqlite3.connect(DB) as con:
        df = pd.read_sql_query('SELECT * FROM rental_cube_base ORDER BY timestamp', con,
                               parse_dates=['timestamp','date_key'])
    return df.rename(columns={'observation_id':'instant','date_key':'dteday','hour_key':'hr',
        'weather_key':'weathersit','season_key':'season','daytype_key':'daytype','month':'mnth'})


def make_features(df):
    """严格按真实时间对齐；t 的需求和天气特征只允许来自 t-1 及以前。"""
    observed = df.set_index('timestamp').sort_index()
    grid = observed.reindex(pd.date_range(observed.index.min(), observed.index.max(), freq='h'))
    out = pd.DataFrame(index=grid.index)
    out['timestamp'] = out.index
    out['instant'] = grid.instant
    out['cnt'] = grid.cnt
    out['hour'] = out.index.hour
    out['weekday'] = (out.index.day_of_week+1)%7
    out['month'] = out.index.month
    out['season'] = grid.season
    out['daytype'] = grid.daytype
    out['trend_days'] = (out.index-out.index.min()).total_seconds()/86400
    out['hour_sin'] = np.sin(2*np.pi*out.hour/24)
    out['hour_cos'] = np.cos(2*np.pi*out.hour/24)
    for lag in [1,2,24,168]:
        out[f'lag_{lag}'] = grid.cnt.shift(lag)
    past = grid.cnt.shift(1)
    out['roll_mean_24'] = past.rolling(24,min_periods=1).mean()
    out['roll_mean_168'] = past.rolling(168,min_periods=1).mean()
    out['roll_std_24'] = past.rolling(24,min_periods=2).std()
    out['observed_past_24'] = past.rolling(24,min_periods=1).count()
    for src,dest in [('temp','temp_prev'),('hum','hum_prev'),('windspeed','windspeed_prev'),('weathersit','weather_prev')]:
        out[dest] = grid[src].shift(1)
    out['last_observed'] = grid.cnt.ffill().shift(1)
    out = out.loc[(out.cnt.notna()) & (out.index >= out.index.min()+pd.Timedelta(hours=168))].copy()
    out['split'] = np.where(out.index < TRAIN_END, 'train', np.where(out.index < VALID_END,'validation','test'))
    assert not set(['casual','registered','cnt','instant','temp','hum','windspeed','weathersit']) & set(FEATURES)
    return out.reset_index(drop=True)


def preprocessing():
    return ColumnTransformer([
        ('calendar', OneHotEncoder(handle_unknown='ignore',sparse_output=False), CAT),
        ('numeric', Pipeline([('impute',SimpleImputer(strategy='median',add_indicator=True)),
                              ('scale',StandardScaler())]), NUM),
    ])


def pipeline(model):
    return Pipeline([('preprocess',preprocessing()),('model',model)])


def regression_models():
    return {
        'Ridge': pipeline(Ridge(alpha=10)),
        'DecisionTree': pipeline(DecisionTreeRegressor(max_depth=12,min_samples_leaf=15,random_state=SEED)),
        'RandomForest': pipeline(RandomForestRegressor(n_estimators=100,min_samples_leaf=3,max_features=0.9,n_jobs=2,random_state=SEED)),
        'HistGradientBoosting': pipeline(HistGradientBoostingRegressor(max_iter=180,max_leaf_nodes=31,
             learning_rate=0.07,l2_regularization=1,early_stopping=False,random_state=SEED)),
    }


def reg_metrics(y, pred):
    return {'MAE':mean_absolute_error(y,pred), 'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':r2_score(y,pred), 'WAPE_percent':100*float(np.abs(y-pred).sum()/np.abs(y).sum())}


def baseline(data, name):
    if name == 'LastObserved': return data.last_observed.to_numpy()
    return data.lag_168.fillna(data.lag_24).fillna(data.last_observed).to_numpy()


def supervised(data):
    train = data.query("split == 'train'")
    val = data.query("split == 'validation'")
    test = data.query("split == 'test'")
    trainval = data.query("split != 'test'")
    split_info = data.groupby('split',sort=False).agg(rows=('cnt','size'),start=('timestamp','min'),end=('timestamp','max')).reset_index()
    csv_save(split_info,'splits')
    data.to_csv(PROCESSED / 'model_features.csv', index=False)
    models = regression_models()
    val_rows, test_rows, cv_rows = [], [], []
    # 训练窗口内部扩展窗验证；每折的填补、编码、标准化只拟合该折训练段。
    for fold,(a,b) in enumerate(TimeSeriesSplit(n_splits=3).split(train),1):
        for name,model in models.items():
            fitted = clone(model).fit(train.iloc[a][FEATURES],train.iloc[a].cnt)
            pred = np.maximum(0,fitted.predict(train.iloc[b][FEATURES]))
            cv_rows.append({'fold':fold,'model':name,'train_end':str(train.iloc[a].timestamp.max()),
                'validation_start':str(train.iloc[b].timestamp.min()), **reg_metrics(train.iloc[b].cnt,pred)})
    csv_save(pd.DataFrame(cv_rows),'regression_temporal_cv')
    for name in ['LastObserved','WeeklySeasonal']:
        val_rows.append({'model':name,**reg_metrics(val.cnt,baseline(val,name))})
    for name,model in models.items():
        model.fit(train[FEATURES],train.cnt)
        val_rows.append({'model':name,**reg_metrics(val.cnt,np.maximum(0,model.predict(val[FEATURES])))})
    val_results = pd.DataFrame(val_rows).sort_values('MAE')
    selected = str(val_results.iloc[0].model)
    csv_save(val_results,'regression_validation')
    predictions = []
    fitted_models = {}
    for name in ['LastObserved','WeeklySeasonal',*models]:
        if name in models:
            fitted = clone(models[name]).fit(trainval[FEATURES],trainval.cnt)
            fitted_models[name] = fitted
            pred = np.maximum(0,fitted.predict(test[FEATURES]))
        else: pred = baseline(test,name)
        test_rows.append({'model':name,'selected_on_validation':name == selected,**reg_metrics(test.cnt,pred)})
        predictions.append(pd.DataFrame({'model':name,'instant':test.instant.astype(int),
            'timestamp':test.timestamp,'actual':test.cnt,'prediction':pred,'absolute_error':np.abs(test.cnt-pred)}))
    test_results = pd.DataFrame(test_rows).sort_values('MAE')
    all_preds = pd.concat(predictions,ignore_index=True)
    csv_save(test_results,'regression_test'); csv_save(all_preds,'regression_predictions')
    # 重要性在验证集上估计，使用只拟合训练集的模型；不据此再次挑选特征。
    best_ml = str(val_results[val_results.model.isin(models)].iloc[0].model)
    importance = permutation_importance(models[best_ml],val[FEATURES],val.cnt,
        scoring='neg_mean_absolute_error',n_repeats=3,random_state=SEED,n_jobs=1,max_samples=1000)
    imp = pd.DataFrame({'feature':FEATURES,'MAE_increase':importance.importances_mean,
                        'std':importance.importances_std}).sort_values('MAE_increase',ascending=False)
    csv_save(imp,'feature_importance_validation')
    # 分位数区间独立于点预测模型选择；报告实际覆盖率，不声称已校准。
    quantiles = []
    for q in [0.05,0.95]:
        m = pipeline(HistGradientBoostingRegressor(loss='quantile',quantile=q,max_iter=180,
            learning_rate=0.07,early_stopping=False,random_state=SEED))
        m.fit(trainval[FEATURES],trainval.cnt)
        quantiles.append(np.maximum(0,m.predict(test[FEATURES])))
    crossed = float(np.mean(quantiles[0]>quantiles[1]))
    low,high = np.minimum(*quantiles),np.maximum(*quantiles)
    interval = {'nominal_coverage':0.90,'empirical_coverage':float(np.mean((test.cnt>=low)&(test.cnt<=high))),
                'mean_width':float(np.mean(high-low)),'raw_quantile_crossing_fraction':crossed}
    save_json(RESULTS / 'prediction_interval.json',interval)
    csv_save(pd.DataFrame({'timestamp':test.timestamp,'actual':test.cnt,'q05_ordered':low,'q95_ordered':high}), 'prediction_intervals')
    high_threshold = float(train.cnt.quantile(0.75))
    class_models = {
        'LogisticRegression':pipeline(LogisticRegression(max_iter=1500,random_state=SEED)),
        'DecisionTree':pipeline(DecisionTreeClassifier(max_depth=8,min_samples_leaf=20,random_state=SEED)),
        'RandomForest':pipeline(RandomForestClassifier(n_estimators=100,min_samples_leaf=3,n_jobs=2,random_state=SEED)),
    }
    def cm(y, proba):
        p = proba>=0.5
        return {'Accuracy':accuracy_score(y,p),'BalancedAccuracy':balanced_accuracy_score(y,p),
                'Precision':precision_score(y,p,zero_division=0),'Recall':recall_score(y,p,zero_division=0),
                'F1':f1_score(y,p,zero_division=0),'ROC_AUC':roc_auc_score(y,proba),
                'AveragePrecision':average_precision_score(y,proba)}
    cv,ct = [],[]
    for name,m in class_models.items():
        m.fit(train[FEATURES],train.cnt>=high_threshold)
        cv.append({'model':name,**cm(val.cnt>=high_threshold,m.predict_proba(val[FEATURES])[:,1])})
    class_selected = max(cv,key=lambda r:r['F1'])['model']
    class_predictions = []
    for name,m in class_models.items():
        m.fit(trainval[FEATURES],trainval.cnt>=high_threshold)
        prob = m.predict_proba(test[FEATURES])[:,1]
        y = test.cnt>=high_threshold
        ct.append({'model':name,'selected_on_validation':name==class_selected,**cm(y,prob)})
        class_predictions.append(pd.DataFrame({'timestamp':test.timestamp,'model':name,'actual_high':y,
                                               'probability':prob,'prediction_high':prob>=0.5}))
        if name==class_selected:
            csv_save(pd.DataFrame(confusion_matrix(y,prob>=0.5),columns=['pred_normal','pred_high'])
                     .assign(actual=['normal','high']), 'classification_confusion')
    # 多数类基线：概率为训练期先验，阈值固定 0.5。
    ct.append({'model':'PriorBaseline','selected_on_validation':False,
               **cm(test.cnt>=high_threshold,np.full(len(test),float((train.cnt>=high_threshold).mean())))})
    csv_save(pd.DataFrame(cv),'classification_validation')
    csv_save(pd.DataFrame(ct),'classification_test')
    csv_save(pd.concat(class_predictions,ignore_index=True),'classification_predictions')
    selection = {'regression_selected':selected,'classification_selected':class_selected,
        'high_demand_threshold':high_threshold,'threshold_fit_period':'train only',
        'probability_cutoff':0.5,'regression_selection':'validation MAE minimum',
        'classification_selection':'validation F1 maximum',
        'prediction_protocol':'Rolling one-step: predict hour t using calendar(t) and observations <= t-1. Model is fixed after train+validation refit.',
        'features':FEATURES,'interval':interval}
    save_json(RESULTS / 'model_selection.json',selection)
    with sqlite3.connect(DB) as con:
        con.execute('PRAGMA foreign_keys=ON')
        con.execute('DELETE FROM fact_predictions');con.execute('DELETE FROM dim_model')
        con.executemany('INSERT INTO dim_model VALUES (?,?,?)',[(n,'regression',int(n==selected)) for n in all_preds.model.unique()])
        con.executemany('INSERT INTO fact_predictions VALUES (?,?,?,?,?,?)',
            [(r.model,int(r.instant),'test',float(r.actual),float(r.prediction),float(r.absolute_error)) for r in all_preds.itertuples()])
        assert not con.execute('PRAGMA foreign_key_check').fetchall()
        csv_save(pd.read_sql_query((ROOT/'sql/07_model_olap.sql').read_text(),con),'07_model_olap')
    fig,ax=plt.subplots(figsize=(10,4))
    ax.barh(test_results.model,test_results.MAE,color=['#da8243' if s else '#187f8c' for s in test_results.selected_on_validation])
    ax.set(xlabel='Test MAE (rentals/hour)',title='Regression comparison; orange = validation-selected')
    ax.invert_yaxis();save_fig(fig,'regression_metrics')
    p = all_preds[all_preds.model==selected].head(168)
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(p.timestamp,p.actual,label='Observed',color='#1a3440',linewidth=1.2)
    ax.plot(p.timestamp,p.prediction,label=selected,color='#db8242',linewidth=1)
    ax.fill_between(p.timestamp,low[:len(p)],high[:len(p)],alpha=.18,label='HGB 5%-95% interval')
    ax.set(ylabel='Rentals/hour',title='First test week: rolling one-step predictions');ax.legend(fontsize=8)
    save_fig(fig,'test_predictions')
    fig,ax=plt.subplots(figsize=(8,4))
    imp.head(10).iloc[::-1].plot.barh(x='feature',y='MAE_increase',ax=ax,legend=False,color='#187f8c')
    ax.set(title=f'Validation permutation importance: {best_ml}',xlabel='Increase in MAE after permutation')
    save_fig(fig,'feature_importance')
    return {'validation':val_results,'test':test_results,'classification':pd.DataFrame(ct),'selection':selection}


def clustering(df):
    # 只分析 24 小时全部存在的日期；不把缺失小时补零后伪造日形状。
    daily = df.pivot(index='dteday',columns='hr',values='cnt').dropna()
    shape = daily.div(daily.sum(axis=1),axis=0)
    scores=[]
    for k in range(2,7):
        labels=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit_predict(shape)
        scores.append({'method':'KMeans','k':k,'silhouette':silhouette_score(shape,labels)})
    k = max(scores,key=lambda r:r['silhouette'])['k']
    labels=KMeans(n_clusters=k,n_init=20,random_state=SEED).fit_predict(shape)
    ward=AgglomerativeClustering(n_clusters=k,linkage='ward').fit_predict(shape)
    scores.append({'method':'AgglomerativeWard','k':k,'silhouette':silhouette_score(shape,ward)})
    assignments=pd.DataFrame({'dteday':daily.index,'kmeans_cluster':labels,'ward_cluster':ward,
                              'daily_rentals':daily.sum(axis=1).to_numpy()})
    metadata=df.groupby('dteday').agg(daytype=('daytype','first'),season=('season','first')).reset_index()
    assignments=assignments.merge(metadata,on='dteday',validate='one_to_one')
    profiles=shape.assign(cluster=labels).groupby('cluster').mean()
    summary=assignments.groupby('kmeans_cluster').agg(days=('dteday','size'),mean_daily_rentals=('daily_rentals','mean'),
        workday_fraction=('daytype',lambda s:float((s==1).mean()))).reset_index()
    csv_save(pd.DataFrame(scores),'cluster_selection');csv_save(assignments,'cluster_assignments')
    csv_save(summary,'cluster_summary');csv_save(profiles.reset_index(),'cluster_profiles')
    fig,ax=plt.subplots(figsize=(10,4))
    for i,row in profiles.iterrows():ax.plot(range(24),row.to_numpy()*100,marker='o',label=f'Cluster {i}')
    ax.set(xlabel='Hour',ylabel='Share of daily rentals (%)',title=f'Daily demand shapes (complete days only), K={k}')
    ax.legend();save_fig(fig,'cluster_profiles')
    return summary


def apriori(transactions, min_support=0.03, max_len=3):
    """频繁项集 Apriori：逐层连接 + 所有 k-1 子集必须频繁的剪枝。"""
    n=len(transactions)
    counts=Counter(item for t in transactions for item in t)
    level={frozenset([item]):c/n for item,c in counts.items() if c/n>=min_support}
    frequent=dict(level)
    for k in range(2,max_len+1):
        previous=set(level)
        candidates={a|b for a,b in itertools.combinations(previous,2) if len(a|b)==k}
        candidates={c for c in candidates if all(frozenset(s) in previous for s in itertools.combinations(c,k-1))}
        level={c:sum(c.issubset(t) for t in transactions)/n for c in candidates}
        level={c:s for c,s in level.items() if s>=min_support}
        frequent.update(level)
        if not level:break
    return frequent


def association_rules(df, threshold):
    # 描述性关联：允许使用当时天气；绝不能把这些规则称为提前预测/因果规律。
    def transactions(frame):
        return [frozenset([f'season={r.season}',f'daytype={r.daytype}',f'period={hour_period(r.hr)}',
            f'weather={r.weathersit}', 'demand=high' if r.cnt>=threshold else 'demand=normal'])
            for r in frame.itertuples()]
    train=transactions(df[df.timestamp<TRAIN_END])
    test=transactions(df[df.timestamp>=VALID_END])
    frequent=apriori(train)
    high=frozenset(['demand=high'])
    base_train=sum(high.issubset(t) for t in train)/len(train)
    base_test=sum(high.issubset(t) for t in test)/len(test)
    rows=[]
    for itemset,support in frequent.items():
        if not high.issubset(itemset) or len(itemset)<2:continue
        lhs=itemset-high
        confidence=support/frequent[lhs]
        lift=confidence/base_train
        if confidence<0.5 or lift<=1:continue
        antecedent_count=sum(lhs.issubset(t) for t in test)
        joint_count=sum(itemset.issubset(t) for t in test)
        test_conf=joint_count/antecedent_count if antecedent_count else np.nan
        rows.append({'antecedent':' & '.join(sorted(lhs)),'consequent':'demand=high',
            'train_support':support,'train_confidence':confidence,'train_lift':lift,
            'test_antecedent_count':antecedent_count,'test_support':joint_count/len(test),
            'test_confidence':test_conf,'test_lift':test_conf/base_test})
    columns=['antecedent','consequent','train_support','train_confidence','train_lift',
             'test_antecedent_count','test_support','test_confidence','test_lift']
    rules=pd.DataFrame(rows,columns=columns).sort_values(['train_lift','train_support'],ascending=False)
    csv_save(rules,'association_rules')
    csv_save(pd.DataFrame([{'items':' & '.join(sorted(k)),'support':v,'length':len(k)} for k,v in frequent.items()]),'frequent_itemsets')
    save_json(RESULTS/'association_config.json',{'min_support':.03,'min_confidence':.5,'max_itemset_length':3,
        'train_transactions':len(train),'test_transactions':len(test),'train_high_base_rate':base_train,
        'test_high_base_rate':base_test,'threshold':threshold,'rule_count':len(rules)})
    return rules


def run_all():
    start=time.perf_counter()
    with threadpool_limits(limits=2):
        print('[1/8] 收集与校验数据',flush=True);collect()
        print('[2/8] 预处理与质量审计',flush=True);df,quality=clean_data()
        print('[3/8] 建立星型数据仓库',flush=True);build_warehouse(df)
        print('[4/8] 六类 OLAP 操作',flush=True);run_olap()
        source=from_warehouse()
        print('[5/8] 特征工程、时间验证、回归与分类',flush=True)
        data=make_features(source);results=supervised(data)
        print('[6/8] 每日用车模式聚类',flush=True);clustering(source)
        print('[7/8] Apriori 关联规则与留出期复核',flush=True)
        association_rules(source,results['selection']['high_demand_threshold'])
        print('[8/8] 生成报告',flush=True)
        environment={'python':platform.python_version(),'platform':platform.platform(),
            'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','scipy','matplotlib','nbformat','nbclient','markdown']},
            'elapsed_seconds':time.perf_counter()-start,'seed':SEED}
        save_json(RESULTS/'environment.json',environment)
        from reporting import write_report
        write_report()
    print(f'完成，用时 {time.perf_counter()-start:.1f} 秒。报告：{ROOT / "reports/report.html"}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-only',action='store_true')
    args=parser.parse_args()
    if args.report_only:
        from reporting import write_report
        write_report()
    else:run_all()
