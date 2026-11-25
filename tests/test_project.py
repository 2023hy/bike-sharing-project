import sys
from pathlib import Path
import sqlite3
import unittest
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import project as p


class TemporalTests(unittest.TestCase):
    def synthetic(self):
        t=pd.date_range('2011-01-01',periods=220,freq='h')
        df=pd.DataFrame({'timestamp':t,'instant':np.arange(len(t)), 'cnt':np.arange(len(t)),
            'season':1,'daytype':1,'temp':.5,'hum':.5,'windspeed':.1,'weathersit':1})
        return df.drop(index=180)

    def test_missing_hour_is_not_previous_row(self):
        f=p.make_features(self.synthetic()).set_index('timestamp')
        t=pd.Timestamp('2011-01-01')+pd.Timedelta(hours=181)
        self.assertTrue(np.isnan(f.loc[t,'lag_1']))
        self.assertEqual(f.loc[t,'lag_24'],157)
        self.assertEqual(f.loc[t,'last_observed'],179)

    def test_future_and_current_target_do_not_change_features(self):
        df=self.synthetic()
        cut=pd.Timestamp('2011-01-09')
        before=p.make_features(df).set_index('timestamp')
        df.loc[df.timestamp>=cut,['cnt','temp','hum','weathersit']]=99999
        after=p.make_features(df).set_index('timestamp')
        pd.testing.assert_frame_equal(before.loc[:cut,p.FEATURES],after.loc[:cut,p.FEATURES])

    def test_target_components_excluded(self):
        self.assertFalse(set(p.FEATURES)&{'cnt','casual','registered','instant'})


class AprioriTests(unittest.TestCase):
    def test_known_support_and_pruning(self):
        t=[frozenset('AB'),frozenset('AB'),frozenset('A'),frozenset('BC')]
        f=p.apriori(t,min_support=.5,max_len=3)
        self.assertEqual(f[frozenset('A')],.75)
        self.assertEqual(f[frozenset('B')],.75)
        self.assertEqual(f[frozenset('AB')],.5)
        self.assertNotIn(frozenset('C'),f)
        self.assertNotIn(frozenset('ABC'),f)


class WarehouseIntegrationTests(unittest.TestCase):
    def test_conservation_and_prediction_integrity(self):
        self.assertTrue(p.DB.exists(),'先运行完整项目，再进行集成验收')
        raw=pd.read_csv(p.RAW/'hour.csv')
        with sqlite3.connect(p.DB) as con:
            self.assertEqual(con.execute('PRAGMA integrity_check').fetchone()[0],'ok')
            self.assertEqual(con.execute('PRAGMA foreign_key_check').fetchall(),[])
            n,total=con.execute('SELECT COUNT(*),SUM(cnt) FROM fact_rentals').fetchone()
            self.assertEqual(n,len(raw));self.assertEqual(total,int(raw.cnt.sum()))
            cube=pd.read_sql_query((p.ROOT/'sql/06_cube.sql').read_text(),con)
            for a in [0,1]:
                for b in [0,1]:
                    self.assertEqual(int(cube[(cube.season_grouped==a)&(cube.daytype_grouped==b)].rentals.sum()),total)
            ntest=int((pd.to_datetime(raw.dteday)>=p.VALID_END).sum())
            counts=con.execute('SELECT model_key,COUNT(*) FROM fact_predictions GROUP BY model_key').fetchall()
            self.assertEqual(len(counts),6)
            self.assertTrue(all(n==ntest for _,n in counts))
            mismatch=con.execute('SELECT COUNT(*) FROM fact_predictions p JOIN fact_rentals f USING(observation_id) WHERE p.actual != f.cnt').fetchone()[0]
            self.assertEqual(mismatch,0)


if __name__=='__main__':unittest.main()
