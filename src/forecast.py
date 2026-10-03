import json
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

# project ka root folder, jahan se data aur models uthayenge
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "raw" / "aqi_hourly_v2.csv"
MODELS = ROOT / "models"

FEATURE_COLS = json.load(open(MODELS / "feature_cols.json"))
WEATHER = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m",
           "wind_direction_10m", "surface_pressure", "precipitation"]

_models = {}


def _load(name):
    # model ek hi baar load hoga
    if name not in _models:
        m = XGBRegressor()
        m.load_model(str(MODELS / name))
        _models[name] = m
    return _models[name]


def load_data(path=DATA_PATH):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    # chhote gaps (6 ghante tak) bhare
    df["AQI"] = df["AQI"].interpolate(limit=6, limit_area="inside")
    return df


def build_features(df):
    # bilkul wahi features jo notebook 04 mein the
    aqi = df["AQI"]
    feat = pd.DataFrame(index=df.index)
    feat["aqi_now"] = aqi
    for l in [1, 2, 3, 6, 12, 24, 48, 72]:
        feat[f"aqi_lag_{l}"] = aqi.shift(l)
    feat["aqi_roll_24"] = aqi.rolling(24, min_periods=18).mean()
    feat["aqi_roll_72"] = aqi.rolling(72, min_periods=54).mean()
    for c in WEATHER:
        feat[c] = df[c]
    feat["hour"] = df.index.hour
    feat["month"] = df.index.month
    feat["dayofweek"] = df.index.dayofweek
    return feat[FEATURE_COLS]


def grap_stage(aqi):
    # pdf ke thresholds: 201-300, 301-400, 401-450, 450 se upar
    if aqi > 450: return "Stage IV"
    if aqi > 400: return "Stage III"
    if aqi > 300: return "Stage II"
    if aqi > 200: return "Stage I"
    return None


def predict(feat, ts):
    # ts = wo ghanta jahan se aage ke 3 din dekhne hain
    ts = pd.Timestamp(ts)
    if ts not in feat.index:
        return None
    row = feat.loc[[ts]]
    if row.isna().any(axis=1).iloc[0]:
        return None   # is ghante ka data adhoora hai

    out = {}
    for d in ["day1", "day2", "day3"]:
        med = float(np.clip(_load(f"xgb_{d}.json").predict(row)[0], 0, 500))
        q80 = float(np.clip(_load(f"xgb_{d}_q80.json").predict(row)[0], 0, 500))
        out[d] = {"expected": round(med), "worst_case": round(q80),
                  "stage_expected": grap_stage(med), "stage_worst": grap_stage(q80)}
    return out


def actuals(df, ts):
    # replay demo ke liye: asli 24 ghante ka average (agle 3 din)
    ts = pd.Timestamp(ts)
    daily = df["AQI"].rolling(24, min_periods=18).mean()
    res = {}
    for d, h in [("day1", 24), ("day2", 48), ("day3", 72)]:
        v = daily.shift(-h).loc[ts]
        res[d] = None if pd.isna(v) else round(float(v))
    return res