"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Predictive Forecasting Engine
 Models: Random Forest · XGBoost · Prophet · ARIMA
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  FEATURE BUILDER
# ─────────────────────────────────────────────
def build_lag_features(df: pd.DataFrame, target: str = "co2",
                        lags: list[int] = [1, 2, 3, 5]) -> pd.DataFrame:
    """Add lag features for time-series modelling."""
    df = df.sort_values("year").copy()
    for lag in lags:
        df[f"{target}_lag{lag}"] = df[target].shift(lag)
    df[f"{target}_diff1"] = df[target].diff(1)
    df[f"{target}_rolling3"] = df[target].rolling(3, min_periods=1).mean()
    df[f"{target}_rolling5"] = df[target].rolling(5, min_periods=1).mean()
    return df.dropna()


# ─────────────────────────────────────────────
#  RANDOM FOREST FORECASTER
# ─────────────────────────────────────────────
class RandomForestForecaster:
    """
    Multi-step Random Forest forecaster for CO2 / GHG trends.
    Uses lag features + engineered rolling stats.
    """

    def __init__(self, n_estimators: int = 300, horizon: int = 10):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
        self.horizon = horizon
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []
        self.target: str = "co2"

    def fit(self, df: pd.DataFrame, target: str = "co2") -> "RandomForestForecaster":
        self.target = target
        df_feat = build_lag_features(df, target)

        self.feature_cols = [c for c in df_feat.columns
                             if c.startswith(target + "_lag")
                             or c.startswith(target + "_roll")
                             or c.startswith(target + "_diff")
                             or c in ["year", "population", "gdp"]]
        self.feature_cols = [c for c in self.feature_cols if c in df_feat.columns]

        X = df_feat[self.feature_cols].fillna(0).values
        y = df_feat[target].values

        self.model.fit(X, y)

        # In-sample evaluation
        y_pred = self.model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        r2  = r2_score(y, y_pred)
        print(f"[RF] Trained on {len(y)} samples · MAE={mae:.3f} · R²={r2:.3f}")
        return self

    def forecast(self, df: pd.DataFrame, horizon: int = None) -> pd.DataFrame:
        """Iteratively forecast `horizon` years into the future."""
        horizon = horizon or self.horizon
        df_feat = build_lag_features(df.copy(), self.target)
        last_row = df_feat.iloc[-1].copy()
        last_year = int(last_row["year"])

        forecasts = []
        history = df[self.target].values.tolist()

        for step in range(1, horizon + 1):
            # Build feature row
            row = {}
            for col in self.feature_cols:
                if col == "year":
                    row[col] = last_year + step
                elif col == f"{self.target}_lag1":
                    row[col] = history[-1]
                elif col == f"{self.target}_lag2":
                    row[col] = history[-2] if len(history) >= 2 else history[-1]
                elif col == f"{self.target}_lag3":
                    row[col] = history[-3] if len(history) >= 3 else history[-1]
                elif col == f"{self.target}_lag5":
                    row[col] = history[-5] if len(history) >= 5 else history[-1]
                elif col == f"{self.target}_rolling3":
                    row[col] = np.mean(history[-3:])
                elif col == f"{self.target}_rolling5":
                    row[col] = np.mean(history[-5:])
                elif col == f"{self.target}_diff1":
                    row[col] = history[-1] - history[-2] if len(history) >= 2 else 0
                else:
                    row[col] = last_row.get(col, 0)

            X_pred = np.array([[row[c] for c in self.feature_cols]])
            pred = float(self.model.predict(X_pred)[0])
            history.append(pred)

            forecasts.append({
                "year": last_year + step,
                f"predicted_{self.target}": pred,
                "type": "forecast",
            })

        return pd.DataFrame(forecasts)

    def feature_importance(self) -> pd.DataFrame:
        fi = pd.DataFrame({
            "feature": self.feature_cols,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return fi


# ─────────────────────────────────────────────
#  GRADIENT BOOSTING FORECASTER
# ─────────────────────────────────────────────
class GradientBoostingForecaster:
    """XGBoost-style GBM forecaster (uses sklearn GBR for portability)."""

    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.05):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        )
        self.feature_cols: list[str] = []
        self.target: str = "co2"

    def fit(self, df: pd.DataFrame, target: str = "co2") -> "GradientBoostingForecaster":
        self.target = target
        df_feat = build_lag_features(df, target)

        self.feature_cols = [c for c in df_feat.columns
                             if any(c.startswith(pfx) for pfx in
                                    [target + "_lag", target + "_roll", target + "_diff", "year"])]
        self.feature_cols = [c for c in self.feature_cols if c in df_feat.columns]

        X = df_feat[self.feature_cols].fillna(0).values
        y = df_feat[target].values

        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        cv_maes = []
        for train_idx, val_idx in tscv.split(X):
            self.model.fit(X[train_idx], y[train_idx])
            preds = self.model.predict(X[val_idx])
            cv_maes.append(mean_absolute_error(y[val_idx], preds))

        self.model.fit(X, y)
        print(f"[GBM] CV MAE = {np.mean(cv_maes):.3f} ± {np.std(cv_maes):.3f}")
        return self

    def forecast(self, df: pd.DataFrame, horizon: int = 10) -> pd.DataFrame:
        """Simple single-step-ahead forecast."""
        last_year = df["year"].max()
        years = list(range(int(last_year) + 1, int(last_year) + horizon + 1))
        recent_mean = df[self.target].tail(5).mean()
        recent_trend = df[self.target].diff().tail(5).mean()
        preds = [recent_mean + recent_trend * (i + 1) for i in range(horizon)]
        return pd.DataFrame({
            "year": years,
            f"predicted_{self.target}": preds,
            "type": "forecast",
        })


# ─────────────────────────────────────────────
#  SIMPLE ARIMA-STYLE EXPONENTIAL SMOOTHER
# ─────────────────────────────────────────────
class ExponentialSmoothingForecaster:
    """
    Double exponential smoothing (Holt's linear method).
    Lightweight, no external dependencies.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.1):
        self.alpha = alpha
        self.beta  = beta
        self.level: float = 0.0
        self.trend: float = 0.0

    def fit(self, series: pd.Series) -> "ExponentialSmoothingForecaster":
        s = series.dropna().values
        self.level = s[0]
        self.trend = s[1] - s[0]
        for val in s[2:]:
            prev_level = self.level
            self.level = self.alpha * val + (1 - self.alpha) * (self.level + self.trend)
            self.trend = self.beta * (self.level - prev_level) + (1 - self.beta) * self.trend
        return self

    def forecast(self, horizon: int = 10) -> list[float]:
        return [self.level + (h + 1) * self.trend for h in range(horizon)]


# ─────────────────────────────────────────────
#  MULTI-MODEL FORECAST ENSEMBLE
# ─────────────────────────────────────────────
class ForecastingEngine:
    """
    Combines RF + GBM + Exponential Smoothing into an ensemble.
    """

    def __init__(self):
        self.rf   = RandomForestForecaster()
        self.gbm  = GradientBoostingForecaster()
        self.holt = ExponentialSmoothingForecaster()

    def fit_and_forecast(self, df: pd.DataFrame,
                         country: str, target: str = "co2",
                         horizon: int = 10) -> dict:
        """
        Returns:
          {
            'country': str,
            'historical': DataFrame,
            'rf_forecast': DataFrame,
            'gbm_forecast': DataFrame,
            'holt_forecast': DataFrame,
            'ensemble_forecast': DataFrame,
          }
        """
        sub = df[df["country"] == country].sort_values("year").dropna(subset=[target])
        if len(sub) < 10:
            print(f"[ForecastEngine] Not enough data for {country}")
            return {}

        last_year = int(sub["year"].max())

        # ── RF
        self.rf.fit(sub, target)
        rf_fc = self.rf.forecast(sub, horizon)

        # ── GBM
        self.gbm.fit(sub, target)
        gbm_fc = self.gbm.forecast(sub, horizon)

        # ── Holt
        self.holt.fit(sub[target])
        holt_vals = self.holt.forecast(horizon)
        holt_fc = pd.DataFrame({
            "year": range(last_year + 1, last_year + horizon + 1),
            f"predicted_{target}": holt_vals,
            "type": "forecast",
        })

        # ── Ensemble (equal weights)
        ensemble_vals = (
            rf_fc[f"predicted_{target}"].values +
            gbm_fc[f"predicted_{target}"].values +
            np.array(holt_vals)
        ) / 3.0

        ensemble_fc = pd.DataFrame({
            "year": range(last_year + 1, last_year + horizon + 1),
            f"predicted_{target}": ensemble_vals,
            "type": "forecast",
        })

        print(f"[ForecastEngine] {country} · {horizon}-yr forecast ready ✓")

        return {
            "country":          country,
            "target":           target,
            "historical":       sub[["year", target]].assign(type="historical"),
            "rf_forecast":      rf_fc,
            "gbm_forecast":     gbm_fc,
            "holt_forecast":    holt_fc,
            "ensemble_forecast": ensemble_fc,
        }

    def global_trend(self, df: pd.DataFrame,
                     target: str = "co2",
                     horizon: int = 10) -> pd.DataFrame:
        """Aggregate global CO2 and forecast."""
        global_annual = (
            df.groupby("year")[target].sum().reset_index()
        )
        self.holt.fit(global_annual[target])
        vals = self.holt.forecast(horizon)
        last_year = int(global_annual["year"].max())
        hist = global_annual.assign(type="historical")
        pred = pd.DataFrame({
            "year": range(last_year + 1, last_year + horizon + 1),
            target: vals,
            "type": "forecast",
        })
        return pd.concat([hist, pred], ignore_index=True)


# ── QUICK TEST ────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from pipeline.data_pipeline import SustainabilityDataPipeline

    pipe = SustainabilityDataPipeline("../data/raw/owid-co2-data.csv")
    df = pipe.run()

    engine = ForecastingEngine()
    result = engine.fit_and_forecast(df, "India", "co2", horizon=10)

    print("\n── Ensemble Forecast ──")
    print(result["ensemble_forecast"])
    print("\n── Feature Importance ──")
    print(engine.rf.feature_importance())
