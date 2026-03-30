"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Anomaly Detection Engine
 Models: Isolation Forest · LSTM Autoencoder · Z-Score
=============================================================
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import joblib
import os
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
ANOMALY_FEATURES = [
    "co2", "co2_per_capita", "co2_yoy_change",
    "temperature_change_from_ghg", "fossil_ratio",
    "primary_energy_consumption",
]

SEVERITY_THRESHOLDS = {
    "critical": -0.35,
    "high":     -0.20,
    "medium":   -0.10,
}


# ─────────────────────────────────────────────
#  ISOLATION FOREST DETECTOR
# ─────────────────────────────────────────────
class IsolationForestDetector:
    """
    Detects global/cross-country anomalies using Isolation Forest.
    Contamination tuned for environmental data (~5 % anomaly rate).
    """

    def __init__(self, contamination: float = 0.05, n_estimators: int = 200):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.feature_cols: list[str] = []

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        self.feature_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
        X = df[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        print(f"[IForest] Trained on {len(X):,} samples, {len(self.feature_cols)} features")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.copy()
        X = result[self.feature_cols].fillna(0).values
        X_scaled = self.scaler.transform(X)

        result["if_score"] = self.model.score_samples(X_scaled)
        result["if_flag"] = self.model.predict(X_scaled)   # -1 = anomaly
        result["is_anomaly"] = result["if_flag"] == -1

        # Severity levels
        result["severity"] = "normal"
        result.loc[result["if_score"] < SEVERITY_THRESHOLDS["medium"],   "severity"] = "medium"
        result.loc[result["if_score"] < SEVERITY_THRESHOLDS["high"],     "severity"] = "high"
        result.loc[result["if_score"] < SEVERITY_THRESHOLDS["critical"],  "severity"] = "critical"

        return result

    def save(self, path: str = "models/isolation_forest.pkl") -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler,
                     "features": self.feature_cols}, path)
        print(f"[IForest] Saved → {path}")

    @classmethod
    def load(cls, path: str = "models/isolation_forest.pkl") -> "IsolationForestDetector":
        obj = cls()
        data = joblib.load(path)
        obj.model, obj.scaler, obj.feature_cols = (
            data["model"], data["scaler"], data["features"]
        )
        return obj


# ─────────────────────────────────────────────
#  Z-SCORE COUNTRY-LEVEL DETECTOR
# ─────────────────────────────────────────────
class ZScoreDetector:
    """
    Detects anomalies within a single country's time-series
    using rolling Z-score on CO2 emissions.
    """

    def __init__(self, window: int = 5, threshold: float = 2.5):
        self.window = window
        self.threshold = threshold

    def detect(self, df: pd.DataFrame, col: str = "co2") -> pd.DataFrame:
        result = df.copy().sort_values("year")

        rolling_mean = result[col].rolling(self.window, min_periods=2).mean()
        rolling_std  = result[col].rolling(self.window, min_periods=2).std().replace(0, 1e-9)

        result["zscore"] = (result[col] - rolling_mean) / rolling_std
        result["zscore_anomaly"] = result["zscore"].abs() > self.threshold

        result["zscore_severity"] = "normal"
        result.loc[result["zscore"].abs() > self.threshold,       "zscore_severity"] = "medium"
        result.loc[result["zscore"].abs() > self.threshold * 1.5, "zscore_severity"] = "high"
        result.loc[result["zscore"].abs() > self.threshold * 2.0, "zscore_severity"] = "critical"

        return result


# ─────────────────────────────────────────────
#  LSTM AUTOENCODER (TensorFlow optional)
# ─────────────────────────────────────────────
class LSTMAutoencoderDetector:
    """
    Sequence-based anomaly detection.
    Requires tensorflow ≥ 2.x.
    Falls back gracefully if TF is unavailable.
    """

    def __init__(self, seq_len: int = 5, latent_dim: int = 16, threshold_pct: float = 95):
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.threshold_pct = threshold_pct
        self.model = None
        self.threshold = None
        self.scaler = StandardScaler()

    def _build_model(self, n_features: int):
        try:
            from tensorflow import keras
            inp = keras.Input(shape=(self.seq_len, n_features))
            enc = keras.layers.LSTM(self.latent_dim, return_sequences=False)(inp)
            rep = keras.layers.RepeatVector(self.seq_len)(enc)
            dec = keras.layers.LSTM(self.latent_dim, return_sequences=True)(rep)
            out = keras.layers.TimeDistributed(keras.layers.Dense(n_features))(dec)
            self.model = keras.Model(inp, out)
            self.model.compile(optimizer="adam", loss="mse")
            return True
        except ImportError:
            print("[LSTM] TensorFlow not installed – skipping LSTM module")
            return False

    def _make_sequences(self, X: np.ndarray) -> np.ndarray:
        seqs = []
        for i in range(len(X) - self.seq_len + 1):
            seqs.append(X[i: i + self.seq_len])
        return np.array(seqs)

    def fit(self, df: pd.DataFrame, epochs: int = 30) -> "LSTMAutoencoderDetector":
        feat_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
        X = self.scaler.fit_transform(df[feat_cols].fillna(0).values)
        sequences = self._make_sequences(X)

        if not self._build_model(X.shape[1]):
            return self

        self.model.fit(
            sequences, sequences,
            epochs=epochs, batch_size=32,
            validation_split=0.1, verbose=0,
        )

        # Compute reconstruction errors on training data
        recon = self.model.predict(sequences)
        errors = np.mean(np.abs(sequences - recon), axis=(1, 2))
        self.threshold = np.percentile(errors, self.threshold_pct)
        print(f"[LSTM] Trained · threshold={self.threshold:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            df = df.copy()
            df["lstm_anomaly"] = False
            df["lstm_error"] = 0.0
            return df

        feat_cols = [c for c in ANOMALY_FEATURES if c in df.columns]
        X = self.scaler.transform(df[feat_cols].fillna(0).values)
        sequences = self._make_sequences(X)
        recon = self.model.predict(sequences)
        errors = np.mean(np.abs(sequences - recon), axis=(1, 2))

        result = df.copy()
        # Pad errors (first seq_len-1 rows have no prediction)
        pad = np.zeros(self.seq_len - 1)
        all_errors = np.concatenate([pad, errors])
        result["lstm_error"] = all_errors
        result["lstm_anomaly"] = result["lstm_error"] > (self.threshold or 1e9)
        return result


# ─────────────────────────────────────────────
#  COMBINED ANOMALY ENGINE
# ─────────────────────────────────────────────
class AnomalyDetectionEngine:
    """
    Orchestrates all detectors and produces a unified anomaly report.
    """

    def __init__(self):
        self.if_detector   = IsolationForestDetector()
        self.zs_detector   = ZScoreDetector()
        self.lstm_detector = LSTMAutoencoderDetector()
        self._trained = False

    def train(self, df: pd.DataFrame) -> "AnomalyDetectionEngine":
        print("\n[Engine] Training anomaly detectors …")
        self.if_detector.fit(df)
        self._trained = True
        print("[Engine] All detectors ready ✓\n")
        return self

    def detect_global(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run Isolation Forest across all countries."""
        if not self._trained:
            raise RuntimeError("Call train() first")
        return self.if_detector.predict(df)

    def detect_country(self, df: pd.DataFrame, country: str) -> pd.DataFrame:
        """Run Z-Score on a single country's time-series."""
        sub = df[df["country"] == country].sort_values("year").copy()
        if "co2" not in sub.columns or len(sub) < 3:
            return sub
        return self.zs_detector.detect(sub)

    def get_anomaly_summary(self, df_flagged: pd.DataFrame) -> pd.DataFrame:
        """Return only the flagged rows sorted by severity."""
        order = {"critical": 0, "high": 1, "medium": 2, "normal": 3}
        anomalies = df_flagged[df_flagged["is_anomaly"]].copy()
        anomalies["_rank"] = anomalies["severity"].map(order)
        return (
            anomalies.sort_values("_rank")
            .drop(columns="_rank")
            .reset_index(drop=True)
        )

    def get_alert_count(self, df_flagged: pd.DataFrame) -> dict:
        counts = df_flagged[df_flagged["is_anomaly"]]["severity"].value_counts().to_dict()
        return {
            "critical": counts.get("critical", 0),
            "high":     counts.get("high",     0),
            "medium":   counts.get("medium",   0),
            "total":    df_flagged["is_anomaly"].sum(),
        }


# ── QUICK TEST ────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from pipeline.data_pipeline import SustainabilityDataPipeline

    pipe = SustainabilityDataPipeline("../data/raw/owid-co2-data.csv")
    df = pipe.run()

    engine = AnomalyDetectionEngine()
    engine.train(df)

    flagged = engine.detect_global(df)
    summary = engine.get_anomaly_summary(flagged)
    counts  = engine.get_alert_count(flagged)

    print(f"\nAnomaly counts: {counts}")
    print(summary[["country", "year", "co2", "severity", "if_score"]].head(15))
