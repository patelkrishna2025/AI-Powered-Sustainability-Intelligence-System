"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Data Pipeline
 Author: Sustainability Analytics Team
=============================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
FEATURE_COLS = [
    "co2", "co2_per_capita", "co2_per_gdp",
    "primary_energy_consumption", "energy_per_capita",
    "methane", "nitrous_oxide", "total_ghg",
    "temperature_change_from_ghg",
    "coal_co2", "oil_co2", "gas_co2",
    "land_use_change_co2", "share_global_co2",
]

KEY_COUNTRIES = [
    "United States", "China", "India", "Germany",
    "United Kingdom", "Brazil", "Russia", "Japan",
    "Canada", "Australia", "France", "South Africa",
]


# ─────────────────────────────────────────────
#  PIPELINE CLASS
# ─────────────────────────────────────────────
class SustainabilityDataPipeline:
    """
    End-to-end data pipeline that:
      1. Loads raw OWID CO2 data
      2. Filters, cleans, and imputes
      3. Engineers features
      4. Normalises for ML models
      5. Exports processed datasets
    """

    def __init__(self, raw_path: str = "data/raw/owid-co2-data.csv"):
        self.raw_path = raw_path
        self.df_raw: pd.DataFrame = pd.DataFrame()
        self.df_clean: pd.DataFrame = pd.DataFrame()
        self.df_processed: pd.DataFrame = pd.DataFrame()
        self.scaler = MinMaxScaler()
        print("[Pipeline] Initialised ✓")

    # ── 1. LOAD ──────────────────────────────
    def load(self) -> "SustainabilityDataPipeline":
        self.df_raw = pd.read_csv(self.raw_path, low_memory=False)
        print(f"[Pipeline] Loaded {len(self.df_raw):,} rows × {self.df_raw.shape[1]} cols")
        return self

    # ── 2. CLEAN ─────────────────────────────
    def clean(self, start_year: int = 1990) -> "SustainabilityDataPipeline":
        df = self.df_raw.copy()

        # Keep only sovereign nations with ISO codes
        df = df[df["iso_code"].notna() & (df["iso_code"].str.len() == 3)]

        # Year filter
        df = df[df["year"] >= start_year]

        # Keep feature columns that exist
        existing = [c for c in FEATURE_COLS if c in df.columns]
        keep = ["country", "iso_code", "year", "population", "gdp"] + existing
        keep = [c for c in keep if c in df.columns]
        df = df[keep]

        # Drop rows where ALL feature cols are NaN
        df = df.dropna(subset=existing, how="all")

        # Forward-fill within each country group
        df = df.sort_values(["country", "year"])
        df[existing] = df.groupby("country")[existing].transform(
            lambda x: x.ffill().bfill()
        )

        # Fill remaining NaNs with 0
        df[existing] = df[existing].fillna(0)

        self.df_clean = df.reset_index(drop=True)
        print(f"[Pipeline] After cleaning: {len(self.df_clean):,} rows")
        return self

    # ── 3. FEATURE ENGINEERING ───────────────
    def engineer_features(self) -> "SustainabilityDataPipeline":
        df = self.df_clean.copy()
        existing = [c for c in FEATURE_COLS if c in df.columns]

        # YoY change for CO2
        if "co2" in df.columns:
            df["co2_yoy_change"] = df.groupby("country")["co2"].pct_change() * 100
            df["co2_yoy_change"] = df["co2_yoy_change"].replace([np.inf, -np.inf], np.nan).fillna(0)

        # Rolling 5-year average CO2
        if "co2" in df.columns:
            df["co2_rolling5"] = (
                df.groupby("country")["co2"]
                .transform(lambda x: x.rolling(5, min_periods=1).mean())
            )

        # Fossil fuel ratio
        fossil_cols = [c for c in ["coal_co2", "oil_co2", "gas_co2"] if c in df.columns]
        if fossil_cols and "co2" in df.columns:
            df["fossil_ratio"] = df[fossil_cols].sum(axis=1) / (df["co2"] + 1e-9)
            df["fossil_ratio"] = df["fossil_ratio"].clip(0, 1)

        # Renewable proxy (1 – fossil_ratio)
        if "fossil_ratio" in df.columns:
            df["renewable_proxy"] = 1 - df["fossil_ratio"]

        # CO2 intensity of economy
        if "co2" in df.columns and "gdp" in df.columns:
            df["co2_intensity"] = df["co2"] / (df["gdp"] + 1e-9)

        self.df_processed = df
        print(f"[Pipeline] Features engineered → {self.df_processed.shape[1]} columns")
        return self

    # ── 4. NORMALISE ─────────────────────────
    def normalise(self) -> pd.DataFrame:
        df = self.df_processed.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude = ["year", "population", "gdp"]
        scale_cols = [c for c in numeric_cols if c not in exclude]

        df[scale_cols] = self.scaler.fit_transform(df[scale_cols])
        print("[Pipeline] Normalisation complete ✓")
        return df

    # ── 5. GET FILTERED SUBSET ───────────────
    def get_country_data(self, country: str) -> pd.DataFrame:
        return self.df_processed[
            self.df_processed["country"] == country
        ].sort_values("year")

    def get_latest_year(self) -> pd.DataFrame:
        idx = self.df_processed.groupby("country")["year"].idxmax()
        return self.df_processed.loc[idx].reset_index(drop=True)

    def get_key_countries(self) -> pd.DataFrame:
        return self.df_processed[
            self.df_processed["country"].isin(KEY_COUNTRIES)
        ]

    # ── 6. RUN FULL PIPELINE ─────────────────
    def run(self, start_year: int = 1990) -> pd.DataFrame:
        return (
            self.load()
                .clean(start_year)
                .engineer_features()
                .df_processed
        )

    # ── 7. EXPORT ────────────────────────────
    def export(self, out_dir: str = "data/processed") -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.df_processed.to_csv(f"{out_dir}/sustainability_processed.csv", index=False)
        self.get_latest_year().to_csv(f"{out_dir}/latest_snapshot.csv", index=False)
        self.get_key_countries().to_csv(f"{out_dir}/key_countries.csv", index=False)
        print(f"[Pipeline] Exported to {out_dir}/ ✓")


# ── QUICK TEST ────────────────────────────────
if __name__ == "__main__":
    pipe = SustainabilityDataPipeline("../data/raw/owid-co2-data.csv")
    df = pipe.run()
    pipe.export("../data/processed")
    print(df.head())
    print(df.dtypes)
