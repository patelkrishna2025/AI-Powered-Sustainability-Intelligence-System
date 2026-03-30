"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Sustainability Scoring Engine (0–100)
=============================================================
Scoring methodology:
  Score = Σ (weight_i × normalised_indicator_i) × 100
  Higher score = more sustainable
=============================================================
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  INDICATOR CONFIGURATION
# ─────────────────────────────────────────────
@dataclass
class Indicator:
    name: str
    col: str
    weight: float          # must sum to 1.0 across all indicators
    lower_is_better: bool  # True for emissions, False for renewables
    global_min: Optional[float] = None
    global_max: Optional[float] = None
    score: float = 0.0


DEFAULT_INDICATORS = [
    Indicator("CO₂ per Capita",          "co2_per_capita",             weight=0.25, lower_is_better=True),
    Indicator("Total GHG per Capita",    "ghg_per_capita",             weight=0.20, lower_is_better=True),
    Indicator("CO₂ Intensity (GDP)",     "co2_per_gdp",                weight=0.15, lower_is_better=True),
    Indicator("Fossil Fuel Ratio",       "fossil_ratio",               weight=0.15, lower_is_better=True),
    Indicator("Energy per Capita",       "energy_per_capita",          weight=0.10, lower_is_better=True),
    Indicator("Methane per Capita",      "methane_per_capita",         weight=0.08, lower_is_better=True),
    Indicator("Temperature Δ (GHG)",     "temperature_change_from_ghg",weight=0.07, lower_is_better=True),
]

# Ensure weights sum to 1
_total_weight = sum(i.weight for i in DEFAULT_INDICATORS)
for ind in DEFAULT_INDICATORS:
    ind.weight /= _total_weight


# ─────────────────────────────────────────────
#  SCORER CLASS
# ─────────────────────────────────────────────
class SustainabilityScorer:
    """
    Calculates a 0–100 sustainability score for each country-year.
    Methodology:
      1. Min-max normalise each indicator globally
      2. Flip direction for lower-is-better indicators
      3. Weighted average → 0–100 score
      4. Add trend bonus (improving YoY) / penalty (worsening)
    """

    def __init__(self, indicators: list[Indicator] = None):
        self.indicators = indicators or DEFAULT_INDICATORS
        self._fitted = False

    # ── FIT (compute global min/max) ─────────
    def fit(self, df: pd.DataFrame) -> "SustainabilityScorer":
        for ind in self.indicators:
            if ind.col in df.columns:
                vals = df[ind.col].replace([np.inf, -np.inf], np.nan).dropna()
                ind.global_min = float(vals.quantile(0.01))  # robust min
                ind.global_max = float(vals.quantile(0.99))  # robust max
        self._fitted = True
        print("[Scorer] Fitted on global dataset ✓")
        return self

    def _normalise(self, value: float, ind: Indicator) -> float:
        """Min-max normalise to [0, 1]."""
        if ind.global_min is None or ind.global_max is None:
            return 0.5
        denom = ind.global_max - ind.global_min
        if denom == 0:
            return 0.5
        norm = (value - ind.global_min) / denom
        norm = np.clip(norm, 0, 1)
        return (1 - norm) if ind.lower_is_better else norm

    # ── SCORE A SINGLE ROW ───────────────────
    def score_row(self, row: pd.Series) -> dict:
        scores = {}
        weighted_sum = 0.0
        total_weight = 0.0

        for ind in self.indicators:
            if ind.col not in row.index:
                continue
            val = row[ind.col]
            if pd.isna(val):
                continue
            norm = self._normalise(float(val), ind)
            scores[ind.name] = round(norm * 100, 2)
            weighted_sum += norm * ind.weight
            total_weight += ind.weight

        base_score = (weighted_sum / total_weight * 100) if total_weight > 0 else 50.0
        return {"base_score": base_score, "breakdown": scores}

    # ── SCORE ENTIRE DATAFRAME ───────────────
    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            self.fit(df)

        results = []
        for _, row in df.iterrows():
            scored = self.score_row(row)
            results.append({
                "country":          row.get("country", "Unknown"),
                "year":             row.get("year", 0),
                "sustainability_score": round(scored["base_score"], 2),
                **{f"score_{k}": v for k, v in scored["breakdown"].items()},
            })

        scored_df = pd.DataFrame(results)

        # Trend bonus: reward YoY improvement
        if "year" in df.columns:
            scored_df = scored_df.sort_values(["country", "year"])
            scored_df["score_yoy_change"] = (
                scored_df.groupby("country")["sustainability_score"]
                .diff()
                .fillna(0)
            )
            # Clamp bonus ±5 points
            bonus = scored_df["score_yoy_change"].clip(-5, 5) * 0.3
            scored_df["sustainability_score"] = (
                scored_df["sustainability_score"] + bonus
            ).clip(0, 100).round(2)

        return scored_df

    # ── GRADE ────────────────────────────────
    @staticmethod
    def grade(score: float) -> str:
        if score >= 80: return "A"
        if score >= 65: return "B"
        if score >= 50: return "C"
        if score >= 35: return "D"
        return "F"

    @staticmethod
    def label(score: float) -> str:
        if score >= 80: return "Excellent 🌿"
        if score >= 65: return "Good 🌱"
        if score >= 50: return "Moderate ⚠️"
        if score >= 35: return "Poor 🔴"
        return "Critical 🚨"

    # ── LEADERBOARD ──────────────────────────
    def leaderboard(self, df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
        scored = self.score_dataframe(df)
        if year:
            scored = scored[scored["year"] == year]
        else:
            # Latest year per country
            idx = scored.groupby("country")["year"].idxmax()
            scored = scored.loc[idx]

        scored = scored.sort_values("sustainability_score", ascending=False).reset_index(drop=True)
        scored["rank"] = scored.index + 1
        scored["grade"] = scored["sustainability_score"].apply(self.grade)
        scored["label"] = scored["sustainability_score"].apply(self.label)
        return scored[["rank", "country", "year", "sustainability_score", "grade", "label"]]

    # ── GLOBAL KPIs ──────────────────────────
    def global_kpis(self, df_scored: pd.DataFrame) -> dict:
        latest_idx = df_scored.groupby("country")["year"].idxmax()
        latest = df_scored.loc[latest_idx]
        return {
            "global_avg_score":    round(latest["sustainability_score"].mean(), 1),
            "global_median_score": round(latest["sustainability_score"].median(), 1),
            "countries_excellent": int((latest["sustainability_score"] >= 80).sum()),
            "countries_critical":  int((latest["sustainability_score"] < 35).sum()),
            "top_country":         latest.loc[latest["sustainability_score"].idxmax(), "country"],
            "bottom_country":      latest.loc[latest["sustainability_score"].idxmin(), "country"],
            "top_score":           round(latest["sustainability_score"].max(), 1),
            "bottom_score":        round(latest["sustainability_score"].min(), 1),
        }


# ─────────────────────────────────────────────
#  WEIGHTED COMPOSITE (alternative approach)
# ─────────────────────────────────────────────
class WeightedCompositeScorer:
    """
    Domain-expert weighted scoring with configurable pillars:
      - Climate    (40%)
      - Energy     (30%)
      - Economy    (20%)
      - Biodiversity (10%)
    """

    PILLARS = {
        "Climate": {
            "cols":   ["co2_per_capita", "temperature_change_from_ghg", "methane_per_capita"],
            "weight": 0.40,
            "lower_is_better": True,
        },
        "Energy": {
            "cols":   ["energy_per_capita", "fossil_ratio"],
            "weight": 0.30,
            "lower_is_better": True,
        },
        "Economy": {
            "cols":   ["co2_per_gdp", "co2_intensity"],
            "weight": 0.20,
            "lower_is_better": True,
        },
        "Land Use": {
            "cols":   ["land_use_change_co2"],
            "weight": 0.10,
            "lower_is_better": True,
        },
    }

    def __init__(self):
        self._global_stats: dict = {}

    def fit(self, df: pd.DataFrame) -> "WeightedCompositeScorer":
        for pillar, cfg in self.PILLARS.items():
            cols = [c for c in cfg["cols"] if c in df.columns]
            self._global_stats[pillar] = {}
            for col in cols:
                vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
                self._global_stats[pillar][col] = {
                    "min": vals.quantile(0.01),
                    "max": vals.quantile(0.99),
                }
        return self

    def score_row(self, row: pd.Series) -> dict:
        pillar_scores = {}
        for pillar, cfg in self.PILLARS.items():
            cols = [c for c in cfg["cols"]
                    if c in row.index and not pd.isna(row.get(c))]
            if not cols:
                pillar_scores[pillar] = 50.0
                continue

            norms = []
            for col in cols:
                stats = self._global_stats.get(pillar, {}).get(col, {})
                mn, mx = stats.get("min", 0), stats.get("max", 1)
                denom = mx - mn if mx != mn else 1
                norm = np.clip((float(row[col]) - mn) / denom, 0, 1)
                if cfg["lower_is_better"]:
                    norm = 1 - norm
                norms.append(norm)
            pillar_scores[pillar] = np.mean(norms) * 100

        total = sum(
            pillar_scores[p] * self.PILLARS[p]["weight"]
            for p in pillar_scores
        )
        return {"total": round(total, 2), "pillars": pillar_scores}

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for _, row in df.iterrows():
            s = self.score_row(row)
            rows.append({
                "country": row.get("country"),
                "year":    row.get("year"),
                "composite_score": s["total"],
                **{f"pillar_{k}": round(v, 2) for k, v in s["pillars"].items()},
            })
        return pd.DataFrame(rows)


# ── QUICK TEST ────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from pipeline.data_pipeline import SustainabilityDataPipeline

    pipe = SustainabilityDataPipeline("../data/raw/owid-co2-data.csv")
    df = pipe.run()

    scorer = SustainabilityScorer()
    scorer.fit(df)
    scored_df = scorer.score_dataframe(df)

    print("\n── Global KPIs ──")
    print(scorer.global_kpis(scored_df))

    print("\n── Top 10 Countries (Latest Year) ──")
    print(scorer.leaderboard(scored_df).head(10).to_string(index=False))
