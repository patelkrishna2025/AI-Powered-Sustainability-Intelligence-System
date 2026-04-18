"""
=============================================================
 AI-Powered Sustainability Intelligence System
 MODULE: Rule-Based Sustainability Chatbot
 Answers questions about CO2, sustainability, countries,
 anomalies, forecasts, and general environmental topics.
=============================================================
"""

import pandas as pd
import numpy as np
import re
from typing import Optional


class SustainabilityChatbot:
    """
    Rule-based chatbot that queries the sustainability dataframe
    and answers natural-language questions about emissions,
    scores, trends, and environmental topics.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.history: list[dict] = []
        self._countries = sorted(df["country"].unique().tolist()) if "country" in df.columns else []
        self._greeting_said = False

    # ─────────────────────────────────────────────
    #  PUBLIC: answer a question
    # ─────────────────────────────────────────────
    def answer(self, question: str) -> str:
        q = question.strip()
        self.history.append({"role": "user", "content": q})
        response = self._route(q)
        self.history.append({"role": "bot", "content": response})
        return response

    # ─────────────────────────────────────────────
    #  ROUTING
    # ─────────────────────────────────────────────
    def _route(self, q: str) -> str:
        ql = q.lower()

        # ── greetings
        if re.search(r"\b(hello|hi|hey|howdy|namaste|hola)\b", ql):
            return ("👋 Hello! I'm your **Sustainability AI Assistant**.\n\n"
                    "Ask me anything like:\n"
                    "- *CO2 emissions of India?*\n"
                    "- *Which country has highest emissions?*\n"
                    "- *What is sustainability score?*\n"
                    "- *Top 5 polluters?*\n"
                    "- *CO2 trend for China?*")

        if re.search(r"\b(help|what can you do|capabilities)\b", ql):
            return self._help()

        # ── country-specific queries
        country = self._detect_country(ql)

        if re.search(r"\b(co2|emission|carbon)\b", ql):
            if country:
                return self._co2_for_country(country, ql)
            return self._global_co2(ql)

        if re.search(r"\b(score|sustainability score|grade|rating)\b", ql):
            if country:
                return self._score_for_country(country)
            return self._global_score_summary()

        if re.search(r"\b(top|highest|worst|most pollut|biggest emitt)\b", ql):
            return self._top_emitters(ql)

        if re.search(r"\b(best|lowest|cleanest|most sustainable|greenest)\b", ql):
            return self._cleanest_countries()

        if re.search(r"\b(trend|over time|year|growth|change)\b", ql):
            if country:
                return self._trend_for_country(country)
            return self._global_trend()

        if re.search(r"\b(temperature|warming|ghg|greenhouse)\b", ql):
            if country:
                return self._temp_for_country(country)
            return self._global_temp()

        if re.search(r"\b(energy|consumption|primary energy)\b", ql):
            if country:
                return self._energy_for_country(country)
            return "Please specify a country for energy data, e.g. *'Energy consumption of Japan?'*"

        if re.search(r"\b(how many|count|total|number of)\b", ql):
            return self._count_facts()

        if re.search(r"\b(compare|vs|versus|difference between)\b", ql):
            return self._compare_prompt(ql)

        if re.search(r"\b(what is|define|explain|meaning)\b", ql):
            return self._definitions(ql)

        if re.search(r"\b(thank|thanks|great|awesome|good)\b", ql):
            return "😊 You're welcome! Feel free to ask more sustainability questions."

        # fallback: try country alone
        if country:
            return self._country_summary(country)

        return ("🤔 I didn't quite catch that. Try:\n"
                "- *'CO2 emissions of Germany?'*\n"
                "- *'Top 5 polluters?'*\n"
                "- *'Sustainability score for Brazil?'*\n"
                "- *'Global CO2 trend?'*\n"
                "Type **help** for all capabilities.")

    # ─────────────────────────────────────────────
    #  HELPER: detect country name in query
    # ─────────────────────────────────────────────
    def _detect_country(self, ql: str) -> Optional[str]:
        for c in self._countries:
            if c.lower() in ql:
                return c
        # common aliases
        aliases = {
            "usa": "United States", "us": "United States", "america": "United States",
            "uk": "United Kingdom", "britain": "United Kingdom", "england": "United Kingdom",
            "uae": "United Arab Emirates",
        }
        for alias, full in aliases.items():
            if re.search(rf"\b{alias}\b", ql) and full in self._countries:
                return full
        return None

    # ─────────────────────────────────────────────
    #  ANSWER GENERATORS
    # ─────────────────────────────────────────────
    def _co2_for_country(self, country: str, ql: str = "") -> str:
        cdf = self.df[self.df["country"] == country]
        if len(cdf) == 0:
            return f"❌ No data found for **{country}**."
        latest = cdf.sort_values("year").iloc[-1]
        co2 = latest.get("co2", None)
        per_cap = latest.get("co2_per_capita", None)
        year = int(latest["year"])
        lines = [f"🏭 **{country} CO₂ Data (latest: {year})**"]
        if co2 is not None:
            lines.append(f"- Total CO₂: **{co2:.2f} Mt**")
        if per_cap is not None:
            lines.append(f"- Per Capita: **{per_cap:.2f} t/person**")
        # YoY change
        if len(cdf) >= 2 and co2 is not None:
            prev = cdf.sort_values("year").iloc[-2].get("co2", None)
            if prev and prev > 0:
                chg = ((co2 - prev) / prev) * 100
                arrow = "📈" if chg > 0 else "📉"
                lines.append(f"- YoY Change: {arrow} **{chg:+.1f}%**")
        return "\n".join(lines)

    def _global_co2(self, ql: str) -> str:
        nations = self.df[self.df["iso_code"].notna()] if "iso_code" in self.df.columns else self.df
        latest_year = int(nations["year"].max())
        latest = nations[nations["year"] == latest_year]
        total = latest["co2"].sum() if "co2" in latest.columns else 0
        avg = latest["co2_per_capita"].mean() if "co2_per_capita" in latest.columns else 0
        return (f"🌍 **Global CO₂ ({latest_year})**\n"
                f"- Total Emissions: **{total:,.1f} Mt**\n"
                f"- Avg Per Capita: **{avg:.2f} t/person**\n"
                f"- Countries tracked: **{latest['country'].nunique()}**")

    def _score_for_country(self, country: str) -> str:
        cdf = self.df[self.df["country"] == country]
        if "sustainability_score" not in cdf.columns:
            return f"ℹ️ Sustainability scores not loaded yet. Use the **Leaderboard** tab."
        if len(cdf) == 0:
            return f"❌ No score data for **{country}**."
        latest = cdf.sort_values("year").iloc[-1]
        score = latest["sustainability_score"]
        grade = _grade(score)
        return (f"🌱 **{country} Sustainability Score**\n"
                f"- Score: **{score:.1f} / 100**\n"
                f"- Grade: **{grade}**\n"
                f"- Year: {int(latest['year'])}")

    def _global_score_summary(self) -> str:
        if "sustainability_score" not in self.df.columns:
            return "ℹ️ Sustainability scores not available in raw data. Check the Leaderboard tab."
        latest_idx = self.df.groupby("country")["year"].idxmax()
        snap = self.df.loc[latest_idx]
        avg = snap["sustainability_score"].mean()
        best = snap.loc[snap["sustainability_score"].idxmax()]
        worst = snap.loc[snap["sustainability_score"].idxmin()]
        return (f"🌍 **Global Sustainability Snapshot**\n"
                f"- Average Score: **{avg:.1f}/100**\n"
                f"- 🥇 Best: **{best['country']}** ({best['sustainability_score']:.1f})\n"
                f"- 🚨 Worst: **{worst['country']}** ({worst['sustainability_score']:.1f})")

    def _top_emitters(self, ql: str) -> str:
        n = 5
        m = re.search(r"\b(\d+)\b", ql)
        if m:
            n = min(int(m.group(1)), 15)
        latest_idx = self.df.groupby("country")["year"].idxmax()
        snap = self.df.loc[latest_idx]
        top = snap.nlargest(n, "co2")[["country", "co2", "co2_per_capita"]].reset_index(drop=True)
        lines = [f"🏭 **Top {n} CO₂ Emitters (latest year)**\n"]
        for i, row in top.iterrows():
            cap = f" | {row['co2_per_capita']:.1f} t/cap" if pd.notna(row.get("co2_per_capita")) else ""
            lines.append(f"{i+1}. **{row['country']}** — {row['co2']:.1f} Mt{cap}")
        return "\n".join(lines)

    def _cleanest_countries(self) -> str:
        latest_idx = self.df.groupby("country")["year"].idxmax()
        snap = self.df.loc[latest_idx]
        snap_valid = snap[snap["co2"] > 0]
        bot = snap_valid.nsmallest(5, "co2_per_capita")[["country", "co2_per_capita"]].reset_index(drop=True)
        lines = ["🌿 **Lowest Per-Capita CO₂ Countries**\n"]
        for i, row in bot.iterrows():
            lines.append(f"{i+1}. **{row['country']}** — {row['co2_per_capita']:.2f} t/person")
        return "\n".join(lines)

    def _trend_for_country(self, country: str) -> str:
        cdf = self.df[self.df["country"] == country].sort_values("year")
        if len(cdf) < 3:
            return f"❌ Not enough data for **{country}**."
        first = cdf.iloc[0]; last = cdf.iloc[-1]
        co2_start = first.get("co2", 0); co2_end = last.get("co2", 0)
        if co2_start and co2_start > 0:
            overall_chg = ((co2_end - co2_start) / co2_start) * 100
            direction = "📈 Increased" if overall_chg > 0 else "📉 Decreased"
        else:
            overall_chg = 0; direction = "➡️ Stable"
        peak = cdf.loc[cdf["co2"].idxmax()]
        return (f"📊 **{country} CO₂ Trend ({int(first['year'])}–{int(last['year'])})**\n"
                f"- Start: **{co2_start:.2f} Mt** → End: **{co2_end:.2f} Mt**\n"
                f"- Overall: {direction} **{overall_chg:+.1f}%**\n"
                f"- 🔝 Peak Year: **{int(peak['year'])}** ({peak['co2']:.2f} Mt)")

    def _global_trend(self) -> str:
        nations = self.df[self.df["iso_code"].notna()] if "iso_code" in self.df.columns else self.df
        yearly = nations.groupby("year")["co2"].sum().reset_index()
        first = yearly.iloc[0]; last = yearly.iloc[-1]
        chg = ((last["co2"] - first["co2"]) / first["co2"]) * 100
        return (f"🌍 **Global CO₂ Trend**\n"
                f"- {int(first['year'])}: **{first['co2']:,.0f} Mt**\n"
                f"- {int(last['year'])}: **{last['co2']:,.0f} Mt**\n"
                f"- Change: 📈 **+{chg:.1f}%** over {int(last['year'] - first['year'])} years")

    def _temp_for_country(self, country: str) -> str:
        cdf = self.df[self.df["country"] == country]
        col = "temperature_change_from_ghg"
        if col not in cdf.columns:
            return f"❌ Temperature data not available for **{country}**."
        latest = cdf.sort_values("year").iloc[-1]
        val = latest.get(col, None)
        return (f"🌡️ **{country} Temperature Change from GHG** (latest: {int(latest['year'])})\n"
                f"- Temperature Change: **{val:.3f} °C**")

    def _global_temp(self) -> str:
        col = "temperature_change_from_ghg"
        if col not in self.df.columns:
            return "❌ Temperature data not found in dataset."
        latest_idx = self.df.groupby("country")["year"].idxmax()
        snap = self.df.loc[latest_idx]
        avg = snap[col].mean()
        worst = snap.loc[snap[col].idxmax()]
        return (f"🌡️ **Global Temperature Change from GHG**\n"
                f"- World Average: **{avg:.3f} °C**\n"
                f"- Highest Impact: **{worst['country']}** ({worst[col]:.3f} °C)")

    def _energy_for_country(self, country: str) -> str:
        cdf = self.df[self.df["country"] == country]
        col = "primary_energy_consumption"
        if col not in cdf.columns or len(cdf) == 0:
            return f"❌ Energy data not available for **{country}**."
        latest = cdf.sort_values("year").iloc[-1]
        val = latest.get(col, None)
        per_cap = latest.get("energy_per_capita", None)
        lines = [f"⚡ **{country} Energy Consumption (latest: {int(latest['year'])})**"]
        if val: lines.append(f"- Primary Energy: **{val:,.1f} TWh**")
        if per_cap: lines.append(f"- Per Capita: **{per_cap:.2f} kWh**")
        return "\n".join(lines)

    def _country_summary(self, country: str) -> str:
        cdf = self.df[self.df["country"] == country]
        if len(cdf) == 0:
            return f"❌ No data found for **{country}**."
        latest = cdf.sort_values("year").iloc[-1]
        year = int(latest["year"])
        lines = [f"🌍 **{country} Summary ({year})**"]
        for col, label in [("co2", "CO₂ Emissions"), ("co2_per_capita", "CO₂ Per Capita"),
                           ("primary_energy_consumption", "Primary Energy"),
                           ("temperature_change_from_ghg", "Temp Change (GHG)")]:
            if col in latest.index and pd.notna(latest[col]):
                lines.append(f"- {label}: **{latest[col]:.2f}**")
        return "\n".join(lines)

    def _count_facts(self) -> str:
        n_countries = self.df["country"].nunique()
        yr_min = int(self.df["year"].min()); yr_max = int(self.df["year"].max())
        n_rows = len(self.df)
        return (f"📊 **Dataset Facts**\n"
                f"- Countries: **{n_countries}**\n"
                f"- Year Range: **{yr_min} – {yr_max}**\n"
                f"- Total Records: **{n_rows:,}**\n"
                f"- Columns: **{self.df.shape[1]}**")

    def _compare_prompt(self, ql: str) -> str:
        countries = [c for c in self._countries if c.lower() in ql]
        if len(countries) < 2:
            return "Please name two countries to compare, e.g. *'Compare India vs China CO2?'*"
        rows = []
        latest_idx = self.df.groupby("country")["year"].idxmax()
        snap = self.df.loc[latest_idx]
        for c in countries[:2]:
            row = snap[snap["country"] == c]
            if len(row) > 0:
                r = row.iloc[0]
                rows.append(f"**{c}**: CO₂={r.get('co2',0):.1f} Mt | Per Capita={r.get('co2_per_capita',0):.2f} t")
        return "⚖️ **Comparison**\n" + "\n".join(rows)

    def _definitions(self, ql: str) -> str:
        if "co2" in ql or "carbon dioxide" in ql:
            return "🏭 **CO₂** (Carbon Dioxide) is the primary greenhouse gas emitted by human activities, measured in **megatonnes (Mt)**."
        if "ghg" in ql or "greenhouse" in ql:
            return "🌡️ **GHGs** (Greenhouse Gases) include CO₂, methane (CH₄), and nitrous oxide (N₂O). They trap heat in Earth's atmosphere."
        if "sustainability" in ql or "score" in ql:
            return ("🌱 **Sustainability Score** (0–100) measures a country's environmental performance:\n"
                    "- **80–100** → Grade A (Excellent 🟢)\n"
                    "- **65–80**  → Grade B (Good 🟡)\n"
                    "- **50–65**  → Grade C (Moderate 🟠)\n"
                    "- **35–50**  → Grade D (Poor 🔴)\n"
                    "- **0–35**   → Grade F (Critical ⛔)")
        if "ndvi" in ql or "deforest" in ql:
            return "🌲 **NDVI** (Normalized Difference Vegetation Index) measures vegetation density. Low NDVI indicates deforestation or bare land."
        return "❓ Try asking about CO₂, GHG, sustainability score, or NDVI."

    def _help(self) -> str:
        return ("🤖 **What I can answer:**\n\n"
                "🏭 **Emissions**\n"
                "- *CO₂ emissions of India?*\n"
                "- *Top 10 polluters?*\n"
                "- *Global CO₂ total?*\n\n"
                "🌱 **Sustainability**\n"
                "- *Sustainability score for Germany?*\n"
                "- *Which countries are cleanest?*\n\n"
                "📈 **Trends**\n"
                "- *CO₂ trend for China?*\n"
                "- *Global emissions over time?*\n\n"
                "🌡️ **Temperature & Energy**\n"
                "- *Temperature change for Brazil?*\n"
                "- *Energy consumption of Japan?*\n\n"
                "📊 **Data Facts**\n"
                "- *How many countries are tracked?*\n"
                "- *Compare India vs China?*")


# ─────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────
def _grade(score: float) -> str:
    if score >= 80: return "A 🟢"
    if score >= 65: return "B 🟡"
    if score >= 50: return "C 🟠"
    if score >= 35: return "D 🔴"
    return "F ⛔"
