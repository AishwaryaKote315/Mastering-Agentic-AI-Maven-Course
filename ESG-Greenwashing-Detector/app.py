import json
import os
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


REQUIRED_COLUMNS = [
    "year",
    "company",
    "ticker",
    "sector",
    "country",
    "revenue_usd_bn",
    "scope1_emissions_mt_co2e",
    "scope2_emissions_mt_co2e",
    "scope3_emissions_mt_co2e",
    "total_s1_s2_mt_co2e",
    "yoy_scope1_change_pct",
    "carbon_intensity_tco2e_per_musd",
    "esg_score_0_100",
    "cdp_climate_score",
    "net_zero_target_set",
    "sbti_committed",
    "emissions_disclosed",
    "third_party_verified",
    "greenwashing_flag",
]

NUMERIC_COLUMNS = [
    "year",
    "revenue_usd_bn",
    "scope1_emissions_mt_co2e",
    "scope2_emissions_mt_co2e",
    "scope3_emissions_mt_co2e",
    "total_s1_s2_mt_co2e",
    "yoy_scope1_change_pct",
    "carbon_intensity_tco2e_per_musd",
    "esg_score_0_100",
    "greenwashing_flag",
]

BOOLEAN_COLUMNS = [
    "net_zero_target_set",
    "sbti_committed",
    "emissions_disclosed",
    "third_party_verified",
]

RISK_COLORS = {"Low": "#15803d", "Medium": "#d97706", "High": "#b91c1c"}


st.set_page_config(
    page_title="ESG Greenwashing Detector",
    page_icon="ESG",
    layout="wide",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 28rem),
                radial-gradient(circle at top right, rgba(217, 119, 6, 0.10), transparent 24rem),
                linear-gradient(180deg, #f8fbff 0%, #ffffff 42%);
        }
        div[data-testid="metric-container"] {
            border: 1px solid #cbd7e6;
            border-radius: 8px;
            padding: 12px 14px;
            background: linear-gradient(135deg, #ffffff 0%, #eef6ff 100%);
            box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
        }
        .risk-note {
            border-left: 4px solid #b91c1c;
            background: #fff7f7;
            padding: 10px 12px;
            border-radius: 6px;
            margin: 8px 0 14px;
        }
        .ok-note {
            border-left: 4px solid #15803d;
            background: #f5fff8;
            padding: 10px 12px;
            border-radius: 6px;
            margin: 8px 0 14px;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #d8e2ef;
            border-radius: 8px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def configure_groq_key() -> None:
    st.sidebar.header("AI Configuration")
    configured_key = get_configured_groq_api_key()

    if configured_key:
        st.session_state.pop("groq_api_key", None)
        st.sidebar.success("Groq key loaded automatically.")
        return

    entered_key = st.sidebar.text_input(
        "Groq API Key",
        value="",
        type="password",
        help="Shown only when GROQ_API_KEY is not available in Streamlit secrets or environment variables.",
    )
    if entered_key:
        st.session_state["groq_api_key"] = entered_key.strip()
        st.sidebar.success("Groq key entered for this session.")
    else:
        st.sidebar.info("Add GROQ_API_KEY to Streamlit secrets or environment variables to load it automatically.")


def get_groq_api_key() -> str | None:
    return get_configured_groq_api_key() or st.session_state.get("groq_api_key")


def get_configured_groq_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None

    env_key = os.getenv("GROQ_API_KEY")
    return secret_key or env_key


def to_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "set", "committed"}


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = clean.columns.str.strip()

    for column in NUMERIC_COLUMNS:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    for column in BOOLEAN_COLUMNS:
        if column in clean.columns:
            clean[column] = clean[column].apply(to_bool)

    clean["greenwashing_flag"] = clean["greenwashing_flag"].fillna(0).astype(int).clip(0, 1)
    clean["year"] = clean["year"].astype("Int64")
    return clean


def validate_data(df: pd.DataFrame) -> list[str]:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    return missing


def minmax_score(series: pd.Series, default: float = 50.0) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return pd.Series(default, index=series.index)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(default, index=series.index)
    return ((values - min_value) / (max_value - min_value) * 100).fillna(default)


def add_risk_scores(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    esg_risk = 100 - scored["esg_score_0_100"].fillna(scored["esg_score_0_100"].median()).fillna(50)
    carbon_risk = minmax_score(scored["carbon_intensity_tco2e_per_musd"])
    trend_risk = ((scored["yoy_scope1_change_pct"].fillna(0) + 25) / 75 * 100).clip(0, 100)

    scored["risk_score"] = (
        esg_risk * 0.22
        + carbon_risk * 0.18
        + trend_risk * 0.14
        + (~scored["net_zero_target_set"]).astype(int) * 12
        + (~scored["sbti_committed"]).astype(int) * 10
        + (~scored["emissions_disclosed"]).astype(int) * 14
        + (~scored["third_party_verified"]).astype(int) * 12
        + scored["greenwashing_flag"].astype(int) * 16
    ).clip(0, 100).round(1)

    scored["risk_category"] = pd.cut(
        scored["risk_score"],
        bins=[-1, 34.99, 64.99, 100],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    return scored


def latest_company_rows(scored: pd.DataFrame) -> pd.DataFrame:
    latest = (
        scored.sort_values(["company", "year"])
        .groupby("company", as_index=False)
        .tail(1)
        .sort_values("risk_score", ascending=False)
    )
    return latest.reset_index(drop=True)


def groq_chat(messages: list[dict[str, str]], temperature: float = 0.2) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set.")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("The groq package is not installed. Run: pip install groq") from exc

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


def generate_company_assessment(company_df: pd.DataFrame, latest: pd.Series) -> str:
    prompt = (
        "Write a 3 sentence professional ESG assessment. Explain ESG trend, emissions trend, "
        "carbon intensity, net-zero target, SBTi commitment, verification, and greenwashing concerns. "
        "Use only this compact company context.\n\n"
        f"Context: {company_context(company_df, latest)}"
    )
    try:
        return groq_chat(
            [
                {
                    "role": "system",
                    "content": "You are a concise ESG analyst. Return exactly 3 polished business sentences.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,
        )
    except Exception:
        return fallback_assessment(company_df, latest)


def fallback_assessment(company_df: pd.DataFrame, latest: pd.Series) -> str:
    ordered = company_df.sort_values("year")
    esg_change = ordered["esg_score_0_100"].iloc[-1] - ordered["esg_score_0_100"].iloc[0]
    intensity_change = (
        ordered["carbon_intensity_tco2e_per_musd"].iloc[-1]
        - ordered["carbon_intensity_tco2e_per_musd"].iloc[0]
    )
    scope1_change = (
        ordered["scope1_emissions_mt_co2e"].iloc[-1] - ordered["scope1_emissions_mt_co2e"].iloc[0]
    )
    risk_phrase = (
        "elevated greenwashing concern"
        if latest["risk_category"] == "High"
        else "moderate greenwashing concern"
        if latest["risk_category"] == "Medium"
        else "lower greenwashing concern"
    )
    commitments = []
    commitments.append("has a net zero target" if latest["net_zero_target_set"] else "does not show a net zero target")
    commitments.append("has an SBTi commitment" if latest["sbti_committed"] else "does not show an SBTi commitment")
    verification = "third-party verified" if latest["third_party_verified"] else "not third-party verified"

    return (
        f"{latest['company']} has an ESG score change of {esg_change:.1f} points over the uploaded period, "
        f"with a latest score of {latest['esg_score_0_100']:.1f}. Carbon intensity changed by "
        f"{intensity_change:.1f} tCO2e per MUSD and Scope 1 emissions changed by {scope1_change:.1f} Mt CO2e. "
        f"The company {commitments[0]} and {commitments[1]}, while its latest disclosures are {verification}. "
        f"Based on the risk score of {latest['risk_score']:.1f}, the profile indicates {risk_phrase}, especially "
        f"where disclosure quality, verification, emissions trend, and flagged greenwashing signals are weak."
    )


def company_context(company_df: pd.DataFrame, latest: pd.Series) -> str:
    context = {
        "latest_company_profile": {
            "company": latest["company"],
            "sector": latest["sector"],
            "country": latest["country"],
            "risk_score": float(latest["risk_score"]),
            "risk_category": latest["risk_category"],
            "esg_score_0_100": float(latest["esg_score_0_100"]),
            "net_zero_target_set": bool(latest["net_zero_target_set"]),
            "sbti_committed": bool(latest["sbti_committed"]),
            "emissions_disclosed": bool(latest["emissions_disclosed"]),
            "third_party_verified": bool(latest["third_party_verified"]),
            "greenwashing_flag": int(latest["greenwashing_flag"]),
        },
        "historical_records": company_df.sort_values("year")[
            [
                "year",
                "esg_score_0_100",
                "carbon_intensity_tco2e_per_musd",
                "scope1_emissions_mt_co2e",
                "scope2_emissions_mt_co2e",
                "scope3_emissions_mt_co2e",
                "revenue_usd_bn",
                "greenwashing_flag",
            ]
        ].to_dict(orient="records"),
    }
    return json.dumps(context, default=str)


def full_dataset_context(scored: pd.DataFrame, latest: pd.DataFrame) -> str:
    sector_summary = (
        latest.groupby("sector", as_index=False)
        .agg(
            avg_risk_score=("risk_score", "mean"),
            companies=("company", "nunique"),
            high_risk_companies=("risk_category", lambda x: int((x == "High").sum())),
        )
        .round(2)
    )
    flagged_companies = (
        scored[scored["greenwashing_flag"] == 1]
        .groupby(["company", "sector"], as_index=False)
        .agg(
            flagged_years=("year", lambda x: sorted([int(y) for y in x.dropna().unique()])),
            latest_risk_score=("risk_score", "max"),
        )
        .sort_values("latest_risk_score", ascending=False)
    )
    improving_companies = []
    for company, company_df in scored.sort_values("year").groupby("company"):
        if len(company_df) < 2:
            continue
        first = company_df.iloc[0]
        last = company_df.iloc[-1]
        if (
            pd.notna(first["scope1_emissions_mt_co2e"])
            and pd.notna(last["scope1_emissions_mt_co2e"])
            and last["scope1_emissions_mt_co2e"] < first["scope1_emissions_mt_co2e"]
        ):
            improving_companies.append(
                {
                    "company": company,
                    "scope1_change_mt_co2e": round(
                        float(last["scope1_emissions_mt_co2e"] - first["scope1_emissions_mt_co2e"]), 2
                    ),
                    "carbon_intensity_latest": round(float(last["carbon_intensity_tco2e_per_musd"]), 2)
                    if pd.notna(last["carbon_intensity_tco2e_per_musd"])
                    else None,
                }
            )

    compact = {
        "top_25_company_latest_risk_scores": latest.head(25)[
            [
                "company",
                "ticker",
                "sector",
                "country",
                "year",
                "esg_score_0_100",
                "carbon_intensity_tco2e_per_musd",
                "risk_score",
                "risk_category",
                "greenwashing_flag",
            ]
        ].to_dict(orient="records"),
        "sector_risk_summary": sector_summary.to_dict(orient="records"),
        "greenwashing_flagged_companies": flagged_companies.head(50).to_dict(orient="records"),
        "companies_with_scope1_improvement": improving_companies[:50],
    }
    return json.dumps(compact, default=str)


def local_chat_answer(question: str, scored: pd.DataFrame, latest: pd.DataFrame) -> str | None:
    normalized = question.lower()
    if "greenwashing" in normalized and any(word in normalized for word in ["which", "what", "list", "companies"]):
        flagged = (
            scored[scored["greenwashing_flag"] == 1]
            .groupby(["company", "sector"], as_index=False)
            .agg(flagged_years=("year", lambda x: ", ".join(map(str, sorted(x.dropna().astype(int).unique())))))
            .merge(latest[["company", "risk_score", "risk_category"]], on="company", how="left")
            .sort_values("risk_score", ascending=False)
        )
        if flagged.empty:
            return "No companies in the uploaded dataset have `greenwashing_flag = 1`."
        rows = [
            f"- {row.company} ({row.sector}): flagged in {row.flagged_years}; risk score {row.risk_score:.1f} ({row.risk_category})"
            for row in flagged.itertuples(index=False)
        ]
        return "Companies marked with `greenwashing_flag = 1`:\n\n" + "\n".join(rows)

    if "sector" in normalized and "highest" in normalized and "risk" in normalized:
        sector = latest.groupby("sector")["risk_score"].mean().sort_values(ascending=False)
        return f"{sector.index[0]} has the highest average risk score at {sector.iloc[0]:.1f}."

    if "highest" in normalized and "risk" in normalized:
        top = latest.iloc[0]
        return (
            f"{top['company']} has the highest greenwashing risk in the uploaded dataset, "
            f"with a risk score of {top['risk_score']:.1f} ({top['risk_category']})."
        )

    return None


def metric_cards(latest: pd.DataFrame) -> None:
    high = int((latest["risk_category"] == "High").sum())
    medium = int((latest["risk_category"] == "Medium").sum())
    low = int((latest["risk_category"] == "Low").sum())
    avg = latest["risk_score"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High Risk Companies", high)
    c2.metric("Medium Risk Companies", medium)
    c3.metric("Low Risk Companies", low)
    c4.metric("Average Risk Score", f"{avg:.1f}")


def key_esg_metrics(scored: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "Companies Analysed",
                "Value": f"{latest['company'].nunique():,}",
                "Description": "Unique companies in the uploaded dataset",
            },
            {
                "Metric": "Greenwashing Cases",
                "Value": f"{scored.loc[scored['greenwashing_flag'] == 1, 'company'].nunique():,}",
                "Description": "Companies with at least one historical greenwashing flag",
            },
            {
                "Metric": "Average ESG Score",
                "Value": f"{latest['esg_score_0_100'].mean():.1f}",
                "Description": "Average latest ESG score across companies",
            },
            {
                "Metric": "Net Zero Commitments",
                "Value": f"{int(latest['net_zero_target_set'].sum()):,}",
                "Description": "Companies reporting a net-zero target",
            },
            {
                "Metric": "Third Party Verified",
                "Value": f"{int(latest['third_party_verified'].sum()):,}",
                "Description": "Companies with third-party verification",
            },
        ]
    )


def greenwashing_cases_by_sector_chart(scored: pd.DataFrame) -> alt.Chart:
    flagged = (
        scored[scored["greenwashing_flag"] == 1]
        .groupby("sector", as_index=False)
        .agg(Greenwashing_Cases=("company", "nunique"))
        .sort_values("Greenwashing_Cases", ascending=False)
    )
    if flagged.empty:
        flagged = pd.DataFrame({"sector": ["No flagged sectors"], "Greenwashing_Cases": [0]})

    return (
        alt.Chart(flagged)
        .mark_bar(cornerRadiusEnd=4, color="#b91c1c")
        .encode(
            x=alt.X("Greenwashing_Cases:Q", title="Greenwashing Cases"),
            y=alt.Y("sector:N", sort="-x", title="Sector"),
            tooltip=["sector", "Greenwashing_Cases"],
        )
        .properties(height=300)
    )


def greenwashing_leaderboard(latest: pd.DataFrame) -> pd.DataFrame:
    leaderboard = latest.head(10).copy()
    leaderboard.insert(0, "Rank", range(1, len(leaderboard) + 1))
    return leaderboard[["Rank", "company", "sector", "risk_score", "risk_category"]].rename(
        columns={
            "company": "Company",
            "sector": "Sector",
            "risk_score": "Risk Score",
            "risk_category": "Risk Category",
        }
    )
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Risk Category:N", sort=["Low", "Medium", "High"]),
            y="Companies:Q",
            color=alt.Color("Risk Category:N", scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values()))),
            tooltip=["Risk Category", "Companies"],
        )
        .properties(height=290)
    )


def company_ranking_chart(latest: pd.DataFrame) -> alt.Chart:
    ranking = latest.sort_values("risk_score")
    return (
        alt.Chart(ranking)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("risk_score:Q", title="Risk Score"),
            y=alt.Y(
                "company:N",
                sort=None,
                title="Company",
                axis=alt.Axis(labelLimit=260, labelFontSize=11),
            ),
            color=alt.Color("risk_category:N", scale=alt.Scale(domain=list(RISK_COLORS), range=list(RISK_COLORS.values()))),
            tooltip=["company", "sector", "risk_score", "risk_category", "greenwashing_flag"],
        )
        .properties(height=max(420, len(ranking) * 24))
    )


def sector_comparison_chart(latest: pd.DataFrame) -> alt.Chart:
    sector = (
        latest.groupby("sector", as_index=False)["risk_score"]
        .mean()
        .sort_values("risk_score", ascending=False)
    )
    return (
        alt.Chart(sector)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color="#2563eb")
        .encode(
            x=alt.X("sector:N", sort="-y", title="Sector"),
            y=alt.Y("risk_score:Q", title="Average Risk Score"),
            tooltip=["sector", alt.Tooltip("risk_score:Q", format=".1f")],
        )
        .properties(height=320)
    )


def line_chart(company_df: pd.DataFrame, y: str, title: str, color: str = "#2563eb") -> alt.Chart:
    base = (
        alt.Chart(company_df)
        .mark_line(point=True, color=color)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y(f"{y}:Q", title=title),
            tooltip=["year", alt.Tooltip(f"{y}:Q", title=title, format=".2f"), "greenwashing_flag"],
        )
    )
    flagged = (
        alt.Chart(company_df[company_df["greenwashing_flag"] == 1])
        .mark_point(size=130, color="#b91c1c", shape="diamond")
        .encode(x="year:O", y=f"{y}:Q", tooltip=["year", "greenwashing_flag"])
    )
    return (base + flagged).properties(height=260)


def revenue_emissions_chart(company_df: pd.DataFrame) -> alt.LayerChart:
    long_df = company_df.melt(
        id_vars=["year", "greenwashing_flag"],
        value_vars=["revenue_usd_bn", "total_s1_s2_mt_co2e"],
        var_name="Metric",
        value_name="Value",
    )
    labels = {
        "revenue_usd_bn": "Revenue USD bn",
        "total_s1_s2_mt_co2e": "Total Scope 1+2 Mt CO2e",
    }
    long_df["Metric"] = long_df["Metric"].map(labels)

    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Value:Q", title="Value"),
            color=alt.Color("Metric:N", scale=alt.Scale(range=["#2563eb", "#b91c1c"])),
            tooltip=["year", "Metric", alt.Tooltip("Value:Q", format=".2f")],
        )
        .properties(height=300)
    )


def combined_company_metrics_chart(company_df: pd.DataFrame) -> alt.Chart:
    metrics = {
        "esg_score_0_100": "ESG Score",
        "carbon_intensity_tco2e_per_musd": "Carbon Intensity",
        "scope1_emissions_mt_co2e": "Scope 1 Emissions",
        "scope2_emissions_mt_co2e": "Scope 2 Emissions",
        "scope3_emissions_mt_co2e": "Scope 3 Emissions",
        "revenue_usd_bn": "Revenue",
        "risk_score": "Risk Score",
    }
    trend = company_df[["year", "greenwashing_flag", *metrics.keys()]].melt(
        id_vars=["year", "greenwashing_flag"],
        value_vars=list(metrics.keys()),
        var_name="metric",
        value_name="actual_value",
    )
    trend["Metric"] = trend["metric"].map(metrics)
    trend["Normalized Value"] = trend.groupby("Metric")["actual_value"].transform(
        lambda values: pd.Series(50, index=values.index)
        if values.max() == values.min()
        else (values - values.min()) / (values.max() - values.min()) * 100
    )

    line = (
        alt.Chart(trend)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:O", title="Year"),
            y=alt.Y("Normalized Value:Q", title="Normalized Trend (0-100)"),
            color=alt.Color(
                "Metric:N",
                scale=alt.Scale(
                    range=["#15803d", "#d97706", "#b91c1c", "#dc2626", "#9333ea", "#2563eb", "#0f766e"]
                ),
            ),
            tooltip=[
                "year",
                "Metric",
                alt.Tooltip("actual_value:Q", title="Actual Value", format=".2f"),
                alt.Tooltip("Normalized Value:Q", format=".1f"),
                "greenwashing_flag",
            ],
        )
    )
    flagged = (
        alt.Chart(trend[trend["greenwashing_flag"] == 1])
        .mark_rule(color="#b91c1c", strokeDash=[4, 4])
        .encode(x="year:O", tooltip=["year", "greenwashing_flag"])
    )
    return (line + flagged).properties(height=430)


def trend_chart_explanation(company_df: pd.DataFrame) -> str:
    flagged_years = sorted(company_df.loc[company_df["greenwashing_flag"] == 1, "year"].dropna().astype(int).unique())
    flagged_text = ", ".join(map(str, flagged_years)) if flagged_years else "none"
    return (
        "This chart compares ESG score, emissions, revenue, carbon intensity, and risk score on a normalized "
        "0-100 scale so that metrics with different units can be viewed together. A higher line means that "
        "metric is higher relative to that company's own historical range, not necessarily higher in absolute "
        "industry terms. Red dashed vertical lines mark years where `greenwashing_flag = 1`; for this company, "
        f"the flagged year(s) are {flagged_text}. The most important pattern to watch is whether ESG score or "
        "revenue improves while emissions, carbon intensity, or risk score also rise, because that may suggest "
        "a gap between sustainability claims and emissions performance."
    )


def yes_no(value: Any) -> str:
    return "Yes" if to_bool(value) else "No"


def company_scorecard(company_df: pd.DataFrame, latest: pd.Series) -> pd.DataFrame:
    flagged_years = sorted(company_df.loc[company_df["greenwashing_flag"] == 1, "year"].dropna().astype(int).unique())
    return pd.DataFrame(
        [
            {"Metric": "ESG Score", "Value": f"{latest['esg_score_0_100']:.1f}"},
            {"Metric": "Risk Score", "Value": f"{latest['risk_score']:.1f}"},
            {"Metric": "Risk Category", "Value": latest["risk_category"]},
            {"Metric": "Net Zero Target", "Value": yes_no(latest["net_zero_target_set"])},
            {"Metric": "SBTi Commitment", "Value": yes_no(latest["sbti_committed"])},
            {"Metric": "Third Party Verified", "Value": yes_no(latest["third_party_verified"])},
            {
                "Metric": "Greenwashing Flagged Years",
                "Value": ", ".join(map(str, flagged_years)) if flagged_years else "None",
            },
        ]
    )


def sector_benchmark(latest: pd.DataFrame, selected_company: str) -> pd.DataFrame:
    company_row = latest[latest["company"] == selected_company].iloc[0]
    sector_rows = latest[latest["sector"] == company_row["sector"]]
    return pd.DataFrame(
        [
            {
                "Metric": "ESG Score",
                selected_company: f"{company_row['esg_score_0_100']:.1f}",
                "Sector Avg": f"{sector_rows['esg_score_0_100'].mean():.1f}",
            },
            {
                "Metric": "Carbon Intensity",
                selected_company: f"{company_row['carbon_intensity_tco2e_per_musd']:.2f}",
                "Sector Avg": f"{sector_rows['carbon_intensity_tco2e_per_musd'].mean():.2f}",
            },
            {
                "Metric": "Risk Score",
                selected_company: f"{company_row['risk_score']:.1f}",
                "Sector Avg": f"{sector_rows['risk_score'].mean():.1f}",
            },
        ]
    )


def tab_upload() -> pd.DataFrame | None:
    st.subheader("Welcome to ESG Greenwashing Detector")
    st.write(
        "This application analyses corporate sustainability disclosures, emissions performance, "
        "and ESG indicators to identify potential greenwashing risks and compare companies across sectors."
    )
    st.subheader("Data Upload & Overview")
    uploaded_file = st.file_uploader("Upload ESG company CSV", type=["csv"], accept_multiple_files=False)

    if uploaded_file is None:
        st.info("Upload a CSV containing the required ESG disclosure fields to begin.")
        return None

    raw = pd.read_csv(uploaded_file)
    missing = validate_data(raw)
    if missing:
        st.error("The uploaded CSV is missing required columns:")
        st.write(missing)
        return None

    df = prepare_data(raw)
    st.session_state["esg_data"] = df
    st.success("CSV uploaded successfully. Continue to the dashboard or company deep dive tabs.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Companies", f"{df['company'].nunique():,}")
    years = df["year"].dropna()
    c3.metric("Years Covered", f"{int(years.min())} - {int(years.max())}" if not years.empty else "N/A")

    c4, c5, c6 = st.columns(3)
    c4.metric("Sectors Covered", f"{df['sector'].nunique():,}")
    c5.metric("Countries Covered", f"{df['country'].nunique():,}")
    c6.metric("Greenwashing Flagged Records", f"{int(df['greenwashing_flag'].sum()):,}")

    with st.expander("Optional: View Uploaded Data"):
        st.dataframe(df, use_container_width=True)

    return df


def tab_dashboard(scored: pd.DataFrame, latest: pd.DataFrame) -> None:
    st.subheader("Greenwashing Risk Dashboard")
    st.markdown("#### Key ESG Metrics")
    metrics = key_esg_metrics(scored, latest)
    metric_columns = st.columns(len(metrics))
    for column, row in zip(metric_columns, metrics.itertuples(index=False)):
        column.metric(row.Metric, row.Value, help=row.Description)

    left, right = st.columns([1, 1.3])
    with left:
        st.markdown("#### Greenwashing Cases by Sector")
        st.altair_chart(greenwashing_cases_by_sector_chart(scored), use_container_width=True)
    with right:
        st.markdown("#### Companies Most Likely to Greenwash")
        st.altair_chart(company_ranking_chart(latest), use_container_width=True)

    st.markdown("#### Risk Score Visualization")
    st.dataframe(greenwashing_leaderboard(latest), use_container_width=True, hide_index=True)

    with st.expander("How is Risk Score Calculated?"):
        st.write(
            "Risk increases when ESG score is low, carbon intensity is high, Scope 1 emissions are rising, "
            "a company has no net-zero target, no SBTi commitment, no emissions disclosure, no third-party "
            "verification, or historical greenwashing flags exist. The score combines these factors into a "
            "0-100 indicator where higher values indicate greater potential greenwashing risk."
        )


def tab_deep_dive(scored: pd.DataFrame, latest: pd.DataFrame) -> None:
    st.subheader("Company ESG Deep Dive")
    sector = st.selectbox("Select sector", sorted(scored["sector"].dropna().unique()))
    sector_companies = sorted(scored.loc[scored["sector"] == sector, "company"].dropna().unique())
    company = st.selectbox("Select company", sector_companies)
    company_df = scored[scored["company"] == company].sort_values("year")
    latest_row = latest[latest["company"] == company].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Latest ESG Score", f"{latest_row['esg_score_0_100']:.1f}")
    c2.metric("Risk Score", f"{latest_row['risk_score']:.1f}")
    c3.metric("Risk Category", latest_row["risk_category"])

    left_snapshot, right_benchmark = st.columns(2)
    with left_snapshot:
        st.markdown("#### Company ESG Snapshot")
        st.dataframe(company_scorecard(company_df, latest_row), use_container_width=True, hide_index=True)
    with right_benchmark:
        st.markdown(f"#### {company} vs {latest_row['sector']} Sector")
        st.dataframe(sector_benchmark(latest, company), use_container_width=True, hide_index=True)

    if company_df["greenwashing_flag"].sum() > 0:
        flagged_years = ", ".join(company_df.loc[company_df["greenwashing_flag"] == 1, "year"].astype(str))
        st.markdown(
            f"<div class='risk-note'><strong>Greenwashing flag detected</strong> in year(s): {flagged_years}.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='ok-note'><strong>No greenwashing flag</strong> appears in the uploaded historical records.</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### ESG, Emissions, Revenue, and Risk Trends")
    st.altair_chart(combined_company_metrics_chart(company_df), use_container_width=True)
    st.info(trend_chart_explanation(company_df))

    st.subheader("AI ESG Assessment")
    assessment_key = f"assessment_{company}_{len(company_df)}_{latest_row['risk_score']}_{bool(get_groq_api_key())}"
    if assessment_key not in st.session_state:
        with st.spinner("Preparing ESG assessment..."):
            st.session_state[assessment_key] = generate_company_assessment(company_df, latest_row)
    st.write(st.session_state[assessment_key])


def tab_chat(scored: pd.DataFrame, latest: pd.DataFrame) -> None:
    st.subheader("AI ESG Analyst")
    st.caption("Powered by Groq when `GROQ_API_KEY` is configured. Uses compact summaries, not the full dataset.")

    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": "Ask me about company risk, sector comparisons, emissions improvements, or ESG trends.",
            }
        ]

    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_question = st.chat_input("Ask an ESG risk question")
    if not user_question:
        return

    st.session_state["chat_messages"].append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    system_context = (
        "You are an ESG analyst for an AI-powered Greenwashing Detector. "
        "Use the uploaded dataset, company risk scores, ESG metrics, emissions metrics, and historical trends. "
        "Be specific, cite company names and metrics where useful, and do not invent data. "
        f"Dataset context JSON: {full_dataset_context(scored, latest)}"
    )
    groq_messages = [{"role": "system", "content": system_context}]
    groq_messages.extend(st.session_state["chat_messages"][-8:])

    with st.chat_message("assistant"):
        with st.spinner("Analyzing ESG data..."):
            try:
                answer = local_chat_answer(user_question, scored, latest)
                if answer is None:
                    answer = groq_chat(groq_messages, temperature=0.15)
            except Exception as exc:
                top_company = latest.iloc[0]
                top_sector = (
                    latest.groupby("sector")["risk_score"].mean().sort_values(ascending=False).index[0]
                )
                answer = (
                    f"Groq could not answer this request: {exc}\n\n"
                    f"From the local scoring model, {top_company['company']} has the highest greenwashing risk "
                    f"with a score of {top_company['risk_score']:.1f} ({top_company['risk_category']}). "
                    f"The highest average-risk sector is {top_sector}. Try a narrower question, such as one company, "
                    f"one sector, or a comparison between two companies."
                )
            st.write(answer)

    st.session_state["chat_messages"].append({"role": "assistant", "content": answer})


def main() -> None:
    apply_styles()
    configure_groq_key()
    st.title("ESG Greenwashing Detector")

    tabs = st.tabs(
        [
            "Data Upload & Overview",
            "Greenwashing Risk Dashboard",
            "Company ESG Deep Dive",
            "AI ESG Analyst Chat",
        ]
    )

    with tabs[0]:
        uploaded_df = tab_upload()

    df = st.session_state.get("esg_data", uploaded_df)
    if df is None:
        for tab in tabs[1:]:
            with tab:
                st.info("Upload a valid ESG CSV in the Data Upload & Overview tab to activate this section.")
        return

    scored = add_risk_scores(df)
    latest = latest_company_rows(scored)
    st.session_state["scored_esg_data"] = scored
    st.session_state["company_risk_scores"] = latest

    with tabs[1]:
        tab_dashboard(scored, latest)
    with tabs[2]:
        tab_deep_dive(scored, latest)
    with tabs[3]:
        tab_chat(scored, latest)


if __name__ == "__main__":
    main()
