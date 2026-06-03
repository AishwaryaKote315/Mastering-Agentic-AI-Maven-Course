# ESG Greenwashing Detector

AI-powered Streamlit application for analysing ESG disclosures, emissions trends, and potential greenwashing risk.

## Features

- CSV upload and dataset overview
- Greenwashing risk scoring and category classification
- Sector-level greenwashing dashboard
- Company ESG deep dive with sector benchmarking
- Combined ESG, emissions, revenue, and risk trend chart
- Groq-powered ESG analyst chat

## Required CSV Columns

```text
year, company, ticker, sector, country, revenue_usd_bn,
scope1_emissions_mt_co2e, scope2_emissions_mt_co2e,
scope3_emissions_mt_co2e, total_s1_s2_mt_co2e,
yoy_scope1_change_pct, carbon_intensity_tco2e_per_musd,
esg_score_0_100, cdp_climate_score, net_zero_target_set,
sbti_committed, emissions_disclosed, third_party_verified,
greenwashing_flag
```

## Run Locally

```powershell
pip install -r requirements.txt
$env:GROQ_API_KEY="your_groq_key_here"
python -m streamlit run app.py
```

The app also supports Streamlit secrets:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

## Deploy On Streamlit Community Cloud

1. Push this folder to GitHub.
2. Go to https://share.streamlit.io.
3. Create a new app from your repository.
4. Set the main file path to:

```text
ESG-Greenwashing-Detector/app.py
```

5. In Streamlit Cloud, open **App settings > Secrets** and add:

```toml
GROQ_API_KEY = "your_groq_key_here"
```

Do not commit your Groq API key to GitHub.
