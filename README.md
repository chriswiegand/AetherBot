# ⚡ AetherBot

**Quantitative weather prediction market bot for Kalshi KXHIGH temperature contracts.**

AetherBot ingests multi-model weather forecasts, constructs calibrated probability distributions, detects statistical edges against market-implied probabilities, and executes trades using fractional Kelly criterion position sizing with layered risk controls. Named for the classical fifth element — the quintessence theorized to fill the upper atmosphere.

---

## Quick Start

```bash
# Clone
git clone https://github.com/chriswiegand/AetherBot.git
cd AetherBot

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env        # Add your Kalshi API credentials
vi config/settings.yaml      # Tune strategy parameters
vi config/cities.yaml        # City/station mapping

# Backfill historical observations (2 years)
python3 scripts/backfill_history.py --days 730

# Run backtest
python3 scripts/run_backtest.py --days 365

# Start the bot (paper trading)
python3 -m src.scheduler.runner

# Launch the dashboard
python3 dashboard/app.py
# Open http://localhost:5050
```

---

## Architecture

```
Data Sources        GFS Ensemble (31 members) · HRRR 3km · NWS · IEM CLI
      │
      ▼
Data Pipeline       Fetch → Parse → Store → Filter
      │
      ▼
Signal Engine       Ensemble Prob → HRRR Correct → Model Blend → Calibrate
      │
      ▼
Strategy Layer      Edge Detect → Kelly Size → Risk Manage
      │
      ▼
Execution           Paper Trader · Live Trader · Settlement
```

The bot runs a 5-minute core cycle: fetch latest forecasts → compute ensemble probabilities → apply HRRR correction → blend models → calibrate via isotonic regression → detect edges → size via fractional Kelly → check risk limits → execute trade.

---

## Market Coverage

| City | Station | Ticker | Timezone |
|------|---------|--------|----------|
| New York | KNYC (Central Park) | `KXHIGHNY` | America/New_York |
| Chicago | KMDW (Midway) | `KXHIGHCHI` | America/Chicago |
| Miami | KMIA | `KXHIGHMIA` | America/New_York |
| Los Angeles | KLAX | `KXHIGHLAX` | America/Los_Angeles |
| Denver | KDEN | `KXHIGHDEN` | America/Denver |

**Settlement**: NWS CLI "Maximum" observed value (integer °F). "Above X" = strictly greater than X.

---

## Data Sources

| Source | Model | Frequency | Purpose |
|--------|-------|-----------|---------|
| Open-Meteo Ensemble | GFS 0.25° (31 members) | 60 min | Primary probability estimate |
| Open-Meteo Forecast | HRRR 3km | 30 min | Short-range correction |
| NWS API | Gridded forecast | 120 min | Human-machine blend |
| IEM CLI | NWS Climate Reports | Daily 11:15 AM ET | Settlement ground truth |

All data sources are **free and open** — no subscriptions required.

---

## Signal Pipeline

1. **Ensemble Probability** — Count fraction of 31 GFS members exceeding threshold. Each member's max is rounded to integer °F to match NWS CLI convention.

2. **HRRR Correction** — Shift ensemble members by lead-time-weighted fraction of HRRR-ensemble disagreement (45% weight at 0h, decaying to 5% at 48h+).

3. **Model Blend** — Weighted average: Ensemble 60%, HRRR 25%, NWS 15%. Deterministic forecasts converted to probabilities via Gaussian error model (σ = 3°F).

4. **Isotonic Calibration** — Non-parametric monotone mapping from raw to calibrated probabilities using scikit-learn's IsotonicRegression. Walk-forward refit every 50 trades.

5. **Edge Detection** — Edge = P_model - P_market. Minimum 8% edge to trade (6% with HRRR confirmation). Pwin floor of 55%.

---

## Position Sizing

**Fractional Kelly Criterion** (15% of full Kelly):

```
f* = (p × b - q) / b      where b = (1-price)/price, p = Pwin, q = 1-p
f_sized = f* × 0.15
dollar_amount = f_sized × bankroll
```

Kelly naturally sizes proportionally to both edge magnitude and win probability — larger edges and higher confidence produce bigger positions.

### Risk Controls

| Control | Limit | Purpose |
|---------|-------|---------|
| Daily Loss Limit | -$300 | Circuit breaker |
| Max Concurrent Positions | 20 | Portfolio concentration |
| Max Positions per City | 6 | Geographic concentration |
| Max Positions per Date | 4 | Temporal concentration |
| Max Position Size | 10% of bankroll or $1,000 | Single-trade cap |

---

## Dashboard

The Flask dashboard (`http://localhost:5050`) provides:

- **Main Dashboard** — Live portfolio view, open positions, PnL chart, trade history with hyperlinked tickers, Brier score tracking
- **Strategy Lab** — Create/edit/activate named strategies, run per-strategy backtests with 6 chart types, parameter grid-search optimization
- **Whitepaper** — Full technical overview with interactive charts

---

## Strategy Lab

The Strategy Lab enables experimentation with different parameter sets:

- **Strategy Management** — Create, clone, and activate named strategy configurations with custom edge thresholds, Kelly fractions, risk limits, and market filters
- **Backtesting** — Run walk-forward backtests per strategy. Results include equity curves, PnL histograms, edge-bucket analysis, and per-city breakdowns
- **Optimization** — Grid search across parameter ranges to find optimal configurations. Sort by Sharpe, PnL, win rate, or Brier score. One-click "Save as Strategy" from best results

---

## Project Structure

```
AetherBot/
├── config/
│   ├── settings.yaml          # Strategy & data source config
│   └── cities.yaml            # City/station definitions
├── src/
│   ├── config/                # Settings & city loaders
│   ├── data/                  # Models, DB, API clients
│   │   ├── models.py          # SQLAlchemy ORM (11 tables)
│   │   ├── ensemble_fetcher.py
│   │   ├── hrrr_fetcher.py
│   │   ├── nws_client.py
│   │   ├── iem_client.py
│   │   └── kalshi_markets.py
│   ├── strategy/              # Signal processing
│   │   ├── ensemble_calc.py   # Ensemble probability computation
│   │   ├── model_blender.py   # Multi-model blending
│   │   ├── calibrator.py      # Isotonic calibration
│   │   ├── edge_detector.py   # Edge detection & filtering
│   │   ├── kelly_sizer.py     # Kelly criterion position sizing
│   │   └── risk_manager.py    # Portfolio risk controls
│   ├── execution/             # Trade execution
│   │   ├── paper_trader.py
│   │   ├── live_trader.py
│   │   └── settlement_checker.py
│   ├── backtest/              # Backtesting framework
│   │   ├── replay_engine.py   # Walk-forward backtest engine
│   │   ├── strategy_runner.py # Per-strategy backtest wrapper
│   │   └── optimizer.py       # Grid search parameter optimization
│   ├── scheduler/             # APScheduler job definitions
│   │   ├── runner.py          # Main entry point
│   │   └── jobs.py            # Job definitions + active strategy loader
│   └── monitoring/            # PnL tracking & alerts
├── dashboard/
│   ├── app.py                 # Flask API (22+ endpoints)
│   ├── index.html             # Main dashboard
│   ├── strategy-lab.html      # Strategy Lab page
│   └── whitepaper.html        # Technical overview
├── scripts/
│   ├── backfill_history.py    # Historical data loader
│   ├── run_backtest.py        # CLI backtest runner
│   ├── smoke_test.py          # End-to-end verification
│   ├── check_status.py        # System health check
│   └── reset_paper_trades.py  # Paper trade reset utility
├── journal/                   # Development journal
│   ├── README.md
│   └── 2026-03-06.md
├── docs/
│   └── whitepaper.md          # Full whitepaper (markdown)
└── data/
    └── weather_bot.db         # SQLite database (gitignored)
```

---

## Database Schema

| Table | Purpose | Volume |
|-------|---------|--------|
| `ensemble_forecasts` | GFS member daily maxes | ~310/city/run |
| `hrrr_forecasts` | HRRR deterministic max | ~2/city/run |
| `nws_forecasts` | NWS 7-day high/low | ~7/city/fetch |
| `observations` | CLI ground truth | 1/city/day |
| `kalshi_markets` | Active contracts | ~12/city (3x/day) |
| `trades` | Paper and live trades | Variable |
| `brier_scores` | Forecast evaluation | 1/settled trade |
| `strategies` | Named strategy configs | User-defined |
| `backtest_runs` | Backtest results + charts | Per strategy |
| `optimization_runs` | Grid search results | Per sweep |

---

## Scheduler Jobs

| Job | Trigger | Frequency |
|-----|---------|-----------|
| Market Discovery | Cron | 6 AM, 12 PM, 6 PM ET |
| GFS Ensemble Fetch | Interval | Every 60 minutes |
| HRRR Fetch | Interval | Every 30 minutes |
| NWS Forecast Fetch | Interval | Every 120 minutes |
| **Scan & Trade** | Interval | **Every 5 minutes** |
| Settlement Check | Cron | 11:15 AM ET daily |
| Daily Report | Cron | 12:00 PM ET daily |

---

## Backtest Results

Walk-forward backtest over 2 years of historical data (5 cities, synthetic markets):

| Metric | Value | Target |
|--------|-------|--------|
| Total Trades | 1,827 | — |
| Win Rate | 71.8% | > 50% |
| Gross PnL | +$6,664 | — |
| Sharpe Ratio | 15.25 | > 1.0 |
| Brier Score | 0.1235 | < 0.20 |
| Final Bankroll | $16,664 | — |

> **Note**: The Sharpe ratio is inflated because synthetic markets are priced at climatological probabilities. Live performance will be lower, but the Brier score of 0.1235 indicates genuine forecast skill.

---

## Configuration

```yaml
strategy:
  edge_threshold: 0.08          # 8% minimum edge to trade
  min_edge_hrrr_confirm: 0.06   # 6% with HRRR confirmation
  min_model_prob: 0.55          # 55% Pwin floor
  fractional_kelly: 0.15        # 15% of full Kelly
  max_position_pct: 0.10        # 10% of bankroll per trade
  max_position_dollars: 1000    # $1,000 absolute cap
  daily_loss_limit: 300         # $300 daily loss limit
  max_concurrent_positions: 20
  max_positions_per_city: 6
  max_positions_per_date: 4
  min_price: 0.08
  max_price: 0.92
  max_lead_hours: 72

model_weights:
  gfs_ensemble: 0.60
  hrrr: 0.25
  nws: 0.15
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ (~7,500 LOC) |
| Database | SQLite (WAL mode) |
| ORM | SQLAlchemy |
| Scheduler | APScheduler |
| HTTP Client | httpx |
| Calibration | scikit-learn (isotonic regression) |
| Statistics | scipy (normal CDF) |
| Auth | cryptography (RSA-PSS for Kalshi API) |
| Dashboard | Flask + Chart.js |

---

## Domain Knowledge

A few non-obvious details that are critical for correct settlement:

- **"Above X" contracts use strict inequality**: 51°F settles YES for "above 50". Exactly 50 settles NO.
- **NWS CLI uses Local Standard Time year-round**: During DST, the civil clock shifts but the observation window does not. UTC boundaries remain fixed.
- **KNYC is Central Park, not an airport**: Unlike most stations, New York's settlement station is in Manhattan.
- **KMDW is Midway, not O'Hare**: Chicago uses the South Side airport station.
- **GFS ensemble has 31 members** (member00–member30), not 30.

---

## License

Private repository. All rights reserved.

---

*AetherBot v1.0 · March 2026 · Built with Python, weather models, and a healthy respect for atmospheric chaos.*

---
---

# Technical Whitepaper (Full HTML)

The complete interactive whitepaper with charts is served at `/whitepaper` on the dashboard. The raw HTML follows below.

<details>
<summary><strong>Click to expand: AetherBot Technical Overview (whitepaper.html)</strong></summary>

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
<title>AetherBot - Technical Overview</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>%E2%9A%A1</text></svg>">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}

  :root{
    --bg-canvas:#0d1117;
    --bg-card:#161b22;
    --bg-card-alt:#1c2128;
    --border:#30363d;
    --border-light:#484f58;
    --text-primary:#e6edf3;
    --text-secondary:#8b949e;
    --text-muted:#6e7681;
    --accent:#58a6ff;
    --green:#3fb950;
    --red:#f85149;
    --yellow:#d29922;
    --orange:#db6d28;
    --blue-badge:#1f6feb;
    --green-badge:#238636;
    --yellow-badge:#9e6a03;
    --radius:12px;
    --radius-sm:8px;
    --shadow:0 1px 3px rgba(0,0,0,.4);
  }

  html{font-size:16px;-webkit-text-size-adjust:100%}
  body{
    font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,sans-serif;
    background:var(--bg-canvas);
    color:var(--text-primary);
    line-height:1.6;
    min-height:100vh;
    overflow-x:hidden;
  }

  .container{
    max-width:900px;
    margin:0 auto;
    padding:12px;
  }
  @media(min-width:768px){
    .container{padding:24px 32px}
  }

  /* Header */
  .header{
    display:flex;
    align-items:center;
    justify-content:space-between;
    flex-wrap:wrap;
    gap:8px;
    padding:16px 0 12px;
    border-bottom:1px solid var(--border);
    margin-bottom:24px;
  }
  .header-left{display:flex;align-items:center;gap:10px}
  .header h1{font-size:1.4rem;font-weight:700;letter-spacing:-.02em}
  .header h1 span{font-size:1.3rem}
  .header-right{display:flex;align-items:center;gap:12px}
  .back-link{color:var(--accent);text-decoration:none;font-size:.8rem}
  .back-link:hover{text-decoration:underline}

  /* Sections */
  .section{margin-bottom:32px}
  .section-title{
    font-size:.85rem;font-weight:600;color:var(--text-secondary);
    text-transform:uppercase;letter-spacing:.06em;
    margin-bottom:12px;padding-bottom:6px;
    border-bottom:1px solid var(--border);
  }

  /* Cards */
  .card{
    background:var(--bg-card);border:1px solid var(--border);
    border-radius:var(--radius);padding:16px;margin-bottom:16px;
    box-shadow:var(--shadow);overflow:hidden;
  }

  /* Abstract */
  .abstract{
    font-size:.9rem;color:var(--text-secondary);
    line-height:1.7;padding:12px 0;
  }

  /* Prose */
  .prose{font-size:.85rem;color:var(--text-primary);line-height:1.7}
  .prose p{margin-bottom:12px}
  .prose strong{color:var(--accent)}

  /* Tables */
  .table-wrap{
    overflow-x:auto;-webkit-overflow-scrolling:touch;
    scrollbar-width:thin;scrollbar-color:var(--border) transparent;
  }
  table{width:100%;border-collapse:collapse;font-size:.8rem;white-space:nowrap}
  th{
    text-align:left;padding:8px 10px;font-weight:600;font-size:.7rem;
    text-transform:uppercase;letter-spacing:.04em;color:var(--text-secondary);
    background:var(--bg-card-alt);border-bottom:1px solid var(--border);
  }
  td{
    padding:7px 10px;color:var(--text-primary);
    border-bottom:1px solid var(--border);
  }
  tbody tr:nth-child(even){background:rgba(22,27,34,.6)}

  /* Chart containers */
  .chart-wrap{position:relative;width:100%;height:220px}
  @media(min-width:768px){.chart-wrap{height:280px}}

  /* Pipeline visual */
  .pipeline{
    display:flex;align-items:center;justify-content:center;
    gap:4px;flex-wrap:wrap;padding:16px 8px;
  }
  .pipe-step{
    background:var(--bg-card-alt);border:1px solid var(--border);
    border-radius:var(--radius-sm);padding:10px 14px;
    text-align:center;min-width:100px;
  }
  .pipe-step .step-label{font-size:.65rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.04em}
  .pipe-step .step-name{font-size:.8rem;font-weight:600;color:var(--text-primary);margin-top:2px}
  .pipe-arrow{color:var(--accent);font-size:1.2rem;font-weight:700;flex-shrink:0}

  /* Formula boxes */
  .formula{
    background:var(--bg-card-alt);border:1px solid var(--border);
    border-radius:var(--radius-sm);padding:12px 16px;
    font-family:'SF Mono',Monaco,Consolas,monospace;
    font-size:.8rem;color:var(--accent);
    overflow-x:auto;margin:8px 0 12px;
  }

  /* Badge helpers */
  .badge{
    display:inline-block;font-size:.65rem;font-weight:600;
    padding:2px 8px;border-radius:20px;text-transform:uppercase;letter-spacing:.03em;
  }
  .badge-green{background:var(--green-badge);color:#fff}
  .badge-yellow{background:var(--yellow-badge);color:#fff}
  .badge-blue{background:var(--blue-badge);color:#fff}

  .text-green{color:var(--green)}
  .text-yellow{color:var(--yellow)}
  .text-accent{color:var(--accent)}
  .text-muted{color:var(--text-muted)}

  /* Architecture diagram */
  .arch-diagram{
    display:flex;flex-direction:column;align-items:center;gap:0;padding:16px 0;
  }
  .arch-layer{
    background:var(--bg-card-alt);border:1px solid var(--border);
    border-radius:var(--radius-sm);padding:10px 20px;
    text-align:center;width:280px;max-width:100%;
  }
  .arch-layer .layer-title{font-size:.75rem;font-weight:600;color:var(--accent)}
  .arch-layer .layer-desc{font-size:.7rem;color:var(--text-secondary);margin-top:2px}
  .arch-arrow{color:var(--border-light);font-size:1.2rem;line-height:1}

  /* Footer */
  .footer{
    text-align:center;padding:24px 0 16px;
    color:var(--text-muted);font-size:.7rem;
    border-top:1px solid var(--border);margin-top:12px;
  }

  /* Sizing algo steps */
  .algo-steps{list-style:none;padding:0}
  .algo-steps li{
    padding:6px 0 6px 24px;position:relative;
    font-size:.8rem;color:var(--text-primary);
  }
  .algo-steps li::before{
    content:attr(data-step);position:absolute;left:0;
    color:var(--accent);font-weight:700;font-size:.75rem;
  }

  /* Grid for side-by-side charts */
  .chart-grid{display:grid;gap:16px}
  @media(min-width:768px){.chart-grid{grid-template-columns:1fr 1fr}}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <header class="header">
    <div class="header-left">
      <h1><span>&#9889;</span> AetherBot Technical Overview</h1>
    </div>
    <div class="header-right">
      <a href="/" class="back-link">&larr; Dashboard</a>
    </div>
  </header>

  <!-- Abstract -->
  <div class="section">
    <div class="abstract">
      AetherBot is an automated trading system for Kalshi's KXHIGH daily high-temperature prediction markets.
      It ingests multi-model weather forecasts, constructs calibrated probability distributions, detects statistical
      edges against market-implied probabilities, and executes trades using fractional Kelly criterion position sizing
      with layered risk controls. The system covers 5 U.S. cities and runs on free weather APIs.
    </div>
  </div>

  <!-- Architecture -->
  <div class="section">
    <div class="section-title">System Architecture</div>
    <div class="card">
      <div class="arch-diagram">
        <div class="arch-layer">
          <div class="layer-title">Data Sources</div>
          <div class="layer-desc">GFS Ensemble &middot; HRRR &middot; NWS &middot; IEM CLI</div>
        </div>
        <div class="arch-arrow">&#8595;</div>
        <div class="arch-layer">
          <div class="layer-title">Data Pipeline</div>
          <div class="layer-desc">Fetch &middot; Parse &middot; Store &middot; Filter</div>
        </div>
        <div class="arch-arrow">&#8595;</div>
        <div class="arch-layer">
          <div class="layer-title">Signal Engine</div>
          <div class="layer-desc">Ensemble Prob &middot; HRRR Correct &middot; Blend &middot; Calibrate</div>
        </div>
        <div class="arch-arrow">&#8595;</div>
        <div class="arch-layer">
          <div class="layer-title">Strategy Layer</div>
          <div class="layer-desc">Edge Detect &middot; Kelly Size &middot; Risk Manage</div>
        </div>
        <div class="arch-arrow">&#8595;</div>
        <div class="arch-layer">
          <div class="layer-title">Execution</div>
          <div class="layer-desc">Paper Trader &middot; Live Trader &middot; Settlement</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Cities -->
  <div class="section">
    <div class="section-title">Market Coverage</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>City</th><th>Station</th><th>Series Ticker</th><th>Timezone</th></tr></thead>
          <tbody>
            <tr><td>New York</td><td>KNYC (Central Park)</td><td class="text-accent">KXHIGHNY</td><td>America/New_York</td></tr>
            <tr><td>Chicago</td><td>KMDW (Midway)</td><td class="text-accent">KXHIGHCHI</td><td>America/Chicago</td></tr>
            <tr><td>Miami</td><td>KMIA</td><td class="text-accent">KXHIGHMIA</td><td>America/New_York</td></tr>
            <tr><td>Los Angeles</td><td>KLAX</td><td class="text-accent">KXHIGHLAX</td><td>America/Los_Angeles</td></tr>
            <tr><td>Denver</td><td>KDEN</td><td class="text-accent">KXHIGHDEN</td><td>America/Denver</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Data Sources -->
  <div class="section">
    <div class="section-title">Data Sources</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Source</th><th>Model</th><th>Frequency</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td>Open-Meteo Ensemble</td><td>GFS 0.25&deg; (31 members)</td><td>Every 60 min</td><td>Primary probability estimate</td></tr>
            <tr><td>Open-Meteo Forecast</td><td>HRRR 3km</td><td>Every 30 min</td><td>Short-range correction</td></tr>
            <tr><td>NWS API</td><td>Gridded forecast</td><td>Every 120 min</td><td>Human-machine blend</td></tr>
            <tr><td>IEM CLI</td><td>NWS Climate Reports</td><td>Daily 11:15 AM ET</td><td>Settlement (ground truth)</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Job Schedule -->
  <div class="section">
    <div class="section-title">Scheduler Jobs</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Job</th><th>Trigger</th><th>Frequency</th></tr></thead>
          <tbody>
            <tr><td>Market Discovery</td><td>Cron</td><td>6 AM, 12 PM, 6 PM ET</td></tr>
            <tr><td>GFS Ensemble Fetch</td><td>Interval</td><td>Every 60 minutes</td></tr>
            <tr><td>HRRR Fetch</td><td>Interval</td><td>Every 30 minutes</td></tr>
            <tr><td>NWS Forecast Fetch</td><td>Interval</td><td>Every 120 minutes</td></tr>
            <tr><td class="text-accent">Scan &amp; Trade</td><td>Interval</td><td>Every 5 minutes</td></tr>
            <tr><td>Settlement Check</td><td>Cron</td><td>11:15 AM ET daily</td></tr>
            <tr><td>Daily Report</td><td>Cron</td><td>12:00 PM ET daily</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Signal Pipeline -->
  <div class="section">
    <div class="section-title">Signal Pipeline</div>
    <div class="card">
      <div class="pipeline">
        <div class="pipe-step">
          <div class="step-label">Step 1</div>
          <div class="step-name">Ensemble Prob</div>
        </div>
        <div class="pipe-arrow">&rarr;</div>
        <div class="pipe-step">
          <div class="step-label">Step 2</div>
          <div class="step-name">HRRR Correct</div>
        </div>
        <div class="pipe-arrow">&rarr;</div>
        <div class="pipe-step">
          <div class="step-label">Step 3</div>
          <div class="step-name">Model Blend</div>
        </div>
        <div class="pipe-arrow">&rarr;</div>
        <div class="pipe-step">
          <div class="step-label">Step 4</div>
          <div class="step-name">Calibrate</div>
        </div>
        <div class="pipe-arrow">&rarr;</div>
        <div class="pipe-step">
          <div class="step-label">Step 5</div>
          <div class="step-name">Edge Detect</div>
        </div>
      </div>
      <div class="prose">
        <p><strong>Ensemble Probability:</strong> Count the fraction of 31 GFS members predicting the outcome. For "above X" contracts: P(high &gt; X) = members exceeding X / total members. Each member's max is rounded to integer &deg;F to match NWS CLI convention.</p>
        <p><strong>HRRR Correction:</strong> Shift ensemble members by a lead-time-weighted fraction of the HRRR-ensemble disagreement. HRRR's 3km resolution is most valuable at short lead times (&lt;12h).</p>
        <p><strong>Model Blend:</strong> Weighted average of ensemble (60%), HRRR (25%), and NWS (15%) probabilities. NWS/HRRR deterministic forecasts are converted to probabilities via Gaussian error model (&sigma; = 3&deg;F).</p>
        <p><strong>Isotonic Calibration:</strong> Non-parametric monotone mapping from raw probabilities to calibrated probabilities using scikit-learn's IsotonicRegression, fit on historical forecast-outcome pairs (walk-forward, refit every 50 trades).</p>
      </div>
    </div>
  </div>

  <!-- Model Weights -->
  <div class="section">
    <div class="section-title">Model Weights</div>
    <div class="chart-grid">
      <div class="card">
        <div class="table-wrap">
          <table>
            <thead><tr><th>Source</th><th>Weight</th><th>Role</th></tr></thead>
            <tbody>
              <tr><td>GFS Ensemble</td><td class="text-green">60%</td><td>Primary (uncertainty quantification)</td></tr>
              <tr><td>HRRR</td><td class="text-yellow">25%</td><td>Short-range correction</td></tr>
              <tr><td>NWS</td><td class="text-accent">15%</td><td>Human-machine judgment</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div style="font-size:.75rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px">HRRR Weight by Lead Time</div>
        <div class="chart-wrap" style="height:200px">
          <canvas id="hrrr-chart"></canvas>
        </div>
      </div>
    </div>
  </div>

  <!-- Edge Detection -->
  <div class="section">
    <div class="section-title">Edge Detection &amp; Entry Criteria</div>
    <div class="card">
      <div class="prose">
        <p>An <strong>edge</strong> is the difference between our calibrated probability and the market price: edge = P<sub>model</sub> - P<sub>market</sub>. Positive edge on YES means the market underprices the event.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Filter</th><th>Criterion</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td>Lead Time</td><td>&le; 72 hours</td><td>Forecast skill degrades beyond 3 days</td></tr>
            <tr><td>Price Range</td><td>YES price in [0.08, 0.92]</td><td>Avoid illiquid tail contracts</td></tr>
            <tr><td>Min Edge</td><td>|edge| &ge; 8%</td><td>Overcome transaction costs</td></tr>
            <tr><td>Min Edge (HRRR confirm)</td><td>|edge| &ge; 6%</td><td>Lower bar when HRRR agrees</td></tr>
            <tr><td class="text-green">Min Win Prob (Pwin)</td><td class="text-green">&ge; 55%</td><td class="text-green">Model must be confident on traded side</td></tr>
            <tr><td>Expected Value</td><td>EV &gt; 0</td><td>Positive expectation required</td></tr>
          </tbody>
        </table>
      </div>
      <div style="margin-top:12px">
        <div style="font-size:.75rem;font-weight:600;color:var(--text-secondary);margin-bottom:6px">CONFIDENCE TIERS</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap">
          <span class="badge badge-green">HIGH: &gt;20% edge</span>
          <span class="badge badge-yellow">MEDIUM: 12-20% edge</span>
          <span class="badge badge-blue">LOW: 8-12% edge</span>
        </div>
      </div>
    </div>
  </div>

  <!-- Kelly Sizing -->
  <div class="section">
    <div class="section-title">Kelly Criterion Position Sizing</div>
    <div class="card">
      <div class="prose">
        <p>The Kelly criterion maximizes long-run growth by finding the optimal bet fraction. AetherBot uses <strong>15% fractional Kelly</strong> to dramatically reduce drawdown risk while preserving most of the expected growth.</p>
      </div>
      <div class="formula">f* = (p &times; b - q) / b &nbsp;&nbsp;where b = (1-price)/price, p = Pwin, q = 1-p</div>
      <div class="prose">
        <p><strong>Key property:</strong> Kelly naturally sizes proportionally to both edge and win probability. Higher Pwin and larger edges produce larger Kelly fractions, meaning bigger bets. The Pwin floor (55%) prevents Kelly from over-sizing on low-confidence signals.</p>
      </div>
      <ol class="algo-steps">
        <li data-step="1.">Compute full Kelly fraction f*</li>
        <li data-step="2.">Apply fractional Kelly: f<sub>sized</sub> = f* &times; 0.15</li>
        <li data-step="3.">Dollar amount = f<sub>sized</sub> &times; bankroll</li>
        <li data-step="4.">Cap at 10% of bankroll or $1,000 (whichever is less)</li>
        <li data-step="5.">Contracts = floor(dollar_amount / price_per_contract)</li>
        <li data-step="6.">Minimum 1 contract if Kelly &gt; 0 (with dollar guard)</li>
      </ol>
    </div>
  </div>

  <!-- Risk Controls -->
  <div class="section">
    <div class="section-title">Risk Management</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Control</th><th>Limit</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td>Daily Loss Limit</td><td style="color:var(--red)">-$300</td><td>Circuit breaker on losing days</td></tr>
            <tr><td>Max Concurrent Positions</td><td>20</td><td>Overall portfolio concentration</td></tr>
            <tr><td>Max Positions per City</td><td>6</td><td>Geographic concentration</td></tr>
            <tr><td>Max Positions per Date</td><td>4</td><td>Temporal concentration</td></tr>
            <tr><td>Bankroll Check</td><td>cost &lt; bankroll</td><td>Solvency check</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Settlement -->
  <div class="section">
    <div class="section-title">Settlement &amp; PnL</div>
    <div class="card">
      <div class="prose">
        <p>Settlement runs daily at <strong>11:15 AM ET</strong> using NWS CLI reports from IEM. "Above X" contracts use <strong>strict inequality</strong> (51&deg;F settles YES for "above 50"). The observation window is Local Standard Time year-round.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Outcome</th><th>PnL Formula</th></tr></thead>
          <tbody>
            <tr><td class="text-green">YES Winner</td><td>+(1 - price) &times; contracts</td></tr>
            <tr><td style="color:var(--red)">YES Loser</td><td>-price &times; contracts</td></tr>
            <tr><td class="text-green">NO Winner</td><td>+price &times; contracts</td></tr>
            <tr><td style="color:var(--red)">NO Loser</td><td>-(1 - price) &times; contracts</td></tr>
          </tbody>
        </table>
      </div>
      <div class="prose" style="margin-top:12px">
        <p><strong>Brier Score:</strong> BS = (forecast - outcome)&sup2; averaged across all trades. Target: &lt; 0.20. Lower is better; 0 = perfect, 0.25 = coin-flip baseline.</p>
      </div>
    </div>
  </div>

  <!-- Backtest Results -->
  <div class="section">
    <div class="section-title">Backtest Results</div>
    <div class="chart-grid">
      <div class="card">
        <div class="table-wrap">
          <table>
            <thead><tr><th>Metric</th><th>Value</th><th>Target</th></tr></thead>
            <tbody>
              <tr><td>Total Trades</td><td>1,827</td><td>--</td></tr>
              <tr><td>Win Rate</td><td class="text-green">71.8%</td><td>&gt; 50%</td></tr>
              <tr><td>Gross PnL</td><td class="text-green">+$6,664</td><td>--</td></tr>
              <tr><td>Sharpe Ratio</td><td class="text-green">15.25</td><td>&gt; 1.0</td></tr>
              <tr><td>Brier Score</td><td class="text-green">0.1235</td><td>&lt; 0.20</td></tr>
              <tr><td>Final Bankroll</td><td>$16,664</td><td>--</td></tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="card">
        <div style="font-size:.75rem;font-weight:600;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.04em;margin-bottom:8px">Performance vs Targets</div>
        <div class="chart-wrap" style="height:220px">
          <canvas id="perf-chart"></canvas>
        </div>
      </div>
    </div>
    <div class="card" style="margin-top:4px">
      <div class="prose">
        <p><strong>Note:</strong> The backtest uses synthetic markets priced at climatological probabilities. The Sharpe ratio of 15.25 is inflated because synthetic markets are less efficient than real Kalshi markets. Live performance will be lower but the Brier score of 0.1235 indicates genuine forecast skill.</p>
      </div>
    </div>
  </div>

  <!-- Observation Windows -->
  <div class="section">
    <div class="section-title">Observation Windows (UTC)</div>
    <div class="card">
      <div class="prose">
        <p>NWS CLI uses <strong>Local Standard Time year-round</strong>. UTC boundaries do not shift with DST.</p>
      </div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>City</th><th>Std Offset</th><th>Window Start (UTC)</th><th>Window End (UTC)</th></tr></thead>
          <tbody>
            <tr><td>NYC</td><td>UTC-5</td><td>06:00</td><td>06:00 +1 day</td></tr>
            <tr><td>Chicago</td><td>UTC-6</td><td>07:00</td><td>07:00 +1 day</td></tr>
            <tr><td>Miami</td><td>UTC-5</td><td>06:00</td><td>06:00 +1 day</td></tr>
            <tr><td>LA</td><td>UTC-8</td><td>09:00</td><td>09:00 +1 day</td></tr>
            <tr><td>Denver</td><td>UTC-7</td><td>08:00</td><td>08:00 +1 day</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Technology Stack -->
  <div class="section">
    <div class="section-title">Technology Stack</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Component</th><th>Technology</th><th>Notes</th></tr></thead>
          <tbody>
            <tr><td>Language</td><td>Python 3.11+</td><td>~7,500 LOC across 55+ files</td></tr>
            <tr><td>Database</td><td>SQLite (WAL mode)</td><td>11 tables, concurrent reads</td></tr>
            <tr><td>ORM</td><td>SQLAlchemy</td><td>Declarative models</td></tr>
            <tr><td>Scheduler</td><td>APScheduler</td><td>Blocking + Interval/Cron triggers</td></tr>
            <tr><td>HTTP Client</td><td>httpx</td><td>Async-capable, sync usage</td></tr>
            <tr><td>Calibration</td><td>scikit-learn</td><td>Isotonic regression</td></tr>
            <tr><td>Statistics</td><td>scipy</td><td>Normal CDF for prob conversion</td></tr>
            <tr><td>Auth</td><td>cryptography</td><td>RSA-PSS for Kalshi API</td></tr>
            <tr><td>Dashboard</td><td>Flask + Chart.js</td><td>Mobile-responsive dark theme</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Database Schema -->
  <div class="section">
    <div class="section-title">Database Schema</div>
    <div class="card">
      <div class="table-wrap">
        <table>
          <thead><tr><th>Table</th><th>Purpose</th><th>Volume</th></tr></thead>
          <tbody>
            <tr><td class="text-accent">ensemble_forecasts</td><td>GFS member daily maxes</td><td>~310/city/run</td></tr>
            <tr><td class="text-accent">hrrr_forecasts</td><td>HRRR deterministic max</td><td>~2/city/run</td></tr>
            <tr><td class="text-accent">nws_forecasts</td><td>NWS 7-day high/low</td><td>~7/city/fetch</td></tr>
            <tr><td class="text-accent">observations</td><td>CLI ground truth</td><td>1/city/day</td></tr>
            <tr><td class="text-accent">kalshi_markets</td><td>Active contracts</td><td>~12/city (3x/day)</td></tr>
            <tr><td class="text-accent">signals</td><td>Model probs + edges</td><td>1/market/scan</td></tr>
            <tr><td class="text-accent">trades</td><td>Paper and live trades</td><td>Variable</td></tr>
            <tr><td class="text-accent">brier_scores</td><td>Forecast evaluation</td><td>1/settled trade</td></tr>
            <tr><td class="text-accent">strategies</td><td>Named strategy configs</td><td>User-defined</td></tr>
            <tr><td class="text-accent">backtest_runs</td><td>Backtest result history</td><td>Per strategy</td></tr>
            <tr><td class="text-accent">optimization_runs</td><td>Grid search results</td><td>Per sweep</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <!-- Footer -->
  <div class="footer">
    AetherBot v1.0 &middot; March 2026 &middot; <a href="/" class="back-link">Back to Dashboard</a>
  </div>

</div>

<script>
// HRRR Weight by Lead Time Chart
(function() {
  var ctx = document.getElementById('hrrr-chart').getContext('2d');
  new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['0h', '6h', '12h', '24h', '48h'],
      datasets: [{
        label: 'HRRR Weight',
        data: [0.45, 0.35, 0.25, 0.15, 0.05],
        borderColor: '#d29922',
        backgroundColor: 'rgba(210,153,34,0.15)',
        borderWidth: 2,
        pointRadius: 5,
        pointBackgroundColor: '#d29922',
        pointBorderColor: '#0d1117',
        pointBorderWidth: 2,
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#1c2128',
          titleColor: '#e6edf3',
          bodyColor: '#e6edf3',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            label: function(ctx) { return 'Weight: ' + (ctx.parsed.y * 100).toFixed(0) + '%'; }
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false },
          ticks: { color: '#8b949e', font: { size: 10 } },
          title: { display: true, text: 'Lead Time', color: '#6e7681', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false },
          ticks: {
            color: '#8b949e',
            font: { size: 10 },
            callback: function(v) { return (v * 100).toFixed(0) + '%'; }
          },
          min: 0,
          max: 0.5
        }
      }
    }
  });
})();

// Performance vs Targets Chart
(function() {
  var ctx = document.getElementById('perf-chart').getContext('2d');
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Win Rate', 'Brier Score', 'Profit Factor'],
      datasets: [
        {
          label: 'Actual',
          data: [71.8, 87.65, 166.4],
          backgroundColor: 'rgba(63,185,80,0.7)',
          borderColor: '#3fb950',
          borderWidth: 1,
          borderRadius: 4
        },
        {
          label: 'Target',
          data: [50, 80, 150],
          backgroundColor: 'rgba(88,166,255,0.3)',
          borderColor: '#58a6ff',
          borderWidth: 1,
          borderRadius: 4
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: true,
          labels: { color: '#8b949e', font: { size: 10 }, boxWidth: 12 }
        },
        tooltip: {
          backgroundColor: '#1c2128',
          titleColor: '#e6edf3',
          bodyColor: '#e6edf3',
          borderColor: '#30363d',
          borderWidth: 1,
          callbacks: {
            label: function(ctx) {
              var label = ctx.dataset.label;
              var v = ctx.parsed.y;
              var idx = ctx.dataIndex;
              if (idx === 0) return label + ': ' + v.toFixed(1) + '%';
              if (idx === 1) return label + ': ' + (100 - v).toFixed(2) + ' (inverted for display)';
              return label + ': ' + (v / 100).toFixed(2) + 'x';
            }
          }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: '#8b949e', font: { size: 10 } }
        },
        y: {
          grid: { color: 'rgba(48,54,61,0.5)', drawBorder: false },
          ticks: { color: '#8b949e', font: { size: 10 } },
          title: { display: true, text: 'Normalized Score', color: '#6e7681', font: { size: 10 } }
        }
      }
    }
  });
})();
</script>
</body>
</html>
```

</details>
