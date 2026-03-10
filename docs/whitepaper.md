# AetherBot: A Quantitative Framework for Weather Prediction Markets

**Version 1.3 | March 2026**

---

## Abstract

AetherBot is an automated trading system for Kalshi's KXHIGH daily high-temperature prediction markets. The system ingests forecasts from four independent global ensemble models (143 total members), a high-resolution regional model, and human-machine guidance to construct calibrated probability distributions. It detects statistical edges against market-implied probabilities and executes trades using fractional Kelly criterion position sizing with layered risk controls. This whitepaper describes the system architecture, multi-model data pipeline, signal generation methodology, adaptive weight modulation, execution framework, and empirical performance characteristics.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Market Structure](#2-market-structure)
3. [System Architecture](#3-system-architecture)
4. [Data Pipeline](#4-data-pipeline)
5. [Signal Engine](#5-signal-engine)
6. [Strategy Layer](#6-strategy-layer)
7. [Execution & Settlement](#7-execution--settlement)
8. [Backtesting Framework](#8-backtesting-framework)
9. [Monitoring & Evaluation](#9-monitoring--evaluation)
10. [Empirical Results](#10-empirical-results)
11. [Discussion & Future Work](#11-discussion--future-work)

---

## 1. Introduction

### 1.1 Motivation

Weather prediction markets represent a class of event contracts where the underlying outcome (daily maximum temperature at a specific station) is determined by a transparent, verifiable physical measurement. Unlike financial markets, where price discovery reflects consensus expectations about complex economic systems, weather markets have outcomes governed by atmospheric physics. This creates a distinct opportunity: weather forecasting models, trained on decades of observational data and physical first principles, can generate probability estimates that are systematically more accurate than the crowd-implied probabilities embedded in market prices.

AetherBot exploits this information asymmetry. The system constructs a multi-model ensemble probability distribution for each city's daily high temperature, calibrates it against historical observations, and compares the calibrated probability to the market price. When the model detects a statistically significant edge, it sizes a position using Kelly criterion mathematics to maximize long-run capital growth while controlling for drawdown risk.

### 1.2 Design Philosophy

The system is built on several core principles:

- **Ensemble thinking**: No single weather model is consistently superior. AetherBot fuses 143 ensemble members across four independent global models (GFS, ECMWF IFS, ICON-EPS, GEM/GEPS), each built on distinct dynamical cores and data assimilation schemes. This multi-model approach captures forecast uncertainty from genuinely independent sources, treating inter-model disagreement as a feature rather than a nuisance.
- **Calibration over precision**: A well-calibrated probabilistic forecast (where events predicted at 70% occur 70% of the time) is more valuable for trading than a precise but overconfident deterministic forecast.
- **Conservative sizing**: The fractional Kelly approach (15% of full Kelly) sacrifices some expected growth rate for substantially reduced drawdown risk.
- **Free data only**: The entire data pipeline uses open APIs (Open-Meteo, NWS, IEM) with no subscription costs, keeping the system accessible and the cost basis at zero.

### 1.3 Naming

"Aether" references the classical fifth element, the quintessence theorized to fill the upper atmosphere. It seemed fitting for a system that reads the sky and bets on what it finds.

---

## 2. Market Structure

### 2.1 Kalshi KXHIGH Contracts

Kalshi offers daily high-temperature markets across five U.S. cities. Each market settles based on the **NWS Climate Report (CLI)** "Maximum Temperature" for that station-day, reported as an integer in degrees Fahrenheit.

| City | Station | Series Ticker | Timezone |
|------|---------|---------------|----------|
| New York | KNYC (Central Park) | KXHIGHNY | America/New_York |
| Chicago | KMDW (Midway) | KXHIGHCHI | America/Chicago |
| Miami | KMIA | KXHIGHMIA | America/New_York |
| Los Angeles | KLAX | KXHIGHLAX | America/Los_Angeles |
| Denver | KDEN | KXHIGHDEN | America/Denver |

Two critical details for settlement:

1. **"Above X" contracts use strict inequality**: An "above 50" contract settles YES only if the observed maximum is 51 or higher. An observation of exactly 50 settles NO.
2. **The observation window uses Local Standard Time year-round**: The NWS CLI "day" runs from 1:00 AM LST to 12:59 AM LST the following calendar day. During daylight saving time, the civil clock reads one hour ahead, but the observation window does not shift. This means that during DST, the effective UTC window changes by one hour relative to what a naive civil-time implementation would compute.

### 2.2 Contract Types

Each city-date combination typically offers:

- **Above contracts**: Binary YES/NO on whether the high exceeds a threshold (e.g., "NYC above 57 on March 7")
- **Bracket contracts**: Binary YES/NO on whether the high falls within a range (e.g., "NYC between 52.5 and 54.5 on March 7")

Prices are quoted in cents (1-99), where the YES price represents the market-implied probability that the event occurs.

---

## 3. System Architecture

### 3.1 Overview

AetherBot follows a five-layer pipeline architecture:

```
                        +-------------------------------------------+
                        |              Data Sources                  |
                        | GFS (31) | ECMWF (51) | ICON (40) | GEM (21) |
                        |           HRRR  |  NWS  |  IEM CLI         |
                        +--------------------+----------------------+
                                             |
                        +--------------------v----------------------+
                        |              Data Pipeline                 |
                        | Fetch | Parse | Store | Filter | Freshness |
                        +--------------------+----------------------+
                                             |
                        +--------------------v----------------------+
                        |              Signal Engine                 |
                        | Per-Model Ensemble Prob (x4)               |
                        | HRRR Correction | Model Blend              |
                        | Adaptive Weights | Calibration             |
                        +--------------------+----------------------+
                                             |
                        +--------------------v----------------------+
                        |              Strategy Layer                |
                        | Edge Detection | Kelly Sizing              |
                        | Risk Management                            |
                        +--------------------+----------------------+
                                             |
                        +--------------------v----------------------+
                        |              Execution                     |
                        | Paper Trader | Live Trader | Settlement    |
                        | Model Scorer | Postmortem                  |
                        +-------------------------------------------+
```

### 3.2 Technology Stack

- **Language**: Python 3.11+
- **Database**: SQLite with WAL mode (concurrent reads, single-writer)
- **ORM**: SQLAlchemy (18 tables)
- **Scheduler**: APScheduler (BlockingScheduler with Interval and Cron triggers)
- **HTTP**: httpx (async-capable, used synchronously)
- **Scientific**: scikit-learn (isotonic regression), scipy (normal CDF)
- **Auth**: RSA-PSS signing for Kalshi API (cryptography library)

### 3.3 Job Schedule

| Job | Trigger | Frequency | Notes |
|-----|---------|-----------|-------|
| Market Discovery | Cron | 6 AM, 12 PM, 6 PM ET | Discovers new KXHIGH contracts |
| Smart Data Fetch | Interval | Every 5 minutes | Freshness-aware; only fetches when new model runs are available |
| GFS Ensemble Fetch | Freshness | ~Every 6h (00Z/06Z/12Z/18Z) | 31 members, triggered by smart fetch |
| ECMWF IFS Ensemble Fetch | Freshness | ~Every 12h (00Z/12Z) | 51 members, ~6h data lag |
| ICON-EPS Ensemble Fetch | Freshness | ~Every 12h (00Z/12Z) | 40 members, ~6h data lag |
| GEM/GEPS Ensemble Fetch | Freshness | ~Every 12h (00Z/12Z) | 21 members, ~6h data lag |
| HRRR Fetch | Freshness | ~Every 1-2h | Hourly CONUS runs |
| NWS Forecast Fetch | Freshness | ~Every 2h | Staleness-based trigger |
| Scan & Trade | Interval | Every 5 minutes | DB-only reads; also triggered on new model arrival |
| Price Discovery Scan | Interval | Every 2 minutes | Fast-polls newly discovered markets |
| Settlement Check | Cron | 11:15 AM ET daily | Fetches CLI, settles trades |
| Model Scorecard | Post-settlement | After each settlement | Scores all models against outcome |
| Adaptive Weight Update | Post-settlement | After each settlement | Recomputes inverse-Brier weights |
| Daily Report | Cron | 12:00 PM ET daily | Email with charts and summary |

---

## 4. Data Pipeline

AetherBot ingests forecasts from four independent global ensemble models, totaling 143 members across distinct dynamical cores and data assimilation systems. This multi-model approach is analogous to triangulating a measurement with independent instruments: systematic biases in one model are unlikely to be correlated with biases in the others, making the aggregate estimate more robust than any single source. All ensemble data is sourced from the Open-Meteo Ensemble API.

### 4.1 GFS Ensemble (NCEP, USA)

The Global Forecast System (GFS) 0.25-degree ensemble, operated by NCEP, provides one of four global ensemble signals in AetherBot's probability framework. Each model run produces 31 members: one control run and 30 perturbed initial-condition runs. The perturbations are designed to sample the space of plausible atmospheric states given observational uncertainty, making the ensemble spread a physically-grounded measure of forecast uncertainty.

**Model**: `gfs_seamless` via Open-Meteo Ensemble API. Runs every 6 hours (00Z/06Z/12Z/18Z).

**Fetch parameters**: Hourly `temperature_2m` for all 31 members, up to 10 forecast days, in Fahrenheit.

**Processing pipeline**:

1. Parse hourly timestamps (UTC) and temperature arrays per member
2. For each target date, compute the observation window in UTC using the city's standard-time offset
3. Filter hourly data to only those timestamps within the observation window
4. Extract the daily maximum temperature per member from the filtered hours
5. Validate: require at least 20 of 31 members to have valid (non-NaN) data
6. Store as `EnsembleForecast` records with unique constraint on (city, model_run_time, target_date, member)

### 4.1b ECMWF IFS Ensemble (ECMWF, Europe)

The European Centre for Medium-Range Weather Forecasts (ECMWF) Integrated Forecasting System is widely regarded as the most skillful global weather model, particularly for medium-range forecasts (3-10 days). Its 51-member ensemble (1 control + 50 perturbed) provides the largest single-model ensemble in AetherBot's system.

**Model**: `ecmwf_ifs025` via Open-Meteo Ensemble API. Runs every 12 hours (00Z/12Z), with data typically available approximately 6 hours after initialization.

**Processing pipeline**: Identical to GFS: hourly temperature extraction, observation-window filtering, daily max computation per member. The minimum valid member threshold is 25 of 51 members.

The ECMWF IFS brings a fundamentally different dynamical core and 4D-Var data assimilation scheme to the blend, meaning its errors are largely independent of the GFS's ensemble Kalman filter approach. This independence is the primary source of multi-model blending's value.

### 4.1c ICON-EPS Ensemble (DWD, Germany)

The ICOsahedral Nonhydrostatic model (ICON), operated by the Deutscher Wetterdienst (DWD), uses an icosahedral grid rather than the latitude-longitude grids employed by GFS and ECMWF. This grid geometry eliminates the polar singularity problem and provides more uniform resolution globally. The ensemble prediction system (EPS) produces 40 members.

**Model**: `icon_seamless_eps` via Open-Meteo Ensemble API. Runs every 12 hours (00Z/12Z), with data typically available approximately 6 hours after initialization.

**Processing pipeline**: Identical to the other ensembles, with a minimum valid member threshold of 20 of 40 members. The icosahedral grid's distinct numerical characteristics provide genuinely independent error modes from the spectral models (GFS, ECMWF).

### 4.1d GEM/GEPS Ensemble (CMC, Canada)

The Global Environmental Multiscale (GEM) model, operated by the Canadian Meteorological Centre (CMC), rounds out the four-model ensemble system. Its Global Ensemble Prediction System (GEPS) produces 21 members, the smallest ensemble in the system but still sufficient for robust probability estimation.

**Model**: `gem_global` via Open-Meteo Ensemble API. Runs every 12 hours (00Z/12Z), with data typically available approximately 6 hours after initialization.

**Processing pipeline**: Identical structure, with a minimum valid member threshold of 10 of 21 members (reflecting the smaller ensemble size). The GEM's semi-Lagrangian dynamical core and distinct physics parameterizations add a fourth independent perspective on atmospheric evolution.

### 4.2 HRRR (Short-Range Correction)

The High-Resolution Rapid Refresh model operates at 3 km horizontal resolution over CONUS, compared to GFS's 25 km. This resolution advantage makes HRRR substantially more skillful at short lead times (0-18 hours), particularly for temperature forecasts influenced by local terrain, land-water boundaries, and convective processes.

AetherBot uses HRRR as a correction signal rather than a standalone probability source. The HRRR provides a single deterministic forecast, which lacks the uncertainty information that the GFS ensemble provides natively. However, the systematic bias between HRRR and the GFS ensemble mean contains valuable information about where the ensemble may be systematically wrong.

**Model**: `ncep_hrrr_conus` via Open-Meteo forecast API. Updated every 30 minutes.

### 4.3 NWS Forecasts (Tertiary Signal)

National Weather Service gridded forecasts are accessed via `api.weather.gov`. These represent human-machine forecasts: NWS meteorologists adjust automated guidance based on local knowledge, pattern recognition, and mesoscale analysis. NWS forecasts carry a 15% weight in the model blend, providing a human-judgment signal that complements the purely numerical ensemble.

### 4.4 IEM CLI Data (Settlement Source)

The Iowa Environmental Mesonet aggregates NWS Climate Reports (CLI) from first-order stations. This is the authoritative source for Kalshi settlement. The system fetches CLI data daily at 11:15 AM ET, after reports are typically published.

**Key fields**: `high_f` (integer Fahrenheit maximum), `low_f`, observation station, and date.

### 4.5 Database Schema

The database has grown from 8 to 18 tables as the system expanded to support multi-model ensembles, per-model scoring, strategy optimization, and continuous price tracking.

| Table | Purpose | Record Volume |
|-------|---------|---------------|
| `ensemble_forecasts` | GFS member daily maxes | ~310/city/run (31 members x 10 days) |
| `ecmwf_forecasts` | ECMWF IFS member daily maxes | ~153/city/run (51 members x 3 days) |
| `icon_eps_forecasts` | ICON-EPS member daily maxes | ~120/city/run (40 members x 3 days) |
| `gem_forecasts` | GEM/GEPS member daily maxes | ~63/city/run (21 members x 3 days) |
| `hrrr_forecasts` | HRRR deterministic max | ~2/city/run (2 forecast days) |
| `nws_forecasts` | NWS 7-day high/low | ~7/city/fetch |
| `observations` | CLI ground truth | 1/city/day |
| `hourly_observations` | Intraday NWS actuals | ~24/city/day |
| `kalshi_markets` | Active contracts | ~12/city (refreshed 3x/day) |
| `market_price_history` | Continuous price snapshots | ~100/market/day |
| `signals` | Model probabilities + edges (all 6 sources) | 1/market/scan |
| `trades` | Paper and live trades | Variable |
| `brier_scores` | Forecast evaluation | 1/settled trade |
| `daily_pnl` | Aggregated daily performance | 1/day |
| `strategies` | Named strategy configurations | User-defined |
| `backtest_runs` | Stored backtest results | Per optimization run |
| `model_scorecards` | Per-model Brier scoring vs. settlement | ~6 sources/market/settlement |
| `optimization_runs` | Parameter grid search results | Per optimization run |

---

## 5. Signal Engine

The signal engine transforms raw weather model output into calibrated probability estimates for each active market contract. This is the intellectual core of the system.

### 5.1 Ensemble Probability

Each of the four global ensemble models independently produces a probability estimate using the same member-counting methodology. This is computed identically for all models; only the member count $N$ and minimum validity thresholds differ.

**For "above X" contracts**:

$$P_k(\text{high} > X) = \frac{|\{m : T_{k,m} > X\}|}{N_k}$$

where $T_{k,m}$ is the rounded daily maximum for member $m$ of model $k$, and $N_k$ is the number of valid members for that model.

**For bracket contracts**:

$$P_k(L \leq \text{high} \leq H) = \frac{|\{m : L \leq T_{k,m} \leq H\}|}{N_k}$$

Critical implementation detail: each member's daily maximum is **rounded to the nearest integer** before comparison, matching the NWS CLI's integer reporting convention. Without this rounding step, the system would systematically misprice contracts near round-number thresholds.

**Minimum valid member thresholds** (below which the model returns 0.5, maximum uncertainty):

| Model | Total Members | Minimum Valid |
|-------|--------------|---------------|
| GFS | 31 | 20 |
| ECMWF IFS | 51 | 25 |
| ICON-EPS | 40 | 20 |
| GEM/GEPS | 21 | 10 |

These per-model probability estimates ($P_{\text{GFS}}$, $P_{\text{ECMWF}}$, $P_{\text{ICON}}$, $P_{\text{GEM}}$) feed into the model blending stage (Section 5.3), where they are combined with HRRR and NWS signals using lead-time-adaptive weights.

### 5.2 HRRR Correction

The HRRR correction adjusts the ensemble distribution based on the disagreement between the HRRR deterministic forecast and the ensemble mean:

$$\text{shift} = T_{\text{HRRR}} - \bar{T}_{\text{ensemble}}$$

$$T_m' = T_m + w(h) \cdot \text{shift}$$

where $w(h)$ is a lead-time-dependent weight function:

| Lead Time (hours) | HRRR Weight $w(h)$ |
|---|---|
| 0 | 0.45 |
| 6 | 0.35 |
| 12 | 0.25 |
| 24 | 0.15 |
| 48+ | 0.05 |

Values between breakpoints are linearly interpolated. The rationale: HRRR's resolution advantage degrades with lead time as synoptic-scale uncertainties dominate over mesoscale detail. By 48 hours, the HRRR offers minimal incremental information over the GFS ensemble.

This approach is analogous to a reference standard correction in analytical chemistry: the HRRR serves as a higher-fidelity "standard" that reveals systematic bias in the ensemble's lower-resolution forecast.

### 5.3 Model Blending

The model blender combines probability estimates from up to six sources: four global ensembles (GFS, ECMWF IFS, ICON-EPS, GEM/GEPS), plus HRRR and NWS deterministic forecasts. The blending architecture uses a hierarchical weight system with lead-time adaptation and performance-based modulation.

**Weight structure**:

The total probability weight is divided into three categories: a global ensemble share ($w_{\text{global}}$), an HRRR share ($w_{\text{HRRR}}$), and an NWS share ($w_{\text{NWS}}$). The global ensemble share is further subdivided among the four ensemble models proportionally to their relative quality and ensemble size:

| Model | Intra-Global Share | Rationale |
|-------|-------------------|-----------|
| GFS | 30% | 31 members, hourly resolution, best CONUS detail |
| ECMWF IFS | 30% | 51 members, highest overall global skill |
| ICON-EPS | 22% | 40 members, independent icosahedral dynamics |
| GEM/GEPS | 18% | 21 members, smallest ensemble but independent physics |

The HRRR weight follows a lead-time-dependent decay curve, reflecting that its 3 km resolution advantage diminishes as synoptic-scale uncertainty dominates at longer ranges:

| Lead Time (hours) | HRRR Weight |
|---|---|
| 0 | 0.45 |
| 6 | 0.35 |
| 12 | 0.25 |
| 24 | 0.15 |
| 48+ | 0.05 |

The remaining weight is allocated to NWS. As the HRRR claim shrinks at longer lead times, the global ensemble share and NWS share both grow, reflecting the ensembles' superior skill at medium range.

**Blended probability**:

$$P_{\text{blend}} = \frac{\sum_{k} w_k \cdot P_k}{\sum_{k} w_k}$$

where the sum runs over all available sources $k \in \{\text{GFS}, \text{ECMWF}, \text{ICON}, \text{GEM}, \text{HRRR}, \text{NWS}\}$.

**Missing model redistribution**: When a model is unavailable (e.g., ICON-EPS data has not yet arrived for a given run), its intra-global share is redistributed proportionally among the remaining available global ensembles. This ensures that the blend degrades gracefully rather than discontinuously when data sources are temporarily absent.

**Deterministic-to-probabilistic conversion**: HRRR and NWS produce single-valued forecasts rather than distributions. The system converts these to probabilities using a Gaussian error model:

$$P(\text{high} > X) = 1 - \Phi\left(\frac{X - T_{\text{forecast}}}{\sigma}\right)$$

where $\sigma = 3^\circ F$ for HRRR and $\sigma = 4^\circ F$ for NWS (reflecting the latter's larger typical error).

**Adaptive modulation**: The fixed weight curves described above are further modulated by a performance-based adaptive component (Section 5.5). The final weight for each source is a blend of 70% physics-based fixed curve and 30% empirical inverse-Brier weight, ensuring that models which are performing well in recent settlements receive a modest boost without allowing the system to chase noise.

Blended probabilities are clamped to [0.01, 0.99] to prevent degenerate edge calculations.

### 5.4 Isotonic Calibration

Raw model probabilities, even when blended, often exhibit systematic biases. Events predicted at 70% may actually occur 75% of the time (underconfidence) or 65% of the time (overconfidence). AetherBot applies **isotonic regression** to map raw probabilities to calibrated probabilities.

Isotonic regression is a non-parametric monotone function estimator. Given historical pairs $(f_i, o_i)$ where $f_i$ is the forecast probability and $o_i \in \{0, 1\}$ is the outcome, it finds the monotonically non-decreasing function $g$ that minimizes:

$$\sum_i (g(f_i) - o_i)^2$$

subject to $g(f_i) \leq g(f_j)$ whenever $f_i \leq f_j$.

**Implementation details**:

- Uses scikit-learn's `IsotonicRegression` with `y_min=0.01, y_max=0.99`
- Requires at least 100 historical forecast-outcome pairs before activation
- During backtesting, refits every 50 trades (walk-forward methodology)
- In production, refits daily as new settlement data becomes available

The calibration step is essential for Kelly criterion sizing. Kelly's formula assumes the input probability is the true probability of winning. If the probability is systematically biased, the Kelly bet will be systematically wrong, and a biased-high probability leads to oversizing and accelerated ruin.

### 5.5 Adaptive Weight Modulation

While the lead-time-dependent weight curves in Section 5.3 encode physical reasoning (HRRR is better at short range, ensembles at medium range), actual model performance varies over time due to seasonal skill differences, model update cycles, and regional biases. The adaptive weight modulation system uses empirical settlement data to reward models that are performing well and reduce reliance on models that are underperforming.

**Mechanism**: After each settlement, the `ModelScorecard` records the Brier score contribution $(P_k - o)^2$ for each model source $k$. The `AdaptiveWeightManager` queries the last 60 days of settlement data and computes the average Brier score per model. These scores are converted to inverse-Brier weights:

$$w_k^{\text{adaptive}} = \frac{1/\overline{BS}_k}{\sum_j 1/\overline{BS}_j}$$

Lower average Brier (better calibration) yields higher weight. A model with perfect recent performance ($\overline{BS} \to 0$) would receive the maximum share, while a poorly calibrated model would see its weight shrink.

**Smoothing**: The adaptive weight is blended with the fixed physics-based weight using a smoothing factor $\alpha = 0.3$:

$$w_k^{\text{final}} = \alpha \cdot w_k^{\text{adaptive}} + (1 - \alpha) \cdot w_k^{\text{fixed}}$$

This design constrains the adaptive component to at most a 30% perturbation of the fixed curves. The physics-based lead-time structure always dominates, preventing the system from overfitting to a short window of settlement data. This is analogous to a Bayesian prior: the fixed curves represent our prior belief about model skill, and the adaptive component is the likelihood update from observed performance.

**Minimum data requirements**: Adaptive weights only activate when at least two model sources have accumulated 10 or more scored markets within the rolling window. Before this threshold, the system operates purely on the fixed curves.

**Persistence**: Adaptive weights are serialized to JSON (`data/adaptive_weights.json`) and loaded on scheduler restart, ensuring continuity across system restarts without requiring a cold-start recalculation period.

---

## 6. Strategy Layer

### 6.1 Edge Detection

An "edge" is the difference between the model's calibrated probability and the market-implied probability:

$$\text{edge} = P_{\text{model}} - P_{\text{market}}$$

A positive edge on the YES side means the model believes YES is underpriced. A positive edge on the NO side means the model believes NO is underpriced ($P_{\text{model}} < P_{\text{market}}$, so $(1 - P_{\text{model}}) > (1 - P_{\text{market}})$).

**Signal generation filters**:

1. **Lead time**: Only trade contracts settling within 72 hours
2. **Price range**: YES price must be in [0.08, 0.92] (avoid illiquid tails)
3. **Minimum edge**: |edge| must exceed 8% (the `edge_threshold`)
4. **Positive expected value**: The trade's expected value per contract must be positive
5. **HRRR confirmation** (optional): If the HRRR probability independently confirms the trade direction, the edge threshold drops to 6%

**Confidence classification**:

| Edge Magnitude | Confidence |
|---|---|
| > 20% | High |
| 12% - 20% | Medium |
| 8% - 12% | Low |

Signals are sorted by |edge| descending, and the strategy processes them in order until risk limits are reached.

### 6.2 Kelly Criterion Position Sizing

The Kelly criterion maximizes the expected logarithm of wealth, producing the growth-optimal betting fraction. For a binary bet with win probability $p$, win payoff $b$-to-1, and loss of the stake:

$$f^* = \frac{pb - q}{b}$$

where $q = 1 - p$ and $b = \frac{1 - \text{price}}{\text{price}}$ (the odds implied by the contract price).

**Fractional Kelly**: Full Kelly produces maximum long-term growth but with severe drawdowns. AetherBot applies a fractional Kelly factor of 0.15 (15% of the optimal fraction), which:

- Reduces expected growth rate by a modest amount
- Reduces drawdown variance dramatically (drawdown scales as $f^2$, growth scales as $f$)
- Provides a substantial margin of safety against probability estimation errors

**Position sizing algorithm**:

```
1. Compute full Kelly fraction f*
2. Apply fractional Kelly: f_sized = f* x 0.15
3. Dollar amount = f_sized x bankroll
4. Apply caps (most restrictive wins):
   a. Percentage cap: 5% of bankroll
   b. Dollar cap: $100 absolute maximum
   c. Minimum: 1 contract if Kelly > 0
5. Guard: if min_contracts override exceeds dollar caps, size to 0
6. Contracts = floor(dollar_amount / price_per_contract)
```

### 6.3 Risk Management

A layered risk management system prevents catastrophic losses:

| Control | Limit | Purpose |
|---------|-------|---------|
| Daily loss limit | -$300 | Circuit breaker on losing days |
| Max concurrent positions | 20 | Overall portfolio concentration |
| Max positions per city | 6 | Geographic concentration |
| Max positions per date | 4 | Temporal concentration |
| Bankroll sufficiency | trade_cost < bankroll | Solvency check |

The risk manager evaluates **all** constraints before each trade. A trade that passes Kelly sizing but violates any risk limit is rejected. The system favors preservation of capital over capturing marginal edges.

---

## 7. Execution & Settlement

### 7.1 Paper Trading

In paper trading mode, the system simulates trade execution with these assumptions:

- Immediate fill at the signal price (no slippage model)
- Full contract count filled (no partial fills)
- Virtual bankroll initialized at $10,000
- All trade records stored identically to live trades (status = "filled")

Paper trading serves two purposes: validating the system before committing real capital, and generating a track record for calibration assessment.

### 7.2 Live Trading

Live execution uses the Kalshi API v2 with RSA-PSS authentication:

1. Sign the request: `message = timestamp_ms + METHOD + path`
2. Generate RSA-PSS-SHA256 signature with the private key
3. Include signature in `KALSHI-ACCESS-SIGNATURE` header
4. Submit limit order at the model's target price

### 7.3 Settlement

The settlement checker runs daily at 11:15 AM ET (after NWS CLI reports are typically published):

1. Query all open trades grouped by city and target date
2. Fetch the CLI observation from IEM for each (city, date) pair
3. Determine settlement outcome:
   - Above contracts: `observed_high > threshold`
   - Bracket contracts: `bracket_low <= observed_high <= bracket_high`
4. Compute PnL per trade:
   - YES winner: `pnl = (1 - price) x contracts`
   - YES loser: `pnl = -price x contracts`
   - NO winner: `pnl = price x contracts`
   - NO loser: `pnl = -(1 - price) x contracts`
5. Record Brier score contribution: $(P_{\text{forecast}} - \text{outcome})^2$
6. Update bankroll and trade status

---

## 8. Backtesting Framework

### 8.1 Walk-Forward Replay Engine

The backtesting engine replays historical data with walk-forward methodology to prevent look-ahead bias in calibration:

```
For each date in [start_date, end_date]:
  For each city:
    1. Retrieve the actual observed high from the observations table
    2. Generate synthetic market contracts around climatological mean
    3. Simulate ensemble forecast using Gaussian noise model
    4. Apply HRRR correction (simulated)
    5. Calibrate using isotonic regression (walk-forward: fit on past data only)
    6. Detect edges against synthetic market prices
    7. Size positions using Kelly criterion
    8. Settle trades using actual observed high
    9. Record trade, update bankroll, accumulate calibration data
```

### 8.2 Synthetic Markets

Because historical Kalshi order book data is not available for backtesting, the system generates synthetic markets:

1. Query historical observations for the target month across all available years
2. Compute the climatological mean high temperature
3. Generate bracket thresholds: [mean - 6, mean - 4, mean - 2, mean, mean + 2, mean + 4] degrees
4. Set market prices equal to climatological probabilities (fraction of historical days above each threshold)
5. Clamp prices to [0.05, 0.95]

This creates markets that are "efficient" in the climatological sense, meaning any edge the model detects comes from short-range forecast skill rather than climatological mispricing.

### 8.3 Walk-Forward Calibration

The isotonic calibrator is refit every 50 trades using only data available up to that point. This mirrors how the production system would operate: accumulating forecast-outcome pairs and periodically refitting. The first 100 trades use uncalibrated probabilities (insufficient data for a reliable fit).

---

## 9. Monitoring & Evaluation

### 9.1 Brier Score

The Brier score is the primary forecast evaluation metric:

$$BS = \frac{1}{N} \sum_{i=1}^{N} (f_i - o_i)^2$$

where $f_i$ is the forecast probability and $o_i \in \{0, 1\}$ is the outcome. Lower is better; 0 is perfect, 0.25 is climatological baseline for a 50/50 event.

**Brier decomposition**:

$$BS = \underbrace{\text{Reliability}}_{\text{calibration error}} - \underbrace{\text{Resolution}}_{\text{discrimination skill}} + \underbrace{\text{Uncertainty}}_{\text{base rate variance}}$$

- **Reliability** (lower is better): Measures calibration. A system where 70% forecasts verify 70% of the time has zero reliability term.
- **Resolution** (higher is better): Measures the system's ability to distinguish between events and non-events. High resolution means the system assigns high probabilities to events that occur and low probabilities to events that don't.
- **Uncertainty** (fixed): Determined by the base rate of the event, outside the forecaster's control.

### 9.2 Performance Metrics

| Metric | Formula | Target |
|--------|---------|--------|
| Brier Score | Mean squared probability error | < 0.20 |
| Win Rate | Wins / Total Trades | > 50% |
| Sharpe Ratio | (Mean PnL / Std PnL) x sqrt(250) | > 1.0 |
| Max Drawdown | Max peak-to-trough decline | < 15% of bankroll |
| Profit Factor | Gross Wins / Gross Losses | > 1.5 |

### 9.3 Monitoring Alerts

The system tracks calibration health through rolling Brier score windows:

- **7-day Brier** vs. **30-day Brier**: If the 7-day score exceeds the 30-day baseline by more than 3%, an alert is triggered indicating potential calibration degradation
- **Per-city Brier scores**: Identifies if a specific city's forecasts are underperforming, which could indicate local model bias
- **Win rate monitoring**: Alert if 30-day win rate drops below 40%

### 9.4 Per-Model Scoring & Postmortem Analysis

The `ModelScorecard` system (`model_scorer.py`) evaluates each model source independently after every settlement. For each settled market, the scorer records:

- **Per-model Brier contribution**: $(P_k - o)^2$ for each source $k \in \{\text{GFS}, \text{ECMWF}, \text{ICON-EPS}, \text{GEM}, \text{HRRR}, \text{NWS}, \text{blended}, \text{calibrated}, \text{market}\}$
- **Convergence trajectory**: How each model's probability evolved over time (first estimate, 24h, 12h, 6h lead)
- **Best/worst model flags**: Which model was closest to and farthest from the settlement outcome
- **Maximum probability swing**: Largest consecutive change in a model's probability estimate, indicating forecast instability

This data feeds three downstream systems:

1. **Adaptive weight modulation** (Section 5.5): Rolling 60-day Brier scores adjust blend weights toward better-performing models
2. **Evolution dashboards**: Per-market charts showing how each model's probability converged (or diverged) as the forecast horizon shortened
3. **Postmortem analysis**: Automated identification of markets where the model disagreed most with the outcome, highlighting systematic failure modes for investigation

### 9.5 Model Evolution & Convergence Dashboards

The monitoring dashboard includes dedicated views for multi-model analysis:

- **Model evolution charts**: For each active market, a time-series plot showing all six model probabilities (GFS, ECMWF, ICON-EPS, GEM, HRRR, NWS) converging toward the market price as the forecast horizon shrinks. These charts are emailed automatically when new GFS data arrives.
- **Inter-model spread**: Visualization of ensemble agreement/disagreement across models. Large inter-model spread signals genuine forecast uncertainty, while tight convergence across independent models signals high confidence.
- **Model arrival email**: When a new GFS run is ingested, the system renders evolution charts for all active markets and emails them to the operator, providing a visual snapshot of the current model landscape without requiring dashboard access.

---

## 10. Empirical Results

### 10.1 Backtest Performance

A walk-forward backtest over 2 years of historical data (5 cities, synthetic markets) produced the following results:

| Metric | Value |
|--------|-------|
| Total Trades | 1,827 |
| Win Rate | 71.8% |
| Gross PnL | +$6,664.36 |
| Sharpe Ratio | 15.25 |
| Brier Score | 0.1235 |
| Starting Bankroll | $10,000 |
| Final Bankroll | $16,664.36 |

The Brier score of 0.1235 is well below the 0.20 target, indicating strong calibration. The Sharpe ratio of 15.25 is extraordinarily high due to the synthetic market construction (markets are priced at climatological probabilities, so any short-range forecast skill generates consistent edges). Live market performance will be lower due to more efficient pricing by other market participants.

### 10.2 Live Paper Trading

Initial paper trading session (March 6, 2026):

- 14 paper trades executed across all 5 cities
- Edges detected ranging from 8.7% to 71.3%
- Total capital deployed: ~$1,400 (14% of $10,000 bankroll)
- Bankroll after deployment: $8,602.32
- Awaiting first settlement cycle for PnL realization

---

## 11. Discussion & Future Work

### 11.1 Known Limitations

**Synthetic backtest bias**: The backtest uses climatological probabilities as market prices, which is a weaker adversary than real market participants. Live Sharpe ratios will be substantially lower.

**No slippage model**: Paper trading assumes immediate fill at the quoted price. In reality, limit orders may not fill, and market orders incur spread costs. The 8% minimum edge threshold provides a buffer, but real-world execution costs will erode some edge.

**API dependency**: While the system now ingests four independent global models (significantly reducing single-model risk), all ensemble data flows through the Open-Meteo API. An Open-Meteo outage would affect all four ensemble sources simultaneously. NWS forecasts via `api.weather.gov` provide a partially independent fallback.

**Calibration cold start**: The isotonic calibrator requires 100+ forecast-outcome pairs to activate. During the cold start period, the system trades on uncalibrated probabilities. Similarly, the adaptive weight system requires 10+ scored markets per model before activating, operating on fixed curves during the cold start.

**Inter-model correlation**: While the four ensemble models use distinct dynamical cores, they share some common inputs (satellite observations, surface stations). In extreme weather regimes where all models struggle equally, the multi-model diversity provides less benefit than in typical conditions.

### 11.2 Planned Improvements

- **Extended ensemble post-processing**: Applying EMOS (Ensemble Model Output Statistics) or BMA (Bayesian Model Averaging) per-model before blending, rather than relying solely on raw member counting for probability estimation
- **Cross-model correlation analysis**: Quantifying the conditional dependence structure between ensemble models to optimally de-weight correlated errors and upweight genuinely independent signals
- **Analog forecasting**: Using historical temperature analogs (similar synoptic patterns) to supplement the ensemble distribution
- **Orderbook-aware execution**: Reading the Kalshi orderbook to optimize limit order placement and reduce adverse selection
- **Dynamic Kelly fraction**: Adjusting the fractional Kelly factor based on recent calibration quality and inter-model agreement (lower fraction when Brier score is elevated or models disagree significantly)
- **Regime-dependent weighting**: Detecting weather regimes (e.g., strong frontal passage, stable high-pressure) and adjusting model weights based on historically observed per-regime skill differences

### 11.3 Operational Considerations

AetherBot is designed for unattended operation. The APScheduler framework handles job timing, and the layered risk controls prevent catastrophic losses even if the model temporarily degrades. The daily loss limit of $300 (3% of initial bankroll) ensures that no single day can cause significant damage. The system logs all decisions, prices, probabilities, and trades for post-hoc analysis, and the Brier score decomposition provides interpretable diagnostics for identifying whether poor performance stems from calibration drift, resolution loss, or market regime change.

---

## Appendix A: Configuration Reference

```yaml
strategy:
  edge_threshold: 0.08          # 8% minimum edge to trade
  min_edge_hrrr_confirm: 0.06   # 6% with HRRR confirmation
  fractional_kelly: 0.15        # 15% of full Kelly
  max_position_pct: 0.05        # 5% of bankroll per trade
  max_position_dollars: 100     # $100 absolute cap
  daily_loss_limit: 300         # $300 daily loss limit
  max_concurrent_positions: 20
  max_positions_per_city: 6
  max_positions_per_date: 4
  min_contracts: 1
  max_price: 0.92
  min_price: 0.08
  max_lead_hours: 72

model_weights:
  # Global ensemble share (total) — split among available models
  gfs_ensemble: 0.60            # w_global (lead-time-adaptive total share)
  nws: 0.15                     # Fixed NWS share

  # Intra-global-ensemble proportional shares (sum = 100)
  global_model_shares:
    gfs: 30                     # 31 members, best CONUS hourly resolution
    ecmwf: 30                   # 51 members, highest global skill
    icon_eps: 22                # 40 members, independent icosahedral grid
    gem: 18                     # 21 members, independent semi-Lagrangian core

  # HRRR weight decays with lead time (high-res advantage fades)
  hrrr_weight_by_lead_hours:
    0: 0.45
    6: 0.35
    12: 0.25
    24: 0.15
    48: 0.05

  # Adaptive weight modulation
  adaptive_smoothing_alpha: 0.30  # 30% adaptive + 70% fixed curves
  adaptive_rolling_window_days: 60
  adaptive_min_scored_markets: 10  # Per-model minimum before adapting
```

## Appendix B: Observation Window Detail

The NWS CLI observation window is defined in Local Standard Time (LST), regardless of whether daylight saving time is in effect:

| City | Standard Offset | Window (UTC, Non-DST) | Window (UTC, During DST) |
|------|----|----|----|
| NYC | UTC-5 | 06:00 - 06:00+1 | 06:00 - 06:00+1 (unchanged) |
| Chicago | UTC-6 | 07:00 - 07:00+1 | 07:00 - 07:00+1 (unchanged) |
| Miami | UTC-5 | 06:00 - 06:00+1 | 06:00 - 06:00+1 (unchanged) |
| LA | UTC-8 | 09:00 - 09:00+1 | 09:00 - 09:00+1 (unchanged) |
| Denver | UTC-7 | 08:00 - 08:00+1 | 08:00 - 08:00+1 (unchanged) |

The "+1" denotes the next calendar day. The key insight is that the UTC boundaries do **not** shift when DST begins or ends, because the observation window is anchored to standard time.

---

*AetherBot v1.3. Built with Python, four independent weather models, 143 ensemble members, and a healthy respect for atmospheric chaos.*
