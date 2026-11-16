# PROJECT PREPARATION MASTER GUIDE: END-TO-END SYSTEM WALKTHROUGH
> Note for Direct Copy-Paste: Complete, exhaustive preparation guide for presenting, explaining, and defending the NVDA 15-Minute AI Directional Trading Indicator in front of evaluators, professors, or technical interviewers. All mathematical equations, formulas, and code references are presented in clean, legible plain-text format without raw LaTeX formatting, backslashes, or curly braces, ready for direct copy-pasting.

---

## TABLE OF CONTENTS
1. [Executive Summary and 60-Second Elevator Pitch](#1-executive-summary-and-60-second-elevator-pitch)
2. [The Core Philosophy: Why Most Trading ML Fails and How We Fixed It](#2-the-core-philosophy-why-most-trading-ml-fails-and-how-we-fixed-it)
3. [End-to-End System Architecture (The 7 Pillars)](#3-end-to-end-system-architecture-the-7-pillars)
   - [Pillar 1: Data Ingestion and Sanitization](#pillar-1-data-ingestion-and-sanitization)
   - [Pillar 2: Target Formulation and 15:40 ET Horizon Cutoff](#pillar-2-target-formulation-and-1540-et-horizon-cutoff)
   - [Pillar 3: The 28 Stationary Engineering Features](#pillar-3-the-28-stationary-engineering-features)
   - [Pillar 4: Chronological Purged Split (Zero Leakage)](#pillar-4-chronological-purged-split-zero-leakage)
   - [Pillar 5: Model Selection and Platt Scaling](#pillar-5-model-selection-and-platt-scaling)
   - [Pillar 6: Out-of-Sample Confidence Gate (The Game Changer)](#pillar-6-out-of-sample-confidence-gate-the-game-changer)
   - [Pillar 7: Production Parity and Live Dashboard](#pillar-7-production-parity-and-live-dashboard)
4. [Complete Codebase File-by-File Walkthrough](#4-complete-codebase-file-by-file-walkthrough)
5. [The Key Numbers to Memorize](#5-the-key-numbers-to-memorize)
6. [Step-by-Step Live Demo Presentation Script](#6-step-by-step-live-demo-presentation-script)
7. [Tough Evaluator Questions and Bulletproof Answers (Viva Defense)](#7-tough-evaluator-questions-and-bulletproof-answers-viva-defense)

---

## 1. EXECUTIVE SUMMARY AND 60-SECOND ELEVATOR PITCH

### The 30-Second Version:
"We built an institutional-grade, zero-repainting machine learning indicator that predicts 15-minute price direction for NVIDIA (NVDA) using 5-minute closed regular-session candles. Intraday financial markets have an extremely low signal-to-noise ratio. Instead of forcing the model to trade every 5 minutes—which loses -55% due to fee churn—we engineered 28 strictly stationary features, purged split boundaries, calibrated SVM probabilities via Platt scaling, and tuned an out-of-sample confidence gate at 44%. This filters out 99% of noisy market chop, firing on only ~1% of high-conviction candles. On the completely untouched out-of-sample test set of 7,450 candles, it achieves a 64.0% win rate across 75 trades and +3.82% net profit after deducting realistic 2 basis points transaction costs, while passive holding lost -83% and an RSI rule lost -37%."

### Key Selling Points:
1. Zero Repainting: Signals emitted at candle close are immutable and verified to have zero discrepancy (Unit Parity Error = 0.00) between batch historical data and live real-time ingestion.
2. Zero Lookahead Leakage: Features use closed bars only; split boundaries purge 3 forward candles; probability thresholds were tuned strictly on validation data before any refitting.
3. Honest Engineering: Evaluated with 2 basis points (0.02%) slippage and fees; statistical win rate backed by Wilson score confidence intervals (53.1% to 74.9%).

---

## 2. THE CORE PHILOSOPHY: WHY MOST TRADING ML FAILS AND HOW WE FIXED IT

Most machine learning trading projects fail when deployed live. In your presentation, highlight these four deadly pitfalls and explain how our architecture solves each one:

| Pitfall in Standard ML Trading | How Our Architecture Explicitly Solves It |
| :--- | :--- |
| 1. Non-Stationary Price Levels (Models memorize nominal dollar values like $120 vs $140, failing when price moves to new all-time highs). | Zero raw prices. All 28 features are scale-invariant returns, relative distances, ATR %, and normalized volume oscillators. |
| 2. Split Boundary Lookahead Leakage (Predicting 15m forward leaks the first 3 candles of the next split into the training set). | Purged chronological 60/20/20 split. The 3 candles before validation/test boundaries are dropped so future labels never contaminate the training set. |
| 3. The Continuous Trading Fallacy (Forcing an active UP/DOWN position every 5m causes death by 1,000 cuts from transaction fee drag). | Selective Confidence Gating (Threshold = 0.44). Standing aside in cash (HOLD / NO_SIGNAL) is an active capital-preservation decision that avoids 99% of market noise. |
| 4. Indicator Repainting (Historical indicators changing retroactively after the fact). | Fixed sliding window (60 bars). Signals freeze at bar close and are written to an atomic disk cache (results/live_signals.json). They never change. |

---

## 3. END-TO-END SYSTEM ARCHITECTURE (THE 7 PILLARS)

```
======================================================================================================
                                   END-TO-END SYSTEM PIPELINE
======================================================================================================

 [1. MASSIVE API / CSV] ---> 5-min OHLCV bars (09:30 - 16:00 ET only; 38,747 candles)
            |
            v
 [2. TARGET ENGINE]     ---> Forward 15m return (3 bars); +/-0.15% threshold -> DOWN (0), HOLD (1), UP (2)
            |                Cutoff safeguard: candles after 15:40 ET receive NO_SIGNAL
            v
 [3. FEATURE ENGINE]    ---> 28 Stationary Features (Returns, EMAs, RSI, MACD, BB, ATR%, Volume, Time)
            |                Verified parity: max absolute difference = 0.00
            v
 [4. PURGED SPLIT]      ---> 60% Train (22,348) | [Purge 3] | 20% Val (7,449) | [Purge 3] | 20% Test (7,450)
            |                Chronological order preserved; zero shuffling
            v
 [5. MODEL SELECTION]   ---> Benchmarked LR, RF, XGBoost, Voting, SVM (RBF Kernel selected on Val Macro-F1)
            |                Platt Scaling: margin distance f(x) -> calibrated probabilities [P_DOWN, P_HOLD, P_UP]
            v
 [6. CONFIDENCE GATE]   ---> Out-of-sample search on pre-refit Val probs: tau = 0.44 (1.0% coverage target)
            |                Fires UP if P_UP >= 0.44 and P_UP > P_DOWN; DOWN if P_DOWN >= 0.44; else NO_SIGNAL
            v
 [7. PRODUCTION UI]     ---> Lightweight Charts v4.2.2 terminal, persistent live_signals.json, Wilson CI
======================================================================================================
```

---

### PILLAR 1: DATA INGESTION AND SANITIZATION
- Asset: NVIDIA Corporation (NASDAQ: NVDA). High liquidity, heavy retail and institutional order flow, clean spreads.
- Horizon and Resolution: 5-minute candles across 24 months (November 2024 to October 2026, totaling 38,747 bars).
- Session Boundary Rules: Strictly Regular Trading Hours (RTH), 09:30:00 to 16:00:00 US/Eastern.
  - Pre-market and post-market are stripped out because illiquid spreads distort technical indicators.
  - Weekends and US stock market holidays (Thanksgiving, Christmas, New Year's Day, etc.) are pruned.
- Overnight Gap Handling: The first candle of day D (09:30 ET) does not calculate returns across the overnight gap from day D-1 (16:00 ET). Each session starts clean.

---

### PILLAR 2: TARGET FORMULATION AND 15:40 ET HORIZON CUTOFF
- Forward Horizon: 15 minutes = exactly 3 subsequent 5-minute bars (k=3):
  
  Forward Return R(t to t+3) = (Close_(t+3) - Close_t) / Close_t

- Threshold Formulation:
  - UP (Class 2): Forward Return >= +0.15% (+0.0015)
  - DOWN (Class 0): Forward Return <= -0.15% (-0.0015)
  - HOLD (Class 1): Forward Return between -0.15% and +0.15%
  - Why +/- 0.15%? A 5-minute bar on NVDA averages ~0.25% volatility. Setting the threshold at +/- 0.15% covers our 2 bps transaction cost plus leaves a clean directional margin, giving a natural class balance of 28% DOWN, 46% HOLD, 26% UP.
- 15:40 ET Operational Cutoff Rule:
  - The market closes at 16:00:00 ET.
  - A signal emitted at 15:45 ET would require a 15-minute horizon ending at 16:00 ET (only 2 candles inside the session).
  - Any candle after 15:40 ET cannot complete a full 3-candle intraday horizon.
  - Therefore, the system automatically tags candles after 15:40 ET with "Late Session Horizon Cutoff" and emits NO_SIGNAL.

---

### PILLAR 3: THE 28 STATIONARY ENGINEERING FEATURES
Never feed raw stock prices to a machine learning model. If NVDA trades at $100 in 2024 and $140 in 2026, a tree split at $115 becomes completely invalid. We engineered 28 scale-invariant features:

1. Returns and Momentum (6 features):
   - Return_1: 5-minute immediate return = (Close_t - Close_(t-1)) / Close_(t-1)
   - Return_3: 15-minute lookback return = (Close_t - Close_(t-3)) / Close_(t-3)
   - Return_6: 30-minute lookback return = (Close_t - Close_(t-6)) / Close_(t-6)
   - Return_12: 1-hour lookback return = (Close_t - Close_(t-12)) / Close_(t-12)
   - Return_24: 2-hour macro momentum = (Close_t - Close_(t-24)) / Close_(t-24)
   - Close_Open_Return: Candle body conviction = (Close_t - Open_t) / Open_t
2. Moving Average Distances and Stretch (4 features):
   - EMA_9_Distance: (Close_t - EMA_9) / EMA_9 (Short-term trend stretch)
   - EMA_20_Distance: (Close_t - EMA_20) / EMA_20 (Mean-reversion anchor)
   - EMA_50_Distance: (Close_t - EMA_50) / EMA_50 (Intraday macro trend)
   - VWAP_Distance: (Close_t - VWAP_t) / VWAP_t (Distance from institutional volume-weighted average price)
3. Oscillators (4 features):
   - RSI_14: Relative Strength Index bounded between 0 and 100
   - MACD: Fast EMA (12) minus Slow EMA (26)
   - MACD_Signal: 9-period EMA of MACD
   - MACD_Histogram: MACD minus Signal (captures momentum acceleration)
4. Volatility and Range (4 features):
   - High_Low_Range: Intra-bar candle spread = (High_t - Low_t) / Close_t
   - BB_Width: (Upper Band - Lower Band) / Middle Band (Bollinger squeeze detector)
   - BB_Percent_B: Position inside the Bollinger Bands = (Close_t - LB) / (UB - LB)
   - ATR_Percent: 14-period Average True Range normalized by Close price
5. Volume and Liquidity (5 features):
   - Log_Volume: ln(Volume_t + 1)
   - Relative_Volume: Volume_t / SMA_20(Volume) (Spike detector)
   - Volume_Change: Rate of change of volume bar-over-bar
   - OBV_Normalized: Z-scored On-Balance Volume
   - Log_Transactions: ln(Trades_t + 1) (Institutional activity proxy)
6. Temporal and Cyclical Time (5 features):
   - Minutes_Since_Open: Elapsed session minutes (0 to 390)
   - Time_Sin, Time_Cos: Cyclical encoding of minute of day: sin / cos (2 * pi * minute / 1440)
   - Weekday_Sin, Weekday_Cos: Cyclical encoding of day of week: sin / cos (2 * pi * day / 7)

---

### PILLAR 4: CHRONOLOGICAL PURGED SPLIT (ZERO LEAKAGE)
- Why Random K-Fold Cross-Validation is strictly forbidden in finance:
  Shuffling time series data causes future data points to predict past data points (lookahead leakage). You get 90% accuracy in the lab and lose 100% of your money in live trading.
- Our Split Architecture:
  - Train Set (60%): Nov 15, 2024 to Nov 25, 2025 (22,348 candles)
  - Validation Set (20%): Nov 26, 2025 to Jun 10, 2026 (7,449 candles)
  - Test Set (20%): Jun 11, 2026 to Oct 18, 2026 (7,450 candles)
- Label Purging Mechanism:
  Because each label spans 3 candles forward (t+3), candle t_train_end uses price data from t_val_start+1 and t_val_start+2. We explicitly purge the final 3 candles of the training set and the final 3 candles of the validation set before training. This guarantees complete mathematical independence between splits.

---

### PILLAR 5: MODEL SELECTION AND PLATT SCALING
We benchmarked five distinct architectures on the validation fold:
1. Logistic Regression (L2 penalty): Macro-F1 = 0.3641
2. Random Forest (200 trees, depth 6, leaf 50): Macro-F1 = 0.3743
3. XGBoost (depth 4, lr 0.05, reg L1, L2): Macro-F1 = 0.3762
4. Soft Voting Ensemble: Macro-F1 = 0.3744
5. Support Vector Machine (RBF Kernel, C = 1.0): Macro-F1 = 0.3789 (WINNER)

#### Why SVM Won Over Tree Ensembles:
In noisy financial datasets, tree models create rigid orthogonal decision boundaries that over-partition sparse edge cases into noisy leaves. The RBF SVM constructs a smooth, maximum-margin boundary in the 28-dimensional standardized continuous feature space. Combined with balanced class weighting, it avoided the majority-class collapse that afflicted Logistic Regression.

#### Platt Probability Calibration:
Raw SVM outputs distance f(x) to the hyperplane, not probabilities. We fit a sigmoid function:

Probability(y = c given x) = 1 / (1 + exp(A * f(x) + B))

and normalize across all three classes via softmax so that Probability(DOWN) + Probability(HOLD) + Probability(UP) = 1.0.

---

### PILLAR 6: OUT-OF-SAMPLE CONFIDENCE GATE (THE GAME CHANGER)
This is the single most important slide and concept in your presentation.

#### The Problem with Raw Argmax:
If you take the raw prediction every candle, the model emits 3,436 trades across 7,450 candles (46% coverage).
- Raw win rate: 50.1% (essentially a coin flip).
- Transaction cost deduction: 3,436 * 0.02% = 68.72% in fee drag!
- Net result: -55.55% loss and -55.84% drawdown.

#### The Out-of-Sample Confidence Gate Solution:
We search the pre-refit validation probabilities for a probability threshold (tau) that isolates the top ~1.0% highest-conviction trades.
- At tau = 0.44 (44%), the model requires that Prob_UP >= 0.44 and Prob_UP > Prob_DOWN to fire an UP signal.
- In all other cases (99% of candles), the indicator outputs NO CLEAR SIGNAL (capital preserved).

#### Why Tuning on Pre-Refit Validation Probs Matters:
In naive pipelines, people refit the model on Train+Val, and then tune the threshold on the validation set. But on data the model has already trained on, SVM probabilities are overconfident! Tuning tau on pre-refit validation probabilities guarantees that tau = 0.44 is a genuine out-of-sample decision parameter.

---

### PILLAR 7: PRODUCTION PARITY AND LIVE DASHBOARD
- Parity Test (test_parity.py): Verified that live_features.compute_features_from_window() matches the historical batch matrix down to machine precision (Max Error = 0.00).
- Massive API Live Feed: During market hours (09:30 to 16:00 ET, Mon-Fri), app.py polls Massive API for closed 5-minute candles. When market is closed, it switches to CLOSED (OUT-OF-SAMPLE REPLAY) using strictly test set data.
- Zero Repainting Persistence: When a candle closes, the emitted signal is written to results/live_signals.json. If the server restarts, historical signals are reloaded from disk. A signal is never recalculated.
- Wilson Score Confidence Interval: Normal Wald CI breaks on small sample sizes (e.g., 2/2 wins gives 100% +/- 0%). Wilson score interval computes valid statistical bounds (53.1% to 74.9%) and hides the win rate until at least 20 signals resolve.

---

## 4. COMPLETE CODEBASE FILE-BY-FILE WALKTHROUGH

When the evaluator asks "Show me where X happens in the code", point directly here:

1. app.py:
   - Lines 80-120: fetch_live_candles() calls Massive API with 15s rate-limit cache and 60-bar warmup.
   - Lines 125-150: is_us_equity_market_open() checks Eastern Time and weekdays.
   - Lines 155-190: load_persisted_signals() and persist_signals() save to results/live_signals.json.
   - Lines 210-235: compute_wilson_ci() calculates binomial Wilson interval with sample size gate.
   - Lines 250-290: 15:40 ET horizon cutoff safeguard suppressing late-session signals.
2. train_models.py:
   - Section 3: Loads cleaned data and calculates forward 15m target return (3 candles).
   - Section 4: Chronological 60/20/20 train/val/test split with boundary purging.
   - Section 6: StandardScaler fitted on Train only, transforming Val and Test.
   - Section 7: Benchmarks 5 models (LR, RF, XGB, SVM, Voting).
   - Section 10B: Saves pre-refit validation probabilities and tunes tau = 0.44 at fixed 1% coverage.
   - Section 10C: Refits SVM on Train+Val, saves best_model.joblib.
   - Section 11: Evaluates once on untouched Test set, saves trading_metrics.csv.
3. feature_engineering.py:
   - Vectorized computation of 28 stationary indicators per trading session.
4. live_features.py:
   - Real-time incremental feature extraction from sliding 60-candle window.
5. test_parity.py:
   - Automated unit test verifying that live_features.py outputs identical numbers to feature_engineering.py (Max Error = 0.00).
6. dashboard.html:
   - Single-page dark/light quantitative terminal built with TradingView Lightweight Charts v4.2.2.
   - Displays candlestick chart, VWAP, EMA-20, Conviction Meter, Probabilities, Snapshot tiles, Safeguards chips, Live Scorecard, Signal Log, and Backtest confusion matrix.

---

## 5. THE KEY NUMBERS TO MEMORIZE

Commit these exact figures to memory for rapid answers:

| Metric | Exact Value |
| :--- | :--- |
| Total Historical Dataset (NVDA 5-min candles) | 38,747 candles (24 months) |
| Untouched Test Set Size | 7,450 candles |
| Forward Prediction Horizon | 15 minutes (3 candles) |
| Directional Return Threshold | +/- 0.15% (+/- 0.0015) |
| Number of Stationary Features | 28 features (0 raw prices) |
| Confidence Threshold (tau) | 0.44 (44.0%) |
| Model Architecture | Support Vector Machine (RBF Kernel) |
| Gated Test Signals Emitted | 75 signals (~1.0% coverage) |
| Gated Win Rate | 64.0% |
| 95% Wilson Confidence Interval | 53.1% to 74.9% |
| Net Backtest PnL (after 2 bps costs) | +3.82% |
| Maximum Drawdown | -4.42% |
| Profit Factor | 1.63 |
| Raw Unconstrained SVM PnL | -55.55% |
| Basic RSI Rule PnL | -37.38% |
| Always UP (Passive Long) PnL | -83.68% |
| Feature Parity Absolute Error | 0.00 (Exact Parity Match) |

---

## 6. STEP-BY-STEP LIVE DEMO PRESENTATION SCRIPT

Follow this exact 5-minute walkthrough when presenting the running dashboard:

### Step 1: Open the Terminal Dashboard
- Open browser to http://localhost:5000.
- Say: "Here is our live quantitative terminal. In the top bar, you can see NVDA's current price, the market status pill, the countdown timer to the next 5-minute candle close, and our operational horizon cutoff at 15:40 ET."

### Step 2: Explain the Chart and Indicators
- Point to the TradingView chart with green/red candles, yellow VWAP, and blue dotted EMA-20.
- Say: "The chart plots 5-minute closed candles. Below are trading volume bars. Notice the signal markers on the chart: UP arrows in green and DOWN arrows in red. Notice that there are very few markers—this is completely intentional. The indicator only fires when high-conviction probability exceeds 44%."

### Step 3: Explain the Signal Card and Conviction Meter (Right Column)
- Point to the "NO CLEAR SIGNAL" (or active UP/DOWN) state box.
- Say: "In retail trading tools, you see constant blinking buy/sell signals that churn accounts into losses. Our terminal makes 'NO CLEAR SIGNAL' the normal, default state. Below the state title, you can see the calibrated class probabilities: P(DOWN), P(HOLD), and P(UP). The white vertical line inside the track marks our 44% threshold. Look at the Conviction Meter below: it tells the trader exactly which direction is leading and how many percentage points are needed to trigger an actionable trade."

### Step 4: Show the Snapshot Tiles and Safeguards
- Point to RSI (14), MACD Hist, ATR %, Rel Vol, VWAP Dist, EMA-20 Dist.
- Point to the active safeguards chips (Early Session, Low Volume, Late Session).
- Say: "Every feature is stationary and relative. Furthermore, our safeguard chips alert the trader to structural risks—such as the 15:40 ET cutoff where any signal is suppressed because the forward 15-minute horizon would extend past market close."

### Step 5: Switch to the "Backtest and Metrics" Tab
- Click the Backtest and Metrics tab at the bottom right.
- Show Table 1 and the Confusion Matrix.
- Say: "Here are the rigorous out-of-sample results on 7,450 unseen test candles. Unconstrained SVM lost -55% from fee drag. The RSI baseline lost -37%. Our confidence-gated SVM achieved a 64.0% win rate across 75 trades, producing +3.82% net profit and a 1.63 Profit Factor under simulated 2 bps costs. This proves that in quantitative finance, selectivity is the true edge."

---

## 7. TOUGH EVALUATOR QUESTIONS AND BULLETPROOF ANSWERS (VIVA DEFENSE)

### Q1: "Your model accuracy is only 42.3%. Why should anyone use a model with 42% accuracy?"
Answer:
"In a 3-class problem (DOWN, HOLD, UP), random chance is 33.3%. More importantly, raw accuracy evaluates a model under the continuous trading fallacy—forcing an active decision on every single noisy 5-minute candle. In financial markets, signal-to-noise ratio is near zero for 95% of the trading day. Evaluating continuous accuracy is meaningless because a trader should never trade noise. When we apply our out-of-sample confidence gate of 44%, we isolate the top 1% highest-conviction setups, where the model achieves a 64.0% win rate and a Profit Factor of 1.63. In trading, selectivity and precision matter; continuous accuracy does not."

### Q2: "75 trades across 7,450 candles is only 1% coverage (~1 trade per day). Isn't that too few trades?"
Answer:
"No, that is the most realistic and defensible aspect of our project. NVDA 5-minute candles are dominated by microstructure noise, market-maker spread capture, and random order flow. True institutional momentum setups occur only once or twice a day—typically during the morning session opening momentum or afternoon trend continuation. If an algorithm traded 50 times a day, paying 2 basis points per trade would destroy 100% of capital in weeks. One high-conviction trade per day with a 64% win rate is an institutional risk-management profile."

### Q3: "What is 'purging' and why did you need it in your train/test split?"
Answer:
"Because our target is a 15-minute forward return spanning 3 candles (t+3), the label for the final 3 candles of the training set uses closing prices that occur inside the validation set. If you don't purge those 3 candles, the training fold literally trains on prices that belong to the validation set—a classic split-boundary lookahead leakage identified by Marcos Lopez de Prado. We explicitly dropped the final 3 candles before each split boundary, guaranteeing complete mathematical independence."

### Q4: "How do you know you didn't overfit the confidence threshold of 0.44?"
Answer:
"We derived the 0.44 threshold strictly on the validation set using the model's pre-refit probabilities, targeting a fixed ~1.0% coverage regime. The test set was kept completely untouched in a sealed vault. Only after the model and threshold were finalized was the pipeline run once on the test set. The fact that the test set delivered a 64.0% win rate and 1.63 Profit Factor with ~1.0% coverage confirms that the threshold generalized out-of-sample."

### Q5: "What is repainting and how did you prevent it?"
Answer:
"Repainting occurs when an indicator retroactively alters historical buy/sell arrows because future candles change past indicator values—a common flaw in ZigZag or unanchored moving averages. We prevented repainting in three ways:
1. Features are computed strictly at candle close using an immutable 60-bar historical window.
2. test_parity.py proves machine precision parity (Error = 0.00) between batch and live modes.
3. Once a signal is emitted, it is written to results/live_signals.json. Even if the server crashes and restarts, past signals are reloaded from disk and never recomputed."

### Q6: "Why did you use Support Vector Machines (SVM) instead of Deep Learning (LSTM / Transformers)?"
Answer:
"Intraday tabular financial data has low signal-to-noise ratio and high non-stationarity. Deep networks (like LSTMs) have millions of parameters and notoriously overfit high-frequency financial noise, memorizing spurious correlations. RBF Support Vector Machines maximize the geometric margin between classes in a continuous standardized feature space. With regularized C = 1.0 and balanced class weighting, SVM acts as a robust maximum-margin separator that resists noise better than deep neural networks or deep tree ensembles."

### Q7: "Why did you use the Wilson Score Interval instead of standard Normal (Wald) Confidence Interval?"
Answer:
"The standard Wald interval assumes a Gaussian approximation. It breaks down completely near boundaries (win rates near 0% or 100%) and collapses on small sample sizes—for example, 2 wins out of 2 trades produces an absurd '100% +/- 0.0%' confidence interval. The Wilson score interval inverts the score test, maintaining valid asymmetric margins across small samples and boundary conditions. Furthermore, our dashboard hides the win rate until at least 20 signals have resolved to ensure statistical integrity."

### Q8: "What is the 15:40 ET horizon cutoff rule and why is it necessary?"
Answer:
"Regular market trading ends at 16:00:00 ET. Because our target predicts 15 minutes into the future (3 candles), a signal emitted at 15:45 ET would require price data at 16:00 ET (only 2 candles), and a signal at 15:50 would require after-hours prices. After-hours trading suffers from illiquid spreads and overnight gap risk. To maintain operational consistency, the indicator halts all directional signals after 15:40 ET, tagging the state as 'Late Session Horizon Cutoff'."

### Q9: "What are the limitations of your project?"
Answer:
"We transparently document two main limitations:
1. Trade Autocorrelation: Because each signal has a 15-minute horizon on 5-minute bars, consecutive signals share two-thirds of their price trajectory. While rare due to our 1% gate, sequential PnL is an approximation.
2. Platt Scaling Monotonicity: Platt scaling assumes a parametric sigmoid curve mapping margin distance to probability. In future work, non-parametric isotonic regression or Venn-Abers predictors could provide tighter empirical calibration."

---
End of Master Preparation Guide. Good luck with your presentation tomorrow!
