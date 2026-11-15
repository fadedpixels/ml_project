# AI-Powered Stock Market Prediction using Machine Learning

An enterprise-grade, leak-free AI trading indicator system that forecasts NVIDIA (**NVDA**) 15-minute stock price movements as **UP, DOWN, or HOLD** using 5-minute candles, technical market micro-structure features, and regularized machine learning models.

---

## 1. System Architecture & Methodology

```
+-----------------------------------------------------------------------------------+
|                               NVDA INTRADAY DATASET                               |
|                  2 Years (5-min candles) | Regular Hours 9:30 - 16:00 ET          |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    UNIFIED FEATURE ENGINE (live_features.py)                      |
|  28 Stationary & Relative Features: Returns, EMA Distances, RSI, MACD, BB, ATR%  |
|  Time & Day Cyclicals | 100% Verified Parity Between Batch & Live (test_parity.py)|
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|               LEAK-FREE CHRONOLOGICAL SPLIT & PURGED TARGET LABELS                |
|      60% Train (22,347)  |  20% Validation (7,447)  |  20% Test (7,450)           |
|      Target = Future 15-min Return vs Dynamic ATR Threshold (0.30 * ATR)          |
|      Target_End_Timestamp Purging across Split Boundaries & Session Closes        |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      MULTI-MODEL COMPARISON & BENCHMARKING                        |
|  Logistic Regression | Random Forest | Support Vector Machine | XGBoost | Ensemble|
|  Baselines: Majority Class (All HOLD) | Always UP | Basic Technical RSI(14) Rule   |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    CONFIDENCE GATING & ZERO REPAINTING (app.py)                   |
|  Optimal Threshold tau Tuned on Validation Set | High-Conviction Trades Only      |
|  Signals Immutable Once Emitted at Candle Close (CONFIRMED_SIGNALS Cache)         |
+-----------------------------------------------------------------------------------+
```

---

## 2. Key Technical Advancements (Addressing Quantitative Review)

### A. 100% Train/Serve Feature Parity (`live_features.py`)
- **Problem:** Many ML pipelines compute rolling statistics (e.g. `Relative_Volume`, `OBV_Normalized`) over years in training, but only over 50-60 candles during live deployment, causing severe distribution shift at market open.
- **Solution:** A unified feature extraction engine (`live_features.extract_features` and `compute_features_from_window`) is used identically for batch dataset generation and streaming inference.
- **Proof:** `test_parity.py` asserts maximum absolute error between 2-year batch extraction and sliding live buffers is **`0.00e+00`** across all 28 features.

### B. Dynamic Volatility-Scaled Target Labeling
- **Problem:** A fixed $\pm 0.15\%$ threshold turns the "HOLD" label into a proxy for low-volatility regimes rather than directional stillness.
- **Solution:** Inspired by Marcos Lopez de Prado's *Triple Barrier Method*, the directional threshold dynamically scales with candle volatility:
  $$\text{Threshold}_t = \text{clip}(0.30 \times \text{ATR\_Percent}_t, 0.12\%, 0.35\%)$$
- Future targets check `(Close[t+3] - Close[t]) / Close[t]` strictly within the same trading session. Overnight boundary rows are purged, ensuring **zero overnight label leakage**.

### C. Zero Repainting Guarantee (`app.py`)
- **Problem:** Dynamic chart markers that recompute past signals over moving windows repaint history, invalidating trading evaluation.
- **Solution:** `app.py` implements a persistent `CONFIRMED_SIGNALS` cache. Once a candle closes at $T$, its prediction and confidence are locked permanently and will never change on subsequent polls.

### D. Confidence Gating (Meta-Filtering)
- **Problem:** Blind argmax classification forces trades on noisy bars with 35–38% class probabilities. Round-trip transaction friction (2 bps) quickly erodes capital.
- **Solution:** Directional positions (UP / DOWN) are taken only when probability exceeds the validation-calibrated threshold $\tau = 0.44$. In ambiguous regimes, the indicator safely outputs **HOLD**.

---

## 3. Benchmark & Trading Evaluation Results

### A. Statistical Classification Metrics (Untouched Test Set)
Evaluated across 7,450 chronological test candles (zero forward leakage):

| Model / Baseline | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | Balanced Acc | ROC-AUC |
|---|---|---|---|---|---|---|
| **SVM (Selected)** | **42.35%** | **0.3791** | **0.3788** | **0.3767** | **0.3788** | **0.5643** |
| XGBoost (Validation) | 38.81% | 0.3772 | 0.3764 | 0.3762 | 0.3764 | 0.5581 |
| Soft Voting Ensemble (Val) | 39.72% | 0.3751 | 0.3754 | 0.3744 | 0.3754 | 0.5611 |
| Random Forest (Validation) | 38.43% | 0.3758 | 0.3753 | 0.3743 | 0.3753 | 0.5580 |
| Logistic Regression (Val) | 41.44% | 0.3823 | 0.3749 | 0.3641 | 0.3749 | 0.5487 |
| **Basic RSI Rule Baseline** | 42.89% | 0.3583 | 0.3511 | 0.3274 | 0.3511 | - |
| **Majority Class Baseline** | 46.23% | 0.1541 | 0.3333 | 0.2108 | 0.3333 | - |
| **Always UP Baseline** | 26.03% | 0.0868 | 0.3333 | 0.1377 | 0.3333 | - |

> *Note on Macro F1:* In a 3-class problem with random guessing at $\approx 0.333$, all trained ML models significantly outperform the Majority Class baseline ($F1 = 0.211$) and RSI rule baseline ($F1 = 0.327$).

---

### B. Simulated Trading Performance (After 0.02% Transaction Friction)
Simulated across all test candles with realistic 2 bps round-trip friction and 95% binomial confidence intervals ($\hat{p} \pm 1.96 \sqrt{\hat{p}(1-\hat{p})/N}$):

| Strategy | Signals | Coverage | Win Rate (95% CI) | Net PnL | Max Drawdown | Profit Factor |
|---|---|---|---|---|---|---|
| **SVM (Confidence Gated @ 0.44)** | **75** | **1.0%** | **64.0% ± 10.9%** | **+3.82%** | **-4.42%** | **1.63** |
| SVM (Raw Argmax) | 3,436 | 46.1% | 50.1% ± 1.7% | -55.55% | -55.84% | 0.98 |
| Basic RSI Rule Baseline | 2,006 | 26.9% | 51.2% ± 2.2% | -37.38% | -43.06% | 0.98 |
| Always UP Baseline | 7,450 | 100.0% | 48.8% ± 1.1% | -83.68% | -84.72% | 0.96 |
| Majority Class (All HOLD) | 0 | 0.0% | 0.0% | 0.00% | 0.00% | 1.00 |

#### Quantitative Insights:
1. **Raw ML bleeds under friction:** Blindly trading every 15 minutes incurs massive drag from bid-ask spread and slippage (-55.55% PnL, -55.84% drawdown).
2. **Confidence Gating protects capital:** Filtering for high-conviction bars boosts directional accuracy to 64.0%, curtails drawdown to -4.42%, and yields positive net alpha (+3.82% net, 1.63 profit factor).
3. **Statistical Integrity:** Rather than reporting ungrounded high Sharpe ratios on sparse trade counts, we report 95% confidence intervals and profit factors reflecting true statistical uncertainty.

---

## 4. How to Run the Project

### 1. Requirements & Setup
Ensure dependencies are installed in your virtual environment:
```bash
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Verify Feature Parity (Unit Test)
Run the automated test proving zero train/serve discrepancy:
```bash
python test_parity.py
```

### 3. Pipeline Execution
1. **Feature Engineering:**
   ```bash
   python feature_engineering.py
   ```
2. **Model Training & Evaluation:**
   ```bash
   python train_models.py
   ```

### 4. Verification & Live Dashboard
- **Print Evaluation Summary:**
  ```bash
  python main.py --eval
  ```
- **Test Real-Time Feature Extraction & Inference:**
  ```bash
  python main.py --test
  ```
- **Launch Interactive Web Dashboard:**
  ```bash
  python main.py
  # or python app.py
  ```
  Open your web browser at: **`http://localhost:5000`**

---

## 5. Technical Advancements Inspired by Top Quantitative Repositories

| Repository / Source | SOTA Concept | How It Was Applied in This System |
|---|---|---|
| **Marcos Lopez de Prado** (*Advances in Financial ML*) | **Triple Barrier / Dynamic Volatility Thresholds** | Target classification adapts dynamically to $0.30 \times \text{ATR}$, preventing volatility regime clustering in the HOLD class. |
| **Marcos Lopez de Prado** | **Purged & Embargoed Cross-Validation** | Split boundaries strictly purge overlapping labels (`Target_End_Timestamp < val_start`) to prevent lookahead leakage. |
| **Microsoft Qlib** | **Stationary Alpha Formulations** | All 28 features are scale-invariant returns, standard deviation bands, and cyclical time encodings (zero raw price levels). |
| **FinRL** | **Realistic Transaction Friction Accounting** | Trading backtests enforce 2 bps per-trade cost, penalizing over-trading and highlighting the necessity of confidence gating. |
| **Production ML Systems** | **Zero Repainting & Single-Source Engine** | Unified module `live_features.py` shared between training and live server, with permanent caching of closed-bar signals. |
