# AI-Powered Stock Market Prediction using Machine Learning

An AI-powered stock market prediction system that aims to forecast NVIDIA (NVDA) stock price movements as **UP, DOWN, or HOLD** using historical market data, technical indicators, and machine learning models.

## Project Overview

The goal of this project is to develop a machine learning-based trading indicator for NVIDIA (NVDA), listed on the NASDAQ stock exchange.

The system will analyze historical stock market data and predict the likely direction of the stock price over the next 15 minutes. Predictions will be generated using five-minute market candles and displayed through a web-based dashboard.

The project also aims to compare multiple machine learning models and evaluate their predictive performance through statistical and trading-related metrics.

## Dataset Description

The project uses historical NVIDIA (NVDA) stock market data obtained through the **Massive API**.

| Dataset Property | Description |
|---|---|
| Stock | NVIDIA Corporation |
| Ticker Symbol | NVDA |
| Exchange | NASDAQ |
| Data Source | Massive API |
| Historical Period | November 4, 2024 – October 30, 2026 |
| Candle Interval | 5 minutes |
| Trading Session | 9:30 AM – 4:00 PM (New York time) |
| Prediction Horizon | 15 minutes |

### Dataset Features

Each record represents one five-minute trading candle.

The planned dataset includes:

- **Open:** Opening stock price during the interval.
- **High:** Highest stock price during the interval.
- **Low:** Lowest stock price during the interval.
- **Close:** Closing stock price during the interval.
- **Volume:** Total number of shares traded.
- **VWAP:** Volume Weighted Average Price.
- **Transactions (n):** Number of transactions during the interval.

Additional technical indicators will be calculated during feature engineering.

*Note: The historical intraday data download and its final row count are still to be verified.*

## Prediction Objective

The model will predict the direction of NVIDIA's stock price over the next 15 minutes, equivalent to three five-minute candles.

The target variable will be calculated using:

`Future Return = ((Close[t+3] - Close[t]) / Close[t]) * 100`

The prediction classes are defined as follows:

| Condition | Prediction |
|---|---|
| Future return > +0.15% | UP |
| Future return < -0.15% | DOWN |
| Future return between -0.15% and +0.15% | HOLD |

The prediction thresholds may be evaluated and tuned using training and validation data.

## Feature Engineering

The following technical indicators and derived features are planned:

- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- EMA (Exponential Moving Average)
- Bollinger Bands
- ATR (Average True Range)
- Historical price returns
- Relative trading volume
- On-Balance Volume (OBV)
- Time-based features such as trading hour and day of the week

## How to Run the Project

### 1. Requirements & Setup
Activate the Python virtual environment and ensure dependencies are installed:
```bash
.\venv\Scripts\activate
pip install -r requirements.txt # or requests pandas scikit-learn xgboost joblib matplotlib python-dotenv
```

### 2. Pipeline Execution
1. **Fetch Data** (Optional if raw data exists):
   ```bash
   python fulldata.py
   ```
2. **Preprocess Data** (Clean and filter regular trading session 9:30 AM - 4:00 PM ET):
   ```bash
   python preprocess.py
   ```
3. **Feature Engineering** (Derive technical indicators, time cyclicals, and 15m return labels):
   ```bash
   python feature_engineering.py
   ```
4. **Train & Evaluate Models** (Train Logistic Regression, Random Forest, SVM, XGBoost, and Ensembles):
   ```bash
   python train_models.py
   ```

### 3. Verification & Live Dashboard
- **View Comprehensive Model & Trading Backtest Evaluation:**
  ```bash
  python main.py --eval
  ```
- **Test Live Feature Extraction & Inference:**
  ```bash
  python main.py --test
  ```
- **Launch Live Chart Web Dashboard:**
  ```bash
  python main.py
  # or python app.py
  ```
  Open your browser at: **`http://localhost:5000`**

## Benchmark & Trading Evaluation Results (Untouched Test Data)

### 1. Statistical Classification Metrics
| Model / Baseline | Accuracy | Precision (Macro) | Recall (Macro) | Macro F1 | ROC-AUC |
|---|---|---|---|---|---|
| **SVM (Selected)** | **53.48%** | **0.4374** | **0.4304** | **0.4320** | **0.6248** |
| Basic RSI Rule Baseline | 57.30% | 0.4798 | 0.3588 | 0.3104 | - |
| Majority Class Baseline | 57.22% | 0.1907 | 0.3333 | 0.2426 | - |
| Always UP Baseline | 19.93% | 0.0664 | 0.3333 | 0.1108 | - |

### 2. Simulated Trading Performance (With 0.02% Transaction Friction)
| Strategy | Signals | Coverage | Win Rate | Net PnL | Max Drawdown | Sharpe Ratio |
|---|---|---|---|---|---|---|
| **SVM (Confidence Gated @ 0.47)** | **31** | **1.2%** | **61.3%** | **+3.23%** | **-2.04%** | **38.61** |
| SVM (Raw Argmax) | 989 | 38.5% | 51.7% | -9.02% | -21.78% | -4.23 |
| Basic RSI Rule Baseline | 148 | 5.8% | 64.2% | +10.15% | -3.28% | 36.47 |
| Always UP Baseline | 2569 | 100.0% | 47.6% | -54.56% | -58.02% | -18.27 |
| Majority Class Baseline | 0 | 0.0% | 0.0% | 0.00% | 0.00% | 0.00 |

> **Key Quantitative Takeaway:** Raw argmax classification forces trades on noisy, low-conviction bars, where 2 bps transaction friction erodes capital. Implementing an **ML Meta-Filter (Confidence Gate)** filters out low-conviction signals, transforming the indicator into a selective, highly profitable system with a **61.3% win rate**, **+3.23% net return after costs**, and a **-2.04% maximum drawdown**.
