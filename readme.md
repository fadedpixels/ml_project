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
