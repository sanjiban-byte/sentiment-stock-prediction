# Sentiment-Driven Stock Prediction

An end-to-end quantitative trading prototype that integrates financial news sentiment analysis with deep learning-based time series forecasting to generate daily equity trading signals.

---

## Overview

This project builds a complete pipeline that combines two sources of information: qualitative sentiment extracted from financial news, and quantitative patterns learned from historical price data.

FinBERT, a transformer model fine-tuned on financial text, scores the sentiment of recent news articles for a given stock. These scores are combined with technical indicators derived from historical OHLCV data and fed into a multi-layer LSTM network that predicts the next-day log return. The predicted return and sentiment score are then passed through a signal layer that outputs a BUY, SELL, or HOLD recommendation.

The project demonstrates the integration of natural language processing, time series modelling, feature engineering, and algorithmic decision logic within a modular and reproducible pipeline.

---

## Problem Motivation

Financial markets are influenced by both numerical price behaviour and qualitative market perception. Traditional quantitative models rely solely on price-derived indicators, while news-driven approaches often ignore temporal price structure. This project bridges that gap.

The three core objectives are:

1. Extract sentiment signals from financial news using a domain-specific transformer model trained on financial vocabulary.
2. Model sequential dependencies in stock prices using an LSTM that treats the previous 60 trading days as a context window.
3. Combine both signals into a trading decision framework with interpretable outputs and honest evaluation metrics.

---

## System Architecture

The pipeline consists of five interconnected stages.

### 1. Financial News Acquisition and Sentiment Modelling

Recent financial news articles are fetched using the NewsAPI client. These articles are processed using FinBERT (ProsusAI/finbert), a BERT model fine-tuned on approximately 10,000 financial news sentences from Reuters.

Long documents are split into 512-token chunks before tokenisation to avoid silent truncation. Each chunk is scored independently and scores are averaged to produce a single scalar per stock per day:

```
Sentiment Score = mean( P(positive) - P(negative) )  across all chunks
```

Daily sentiment scores are cached to a CSV file so the FinBERT inference pass runs only once per stock. On subsequent runs the cache is loaded directly.

### 2. Market Data Processing and Feature Engineering

Historical OHLCV data is retrieved from Yahoo Finance. Seven features are computed:

| Feature | Description |
|---|---|
| Return | Log return: ln(Close_t / Close_{t-1}). Used as both prediction target and feature. |
| Volatility | Rolling 10-day standard deviation of log returns. Captures uncertainty regime. |
| RSI | Relative Strength Index over 14 days. Captures overbought and oversold conditions. |
| MACD | Difference between 12-day and 26-day exponential moving averages. Captures trend momentum. |
| Volume | Raw daily trading volume. Captures liquidity and conviction behind price moves. |
| Sentiment | Daily FinBERT score merged from the news cache. Zero-filled for days without coverage. |
| VWAP_Dev | Deviation of closing price from the 20-day rolling VWAP. Captures mean-reversion pressure. |

All seven features are standardised using a StandardScaler fitted exclusively on the training partition. The fitted scaler is persisted to disk and reloaded at inference time to ensure consistent feature distributions between training and prediction.

### 3. Sequence Modelling Using LSTM

A two-layer LSTM with 64 hidden units processes a sliding window of 60 trading days. Key details:

| Parameter | Value |
|---|---|
| Input dimension | 7 features per time step |
| Hidden dimension | 64 units |
| Number of layers | 2 (stacked LSTM) |
| Sequence length | 60 trading days |
| Loss function | Mean Squared Error |
| Optimiser | Adam, learning rate 1e-3 |
| Gradient clipping | Max norm 1.0 |

Model weights are saved at the epoch of best validation loss and reloaded at inference time.

### 4. Integration of Sentiment and Momentum

The LSTM outputs a predicted next-day log return. A sentiment-weighted signal rule is then applied:

```
BUY   if predicted_return > 0.001  and  sentiment_score > 0.2
SELL  if predicted_return < -0.001 and  sentiment_score < -0.2
HOLD  otherwise
```

Requiring agreement between the technical and sentiment signals before taking a directional position reduces false signals that arise from either source alone.

### 5. End-to-End Inference

The inference notebook (final.ipynb) performs the following steps for each stock:

1. Download two years of historical price data.
2. Compute all seven features.
3. Fetch recent news and compute today's sentiment score using chunked FinBERT.
4. Apply the training scaler to the feature matrix.
5. Construct a 60-day sequence and run LSTM inference.
6. Apply the signal rule.
7. Output a report with predicted return, sentiment score, article count, and recommended action.

---

## Training and Evaluation

### Data Split

The dataset is split temporally into training (70%), validation (15%), and test (15%) partitions in strict chronological order. No shuffling is applied at any stage. This prevents any form of look-ahead bias where the model could implicitly observe future data during training.

### Evaluation Metrics

Model performance is reported using two metrics meaningful in a trading context:

**Directional Accuracy** measures the fraction of test days where the predicted direction of return matches the actual direction. A naive baseline predicts the majority class and typically scores near 50 percent. Anything above 55 percent is considered informative.

**Paper-Trade Sharpe** is the annualised Sharpe ratio of a paper portfolio that goes long when the predicted return exceeds the threshold, short when it falls below the negative threshold, and flat otherwise. A Sharpe above 1.0 is considered good.

MSE is tracked during training for optimisation but is not the primary evaluation criterion, since directional accuracy is more relevant than magnitude accuracy for a trading signal.

---

## Default Stock Watchlist

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corporation | Technology |
| GOOGL | Alphabet Inc. | Technology |
| NVDA | NVIDIA Corporation | Semiconductors |
| TSLA | Tesla Inc. | Automotive / Clean Energy |

The watchlist can be modified by editing the `STOCKS` dictionary in `final.ipynb`.

---

## Known Limitations

**Sentiment coverage is sparse during training.** Due to the NewsAPI free-tier 30-day window, the model trains primarily on price features and implicitly learns to down-weight sentiment.

**Signal thresholds are fixed.** The values of 0.001 for return and 0.2 for sentiment are reasonable defaults but have not been tuned. These should be swept on the validation set and selected to maximise the paper-trade Sharpe.

**Single-stock training.** The model is trained independently for one stock (AAPL by default). Generalisation to other tickers without retraining has not been evaluated.

**No transaction costs.** The paper-trade Sharpe is a gross estimate before execution frictions, slippage, or position sizing constraints.

---

## Key Dependencies

| Package | Purpose |
|---|---|
| torch | LSTM model training and inference. |
| transformers | FinBERT tokeniser and sequence classification model. |
| yfinance | Yahoo Finance wrapper for historical OHLCV data. |
| newsapi-python | NewsAPI client for fetching financial news articles. |
| scikit-learn | StandardScaler for feature normalisation. |
| pandas | Data manipulation and time series alignment. |
| numpy | Numerical computation. |
| matplotlib | Training curves and equity chart visualisation. |
| scipy | Softmax computation for FinBERT output probabilities. |
