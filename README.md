Sentiment-Driven Stock Prediction
Overview

This project presents an end-to-end quantitative trading prototype that integrates financial news sentiment analysis with deep learning-based time series forecasting. The objective is to explore whether combining qualitative information extracted from financial text with quantitative market indicators can improve directional stock prediction.

The system leverages a transformer-based language model (FinBERT) to extract domain-specific sentiment from financial news and integrates it with a sequence-based Long Short-Term Memory (LSTM) network trained on historical price dynamics. The final output is a structured trading signal (BUY / SELL / HOLD) derived from both predicted return momentum and sentiment alignment.

This project demonstrates the integration of natural language processing, time series modeling, and algorithmic decision logic within a modular and extensible pipeline.

Problem Motivation

Financial markets are influenced by both numerical price behavior and qualitative market perception. Traditional quantitative models rely solely on price-derived indicators, while news-driven models often ignore temporal price structure.

This project aims to bridge that gap by:

Extracting sentiment signals from financial news using a domain-specific transformer model.

Modeling sequential dependencies in stock prices using LSTM networks.

Combining both signals into a coherent trading decision framework.

The emphasis is not only on predictive modeling but also on architectural consistency, modularity, and system integration.

System Architecture

The pipeline consists of five interconnected stages:

1. Financial News Acquisition and Sentiment Modeling

Recent financial news articles are fetched using NewsAPI. These articles are processed using the FinBERT transformer model (ProsusAI/finbert), which is specifically fine-tuned for financial sentiment classification.

Each article is tokenized and truncated to respect the 512-token transformer constraint. Model logits are converted into class probabilities (positive, neutral, negative), and a scalar sentiment score is computed as:

Sentiment Score = P(Positive) − P(Negative)

Multiple articles for a stock are aggregated into a single daily sentiment signal, ensuring a robust representation of current market perception.

2. Market Data Processing and Feature Engineering

Historical market data (OHLCV) is retrieved using Yahoo Finance. From this raw data, several technical features are computed:

Log returns

Rolling volatility

Relative Strength Index (RSI)

Moving Average Convergence Divergence (MACD)

Trading volume

These indicators capture momentum, mean reversion, volatility regimes, and liquidity characteristics of the stock.

The sentiment score derived from FinBERT is then incorporated into the feature space, enabling multimodal fusion of textual and numerical information.

3. Sequence Modeling using LSTM

Stock price dynamics are inherently sequential. To capture temporal dependencies, a multi-layer LSTM architecture is employed.

Key architectural details:

Input dimension: 7 features

Hidden dimension: 64 units

Number of layers: 2

Sequence length: 60 trading days

Loss function: Mean Squared Error

Optimizer: Adam

A sliding window approach converts the time series into supervised learning samples, where the model learns to predict the next-day return from the previous 60 days of multivariate features.

Model weights are saved using PyTorch checkpoints (.pth) and reloaded during the integration phase, ensuring reproducibility and architectural consistency.

4. Integration of Sentiment and Momentum

The LSTM outputs a predicted next-day return. However, predictions alone do not constitute a trading strategy. A sentiment-weighted decision rule is applied:

BUY if predicted return is positive and sentiment is strongly positive.

SELL if predicted return is negative and sentiment is strongly negative.

HOLD otherwise.

This alignment constraint reduces false signals by requiring agreement between technical momentum and market sentiment.

5. End-to-End Inference Pipeline

The integrated system performs the following operations:

Fetch one year of historical price data.

Compute technical indicators.

Retrieve recent news articles.

Compute aggregated sentiment using FinBERT.

Construct a 60-day feature sequence.

Run LSTM inference to predict next-day return.

Apply sentiment-weighted trading logic.

Generate a structured report per stock.

The final output is a clear and interpretable decision table containing predicted return, sentiment score, and recommended action.

Engineering Considerations

Several non-trivial engineering challenges were addressed during development:

Handling transformer token length limits (512 tokens).

Aggregating multiple news articles into a coherent daily sentiment signal.

Managing feature-dimension consistency between training and inference.

Saving and reloading PyTorch model weights without architectural mismatch.

Resolving input dimensionality errors during integration.

Designing a modular structure separating data ingestion, modeling, and decision logic.

The LSTM model was initially trained on a synthetic feature space to validate architectural correctness and checkpoint handling before integrating real market data. During final integration, a dummy feature was appended to maintain compatibility with the trained architecture while mapping real financial features into the expected input space.

