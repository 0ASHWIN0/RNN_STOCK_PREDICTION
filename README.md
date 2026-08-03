# RNN Stock Prediction

LSTM-based stock price prediction for MasterCard (MA) using historical data.

## Overview

This project trains an LSTM neural network to forecast MasterCard's stock
price. It downloads historical price data, preprocesses it, and trains a
single-layer LSTM (125 units) to predict the next day's high price from the
previous 60 trading days.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
uv sync
```

Requires Python 3.12+, TensorFlow/Keras 3, yfinance, pandas, numpy, and
scikit-learn.

## Usage

```bash
python main.py
```

If `data/Mastercard_stock_history.csv` is missing, the script downloads it
automatically via yfinance.

Optional CLI arguments:

```bash
python main.py --ticker MA --start 2010-01-01 --epochs 50 --batch-size 32 --seed 455
```

| Flag          | Default      | Description                          |
| ------------- | ------------ | ------------------------------------ |
| `--ticker`    | `MA`         | Stock ticker symbol                  |
| `--start`     | `2010-01-01` | Start date (YYYY-MM-DD)              |
| `--epochs`    | `50`         | Number of training epochs            |
| `--batch-size`| `32`         | Training batch size                  |
| `--seed`      | `455`        | Random seed for reproducibility      |

## How it works

1. **Data** — Daily OHLC data for MA (from 2010), downloaded via yfinance.
2. **Split** — Train on 2016–2020 highs, test on 2021 and beyond.
3. **Preprocess** — Scale prices to [0, 1] with MinMaxScaler and build
   60-step sliding windows (`X` = last 60 days, `y` = next day).
4. **Model** — LSTM (125 units, tanh) + Dense (1) compiled with RMSprop.
5. **Evaluate** — Inverse-scale predictions and report RMSE against the real
   test prices.

## Project structure

- `main.py` — entry point
- `Rnn.py` — data loading, preprocessing, model, training, evaluation
- `data/` — historical stock CSV

## Results

A sample run produces an RMSE of ~8.3 on the held-out test period.

Latest run (50 epochs, seed 455): test-set RMSE of 8.27.
