# Stock Market Anomaly Detection

A comprehensive Python-based pipeline for detecting price and volume anomalies in stock market data using **K-Means clustering**, **DBSCAN**, and **rule-based methods**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project implements a walk-forward anomaly detection system that identifies unusual trading patterns in stock market data. It uses multiple detection algorithms to flag anomalous price movements and volume spikes, providing interpretable results with severity scores and explanations.

### Key Features

- **Multi-algorithm Detection**: Combines K-Means, DBSCAN, and rule-based detectors for robust anomaly identification
- **Walk-forward Validation**: Prevents data leakage by using only past data for feature computation
- **Market-wide Analysis**: Detects both individual stock anomalies and market-level events
- **Interpretable Results**: Provides explanations for why each anomaly was flagged
- **Severity Scoring**: Ranks anomalies from 0-100 based on feature extremity
- **Query Tools**: CLI tools for querying results by date or generating monthly reports

---

## 🏗️ Project Structure

```
stock-anomaly-detection/
├── data/
│   └── raw/
│       └── stocks/           # CSV files per ticker (AAPL.csv, MSFT.csv, etc.)
├── outputs/
│   ├── full_results.csv      # Complete detection results
│   ├── anomaly_cards.csv     # Flagged anomalies with explanations
│   ├── market_days.csv       # Market-level anomaly days
│   └── monthly_reports/      # Monthly summary reports
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Data loading and preprocessing
│   ├── features.py           # Feature engineering (leakage-free)
│   ├── detectors.py          # Anomaly detection algorithms
│   ├── walkforward.py        # Main pipeline orchestrator
│   ├── query.py              # Date query tool
│   └── monthly.py            # Monthly report generator
└── requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd stock-anomaly-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the dataset**
   
   Download the stock market dataset from Kaggle:
   - Go to: [Stock Market Dataset](https://www.kaggle.com/datasets/jacksoncrow/stock-market-dataset)
   - Download and extract to `data/raw/`
   - Ensure the `stocks/` folder contains individual ticker CSV files

   *Or use the Kaggle API:*
   ```python
   from src.data_loader import download_kaggle_data
   download_kaggle_data()
   ```

---

## 📖 Usage

### Running the Full Pipeline

```bash
python -m src.walkforward --tickers QQQ AAPL MSFT AMZN NVDA META
```

#### Command-line Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--tickers` | QQQ AAPL MSFT... | Stock tickers to analyze |
| `--data-dir` | `data/raw` | Directory containing stock data |
| `--output-dir` | `outputs` | Directory to save results |
| `--kmeans-k` | 5 | Number of K-Means clusters |
| `--kmeans-percentile` | 97.5 | Percentile threshold for K-Means |
| `--dbscan-eps` | 0.5 | DBSCAN epsilon parameter |
| `--dbscan-min-samples` | 10 | DBSCAN minimum samples |

### Querying Results

**Query a specific date:**
```bash
python -m src.query --date 2020-03-16
```

**Query a date range:**
```bash
python -m src.query --start 2020-03-01 --end 2020-03-31
```

### Generating Monthly Reports

**Generate a specific month's report:**
```bash
python -m src.monthly --month 2020-03
```

**Generate all monthly reports:**
```bash
python -m src.monthly --all
```

---

## 🔬 Methodology

### Features (Leakage-Free)

All features use rolling windows with `.shift(1)` to ensure no future data is used:

| Feature | Description | Window |
|---------|-------------|--------|
| `ret_z` | Return z-score | 63-day rolling |
| `volz` | Log-volume z-score | 21-day rolling |
| `range_pct` | Intraday range percentile | 63-day rolling |

### Detection Algorithms

1. **K-Means Detector**
   - Clusters data into k groups (default: 5)
   - Flags points with distance > 97.5th percentile within their cluster
   - Trained on 2018 data, applied walk-forward

2. **DBSCAN Detector**
   - Density-based clustering
   - Noise points (cluster = -1) are flagged as anomalies
   - Refitted in walk-forward mode

3. **Rule-Based Detector** (Baseline)
   - Flags if: `|ret_z| > 2.5` OR `volz > 2.5` OR `range_pct > 95`

4. **Market Anomaly Detector**
   - Detects market-wide events
   - Flags if: `|market_ret|` > 95th percentile OR `breadth < 30%`

### Data Splits

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2018 | Fit detectors |
| Validation | 2019 | Tune hyperparameters |
| Test | 2020 Q1 | Final evaluation |

---

## 📊 Output Files

### `full_results.csv`
Complete dataset with all features and detection flags:
- OHLCV data, computed features, anomaly flags, severity scores

### `anomaly_cards.csv`
Filtered anomalies with explanations:
- Date, Ticker, Type (kmeans/dbscan/rule), Feature values, Why column

### `market_days.csv`
Market-level anomaly days:
- Market return, breadth, anomaly flag

### Monthly Reports
Human-readable summaries for each month with:
- Anomaly counts, top events, detector breakdown

---

## 🛠️ Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
kaggle>=1.5.0
scipy>=1.10.0
```

---

## 📝 Example

```python
from src.data_loader import StockDataLoader
from src.features import FeatureEngine
from src.detectors import KMeansDetector, DBSCANDetector

# Load data
loader = StockDataLoader("data/raw")
df = loader.load_universe(['AAPL', 'MSFT', 'AMZN'])

# Compute features
engine = FeatureEngine()
df = engine.compute_features(df)

# Detect anomalies
kmeans = KMeansDetector(n_clusters=5)
kmeans.fit(df[df['Date'].dt.year == 2018])
results = kmeans.predict(df)

# View flagged anomalies
anomalies = results[results['kmeans_anomaly'] == True]
print(anomalies[['Date', 'ticker', 'ret_z', 'volz', 'range_pct']])
```

---

