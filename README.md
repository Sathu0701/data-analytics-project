# AlphaPulse — Investment Risk & Volatility Monitor

> A premium, real-time financial analytics dashboard built with **Plotly Dash**, featuring Monte Carlo simulation, Value at Risk (VaR), rolling volatility, and portfolio correlation analysis.

---

## 📸 Dashboard Preview

The AlphaPulse dashboard provides an interactive, dark-themed interface for institutional-grade risk analytics across a 10-stock diversified portfolio.

---

## 🚀 Features

| Module | Description |
|---|---|
| 📈 **Price Chart** | Normalised price trends (base = 100) for multi-stock comparison |
| 📊 **Trading Volume** | 60-day volume bar chart for the lead ticker |
| 📉 **Daily % Returns** | 120-day daily return line chart |
| 🎲 **Monte Carlo** | GBM simulation (10,000–50,000 runs, 1-year horizon) with histogram & KDE |
| 🔥 **Correlation Heatmap** | Pairwise Pearson correlation matrix using NumPy matrix multiplication |
| 🌊 **Rolling Volatility** | 30-day rolling annualised standard deviation |
| ⚠️ **VaR Panel** | Historical VaR at 95% and 99% confidence levels per ticker |

---

## 🗂️ Project Structure

```
Data-Analytics_Project-2/
├── app.py              # Main Dash application & all callbacks
├── analytics.py        # Core analytics engine (NumPy/SciPy)
├── data_fetcher.py     # yfinance data fetching with local CSV cache
├── requirements.txt    # Python dependencies
├── assets/
│   └── custom.css      # Custom CSS for the dashboard
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Plotly Dash** — interactive web dashboard
- **yfinance** — Yahoo Finance market data
- **NumPy / SciPy** — high-performance analytics
- **Pandas** — data manipulation
- **Dash Bootstrap Components** — UI layout

---

## ⚙️ Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/SomashekarM2002/Data-Analytics_Project-2.git
cd Data-Analytics_Project-2
```

### 2. Create a virtual environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the dashboard
```bash
python app.py
```

Open your browser at 👉 **http://127.0.0.1:8050**

---

## 📦 Portfolio Tickers

The default portfolio covers **10 stocks** across sectors:

| Ticker | Company | Sector |
|---|---|---|
| AAPL | Apple Inc. | Technology |
| MSFT | Microsoft Corp. | Technology |
| GOOGL | Alphabet Inc. | Technology |
| AMZN | Amazon.com Inc. | Consumer Discretionary |
| JPM | JPMorgan Chase | Financials |
| GS | Goldman Sachs | Financials |
| XOM | ExxonMobil Corp. | Energy |
| JNJ | Johnson & Johnson | Healthcare |
| TSLA | Tesla Inc. | Industrials / EV |
| NVDA | NVIDIA Corp. | Semiconductors |

---

## 📐 Analytics Methodology

### Log Returns
```
r_t = ln(P_t / P_{t-1})
```

### Value at Risk (Historical Simulation)
```
VaR_α = -quantile(returns, 1 - α)
```

### Monte Carlo (Geometric Brownian Motion)
```
S_t = S_0 · exp[(μ - σ²/2)·dt + σ·√dt·Z]
```
where Z ~ N(0,1) with ≥ 10,000 simulation paths.

### Portfolio Variance
```
σ²_p = wᵀ · Σ · w
```

---

## 👤 Author

**Somashekar M**
- GitHub: [@SomashekarM2002](https://github.com/SomashekarM2002)
- Email: somashekarmsoma1@gmail.com

---

## 🏢 Built for

**Zaalima Development Pvt. Ltd.**

---

## 📄 License

This project is for educational and demonstration purposes.
