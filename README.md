# quantitative-trading-
# 🔬 MOMENT: Methodology & Technical Implementation

**A Professional Quantitative Stock Prediction System for Indian Markets (NSE)**

**Project:** BTP (Bachelor's Thesis Project)  
**Author:** Aman  
**Version:** 2.3.0  
**Last Updated:** December 2025

---

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Tech Stack](#tech-stack)
3. [Data Pipeline](#data-pipeline)
4. [Feature Engineering](#feature-engineering)
5. [Label Generation](#label-generation)
6. [Model Training](#model-training)
7. [Ensemble Architecture](#ensemble-architecture)
8. [Inference & Decision Engine](#inference--decision-engine)
9. [Sentiment Analysis](#sentiment-analysis)
10. [API & Deployment](#api--deployment)
11. [Model Performance](#model-performance)
12. [Validation & Quality Assurance](#validation--quality-assurance)

---

## 🏗️ System Overview

**MOMENT** is a production-grade quantitative trading system that combines:

- **5 Distinct Machine Learning Models** for robust stock predictions
- **Multi-timeframe Trading Signals** (5-day scalping, 20-day swing, 90-day investing)
- **AI-Powered Insights** using Groq API (Llama-3)
- **Live Market Data** via Yahoo Finance (yfinance)
- **Sentiment Analysis** from financial news (MoneyControl, Economic Times)
- **FastAPI Backend** for real-time predictions
- **Dark-Mode Web Dashboard** for trader-friendly visualization

### 🎯 Trading Horizons

| Horizon | Strategy                | Model Focus                       | Signal Types         |
| ------- | ----------------------- | --------------------------------- | -------------------- |
| 5 days  | **Scalping**            | LightGBM (High Sensitivity)       | BUY_SHORT, SHORT_5D  |
| 20 days | **Swing Trading**       | XGBoost (Trend Capture)           | BUY_SWING, SHORT_20D |
| 90 days | **Long-Term Investing** | Random Forest (Stability)         | BUY_LONG, SELL       |
| Risk    | **Position Sizing**     | Logistic Regression (Risk Factor) | HIGH_RISK, AVOID     |

---

## 💻 Tech Stack

### Backend & Core

| Component           | Technology     | Purpose                                       |
| ------------------- | -------------- | --------------------------------------------- |
| **API Framework**   | FastAPI 0.104+ | Production-grade REST API with async support  |
| **Server**          | Uvicorn        | ASGI server for FastAPI                       |
| **Language**        | Python 3.10+   | Core system implementation                    |
| **Data Processing** | Pandas, NumPy  | DataFrame manipulation, numerical computation |
| **Serialization**   | Joblib         | Model persistence & compression               |

### Machine Learning & Deep Learning

| Component              | Technology         | Purpose                                   |
| ---------------------- | ------------------ | ----------------------------------------- |
| **ML Library**         | Scikit-Learn       | Preprocessing, model training, evaluation |
| **Gradient Boosting**  | LightGBM, XGBoost  | Tree-based ensemble models                |
| **Classical ML**       | Scikit-Learn       | Random Forest, Logistic Regression        |
| **Deep Learning**      | PyTorch            | LSTM, Transformer models (future)         |
| **Model Acceleration** | MPS (Metal) / CUDA | GPU acceleration for training             |

### Data & Market

| Component          | Technology            | Purpose                               |
| ------------------ | --------------------- | ------------------------------------- |
| **Market Data**    | yfinance              | Live OHLCV data for NSE stocks        |
| **Sentiment**      | Groq API + DistilBERT | Financial news sentiment              |
| **News Feeds**     | feedparser (RSS)      | MoneyControl, Economic Times scraping |
| **Stock Universe** | niftystocks library   | NIFTY50, NIFTY500 constituents        |

### Infrastructure

| Component       | Technology       | Purpose                             |
| --------------- | ---------------- | ----------------------------------- |
| **Database**    | CSV (Local)      | Feature store for processed data    |
| **Storage**     | Parquet, pickle  | Compressed model & dataset storage  |
| **Logging**     | Python logging   | System diagnostics & audit trails   |
| **Environment** | python-dotenv    | Configuration management (API keys) |
| **Testing**     | pytest, unittest | Unit & integration tests            |

### Frontend

| Component        | Technology                | Purpose                            |
| ---------------- | ------------------------- | ---------------------------------- |
| **UI Framework** | HTML5 + CSS3 + JavaScript | Dark-mode dashboard                |
| **Charts**       | Chart.js or Plotly        | Real-time prediction visualization |
| **Styling**      | Bootstrap 5 / Tailwind    | Responsive design                  |

---

## 📊 Data Pipeline

### Step 1: Raw Data Collection

**Objective:** Download historical OHLCV data for 50+ stocks + indices

**Process:**

```
┌─────────────────────────────────────────────────────────────┐
│  1. DATA COLLECTION (data_pipeline.py)                      │
├─────────────────────────────────────────────────────────────┤
│  INPUT:                                                      │
│  - NIFTY50, NIFTY500, SENSEX constituents                   │
│  - Date Range: 2000-01-01 to Present                        │
│  - Source: Yahoo Finance (yfinance API)                     │
│                                                              │
│  PROCESS:                                                    │
│  1. Get 50+ ticker symbols via niftystocks                  │
│  2. Download OHLCV data (8 workers, with retries)           │
│  3. Handle rate limiting (1.8x exponential backoff)         │
│  4. Validate data quality (no NaNs, duplicates)             │
│  5. Save individual CSVs to data/raw/yahoo/stocks/          │
│  6. Generate manifest.csv (metadata)                        │
│                                                              │
│  OUTPUT:                                                     │
│  - Individual stock CSVs (e.g., RELIANCE.NS.csv)            │
│  - Index CSVs (NIFTY50, NIFTYBANK, etc.)                    │
│  - manifest.csv with data quality stats                     │
└─────────────────────────────────────────────────────────────┘
```

**Key Parameters:**

- **Max Workers:** 8 (balance speed vs. throttling)
- **Retry Attempts:** 6 (exponential backoff with jitter)
- **Date Range:** 2000-01-01 to 2025-01-01 (~25 years)
- **Timeout:** 30 seconds per request

**File Structure:**

```
data/
├── raw/yahoo/
│   ├── stocks/
│   │   ├── RELIANCE.NS.csv
│   │   ├── HDFCBANK.NS.csv
│   │   └── ... (50+ stocks)
│   ├── indices/
│   │   ├── NIFTY50.csv
│   │   ├── NIFTYBANK.csv
│   │   └── ... (8 indices)
│   └── manifest.csv
└── ...
```

### Step 2: Data Cleaning & Alignment

**Objective:** Solve the "Ragged Data" problem (stocks have holidays, listing/delisting dates)

**Process:**

```
┌──────────────────────────────────────────────────────────────┐
│  2. DATA ALIGNMENT (clean_and_align.py)                      │
├──────────────────────────────────────────────────────────────┤
│  PROBLEM:                                                     │
│  - Each stock has different trading days (holidays differ)   │
│  - Stocks listed/delisted at different times                 │
│  - ML models need aligned, rectangular data                  │
│                                                               │
│  SOLUTION:                                                    │
│  1. Establish Master Calendar from NIFTY50                   │
│  2. Reindex each stock to Master Calendar                    │
│  3. Forward-fill gaps (max 5 days) - holiday handling       │
│  4. Preserve first_valid_date (no pre-listing fill)         │
│  5. Drop stocks with >10% missing data                       │
│  6. Standardize columns: Date, Open, High, Low, Close, Vol  │
│                                                               │
│  OUTPUT:                                                      │
│  - Aligned CSVs (100% rectangular) → data/processed/aligned  │
│  - Quality report (stocks kept/dropped)                      │
└──────────────────────────────────────────────────────────────┘
```

**Quality Thresholds:**

- **Max Missing Data:** 10% (relative to stock's lifespan)
- **Max Forward-Fill Gap:** 5 consecutive days
- **Min Trading History:** 500 days

---

## 🔧 Feature Engineering

### Step 3: Technical Indicator Calculation

**Objective:** Extract 50+ financial features from raw OHLCV data

**Process:**

```
┌──────────────────────────────────────────────────────────────┐
│  3. FEATURE ENGINEERING (feature_engineering.py)             │
├──────────────────────────────────────────────────────────────┤
│  INPUT: Aligned stock data (Date, OHLCV)                    │
│                                                               │
│  FEATURE CATEGORIES:                                         │
│                                                               │
│  A) TREND INDICATORS (capture direction)                     │
│     • SMA (20, 50, 200) - Simple Moving Average              │
│     • EMA (12, 26) - Exponential Moving Average              │
│     • MACD (12-26-9) - Momentum                              │
│     • ADX (14) - Trend Strength                              │
│     • Ichimoku (9-26-52) - Support/Resistance               │
│     • Vortex (14) - Directional Movement                     │
│                                                               │
│  B) MOMENTUM INDICATORS (capture velocity)                   │
│     • RSI (14) - Relative Strength Index                     │
│     • ROC (10, 20) - Rate of Change                          │
│     • TRIX (15) - Triple Exponential Moving Average          │
│     • TSI - True Strength Index                              │
│     • CCI (20) - Commodity Channel Index                     │
│                                                               │
│  C) VOLATILITY INDICATORS (capture risk)                     │
│     • ATR (14) - Average True Range                          │
│     • Bollinger Bands (20, 2σ) - BB Width, %B               │
│     • Keltner Channel (20, 2) - Channel Width                │
│     • Donchian Channel (20) - Range                          │
│                                                               │
│  D) VOLUME INDICATORS (capture strength)                     │
│     • OBV (On Balance Volume) - Cumulative volume            │
│     • MFI (14) - Money Flow Index                            │
│     • Volume MA (20) - Smoothed volume                       │
│                                                               │
│  E) RETURNS & CROSS-ASSETS                                   │
│     • Return_5d, Return_20d, Return_90d                      │
│     • Stock vs. Index correlation                            │
│     • Sector rotation (NIFTYBANK relative strength)          │
│                                                               │
│  F) SENTIMENT (news-based)                                   │
│     • Daily sentiment score (-1 to +1)                       │
│     • Sentiment volatility                                   │
│     • News frequency                                         │
│                                                               │
│  OUTPUT:                                                      │
│  - 50+ features per stock per day                            │
│  - Saved to: data/processed/features/                        │
│  - Format: CSV with DatetimeIndex                            │
└──────────────────────────────────────────────────────────────┘
```

**Top 10 Most Important Features** (by average importance):

| Rank | Feature          | Importance | Type        |
| ---- | ---------------- | ---------- | ----------- |
| 1    | BB_Width         | 0.158      | Volatility  |
| 2    | Vortex_Pos       | 0.036      | Trend       |
| 3    | HDFCBANK_Close   | 0.029      | Cross-Asset |
| 4    | Mass_Index       | 0.028      | Volatility  |
| 5    | BANKBARODA_Close | 0.026      | Cross-Asset |
| 6    | ATR_14           | 0.024      | Volatility  |
| 7    | Vortex_Neg       | 0.021      | Trend       |
| 8    | AXISBANK_Close   | 0.020      | Cross-Asset |
| 9    | Keltner_Lower    | 0.018      | Volatility  |
| 10   | Donchian_Lower   | 0.018      | Volatility  |

**Insight:** Volatility-based features dominate (Bollinger Bands, ATR, Keltner), followed by cross-asset correlations within banking sector.

---

## 🎫 Label Generation

### Step 4: Dynamic Target Labels

**Objective:** Generate 8 trading signals based on future returns and ATR volatility

**Process:**

```
┌─────────────────────────────────────────────────────────────┐
│  4. LABEL GENERATION (dataset_builder_v5.py)               │
├─────────────────────────────────────────────────────────────┤
│  INPUT: Features + 5-day, 20-day, 90-day forward returns   │
│                                                             │
│  DYNAMIC THRESHOLDS (per stock, based on ATR):            │
│                                                             │
│  short_term_mult  = 1.5x ATR   (5-day volatility)         │
│  swing_mult       = 3.0x ATR   (20-day volatility)        │
│  long_term_mult   = 6.0x ATR   (90-day volatility)        │
│  high_risk_mult   = 5% (if ATR/Price > 5%, too risky)    │
│                                                             │
│  LABEL GENERATION LOGIC:                                   │
│                                                             │
│  For each timestamp (t):                                   │
│    1. Calculate 5-day, 20-day, 90-day forward returns     │
│    2. Get stock's ATR (volatility measure)                 │
│    3. Calculate per-stock threshold = ATR × multiplier    │
│    4. Compare return against threshold                     │
│                                                             │
│  LABELS (8 classes):                                       │
│                                                             │
│  LONG SIGNALS:                                             │
│  • BUY_SHORT  → Return_5d > 1.5×ATR  (+ brokerage)       │
│  • BUY_SWING  → Return_20d > 3.0×ATR (+ brokerage)       │
│  • BUY_LONG   → Return_90d > 6.0×ATR (+ brokerage)       │
│                                                             │
│  SHORT SIGNALS:                                            │
│  • SHORT_5D   → Return_5d < -1.5×ATR (short opportunity)  │
│  • SHORT_20D  → Return_20d < -3.0×ATR                     │
│                                                             │
│  EXIT SIGNALS:                                             │
│  • SELL       → Existing position exit signal              │
│                                                             │
│  NEUTRAL/RISK:                                             │
│  • AVOID      → Return between -threshold and +threshold  │
│  • HIGH_RISK  → If ATR/Price > 5% (too volatile)         │
│                                                             │
│  OUTPUT:                                                    │
│  - Dataset with 8-class labels                             │
│  - Format: Parquet (compressed)                            │
│  - 80/10/10 split (train/val/test)                        │
│  - Time-series aware split (no future leakage)            │
└─────────────────────────────────────────────────────────────┘
```

**Example Label Assignment:**

```
Date: 2023-01-15
Stock: RELIANCE
Close: ₹2800
Return_5d: +2.5%
ATR_14: ₹50

Threshold_5d = 50 × 1.5 = ₹75
Expected Move = ₹2800 × 1.5% = ₹42

Since 42 < 75 (below threshold):
Label = AVOID (not enough expected profit to justify transaction costs)

---

Alternative Scenario:
Return_5d: +5.5%
Expected Move = ₹2800 × 5.5% = ₹154

Since 154 > 75:
Label = BUY_SHORT ✅ (good risk-reward for 5-day trade)
```

**Label Distribution Characteristics:**

- **Class Balance:** AVOID is most common (~60%), reduces overfitting risk
- **Temporal Structure:** Preserved in train/val/test split
- **Cost-Adjusted:** Accounts for brokerage (0.05%), slippage (0.1%), taxes (0.5%)

---

## 🤖 Model Training

### Step 5: Multi-Model Ensemble Training

**Objective:** Train 4 complementary models optimized for different aspects

**Process:**

```
┌────────────────────────────────────────────────────────────────┐
│  5. MODEL TRAINING (train_models_v5.py)                       │
├────────────────────────────────────────────────────────────────┤
│  MODELS TRAINED:                                              │
│                                                                │
│  Model 1: LightGBM (Gradient Boosting)                        │
│  ├─ Use Case: Fast, high sensitivity for scalping            │
│  ├─ Params:                                                   │
│  │  • num_leaves: 63 (complex trees)                          │
│  │  • learning_rate: 0.03 (slow, stable learning)            │
│  │  • feature_fraction: 0.8 (random subsampling)             │
│  │  • L1/L2 regularization: 0.1 (prevent overfitting)        │
│  ├─ Training Time: ~30 min                                    │
│  └─ Accuracy: 49.3%                                           │
│                                                                │
│  Model 2: XGBoost (Gradient Boosting)                         │
│  ├─ Use Case: Trend capture for swing trading                │
│  ├─ Params:                                                   │
│  │  • max_depth: 8 (moderate tree depth)                     │
│  │  • subsample: 0.8 (row subsampling)                       │
│  │  • colsample_bytree: 0.8 (column subsampling)            │
│  │  • tree_method: hist (fast histogram)                     │
│  ├─ Training Time: ~20 min                                    │
│  └─ Accuracy: 47.2%                                           │
│                                                                │
│  Model 3: Random Forest (Ensemble of Trees)                  │
│  ├─ Use Case: Stable long-term predictions                   │
│  ├─ Params:                                                   │
│  │  • n_estimators: 300 (deep ensemble)                      │
│  │  • max_depth: 15 (moderate depth)                         │
│  │  • max_features: sqrt (feature randomization)            │
│  │  • min_samples_leaf: 10 (regularization)                 │
│  ├─ Training Time: ~15 min                                    │
│  └─ Accuracy: 41.5%                                           │
│                                                                │
│  Model 4: Logistic Regression (Linear)                        │
│  ├─ Use Case: Risk factor (macro-level prediction)           │
│  ├─ Params:                                                   │
│  │  • max_iter: 500                                           │
│  │  • solver: lbfgs (stable)                                 │
│  │  • C: 1.0 (regularization strength)                       │
│  ├─ Training Time: ~5 min                                     │
│  └─ Accuracy: 48.7%                                           │
│                                                                │
│  TRAINING PROCESS:                                            │
│  1. Load train/val/test data (80/10/10)                      │
│  2. Standardize features (StandardScaler)                    │
│  3. Impute missing values (SimpleImputer)                    │
│  4. Compute class weights (handle imbalance)                 │
│  5. Train with Time-Series Cross-Validation (no leakage)    │
│  6. Evaluate on val/test sets                                │
│  7. Save models to Models_V5/                                │
│  8. Save scalers, encoders, feature list                     │
│                                                                │
│  OUTPUT:                                                       │
│  - Models_V5/short_term/lightgbm.txt (LGB model)            │
│  - Models_V5/swing/xgboost.json (XGB model)                 │
│  - Models_V5/long_term/random_forest.pkl (RF model)         │
│  - Models_V5/risk/logistic_regression.pkl (LR model)        │
│  - Models_V5/scaler.pkl (StandardScaler)                    │
│  - Models_V5/label_encoder.pkl (Label encoding)             │
│  - Models_V5/reports/evaluation_results.json (metrics)      │
└────────────────────────────────────────────────────────────────┘
```

### Hyperparameter Optimization

**LightGBM Parameters:**

```python
{
    'boosting_type': 'gbdt',
    'metric': 'multi_logloss',
    'num_leaves': 63,
    'learning_rate': 0.03,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 50,
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 0.1, # L2 regularization
}
```

**XGBoost Parameters:**

```python
{
    'objective': 'multi:softprob',
    'max_depth': 8,
    'learning_rate': 0.03,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 50,
    'tree_method': 'hist',  # Fast
}
```

**Random Forest Parameters:**

```python
{
    'n_estimators': 300,
    'max_depth': 15,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'max_features': 'sqrt',
}
```

**Cross-Validation:** Time-Series Split (prevents future data leakage)

---

## 🔀 Ensemble Architecture

### Step 6: Weighted Voting Ensemble

**Objective:** Combine 4 models with dynamic weights for robust predictions

**Process:**

```
┌─────────────────────────────────────────────────────────────────┐
│  6. ENSEMBLE PREDICTION (ensemble_predict.py)                  │
├─────────────────────────────────────────────────────────────────┤
│  INPUT: Stock features (50+)                                  │
│                                                                 │
│  STEP 1: Get Individual Predictions                           │
│  ├─ LightGBM  → Prediction_1 (class probabilities)           │
│  ├─ XGBoost   → Prediction_2 (class probabilities)           │
│  ├─ RF        → Prediction_3 (class probabilities)           │
│  └─ LogReg    → Prediction_4 (class probabilities)           │
│                                                                 │
│  STEP 2: Apply Model Weights (based on performance)          │
│  ├─ LightGBM:   Weight = 3 (49.3% accuracy - best)          │
│  ├─ XGBoost:    Weight = 3 (47.2%)                           │
│  ├─ Random Forest: Weight = 2 (41.5% - most conservative)   │
│  └─ LogReg:     Weight = 2 (48.7% - risk signals)           │
│                                                                 │
│  STEP 3: Weighted Average of Probabilities                   │
│  weighted_probs[class] = Σ(weight_i × prob_i) / Σ(weights)  │
│                                                                 │
│  STEP 4: Pick Argmax Class                                    │
│  final_prediction = argmax(weighted_probs)                    │
│                                                                 │
│  STEP 5: Calculate Confidence                                 │
│  confidence = max(weighted_probs) (0 to 1)                   │
│                                                                 │
│  OUTPUT:                                                        │
│  {                                                              │
│    'prediction': 'BUY_SHORT',                                 │
│    'confidence': 0.72,                                        │
│    'probabilities': {...},  # per-class                      │
│    'individual_votes': {                                      │
│      'lightgbm': 'BUY_SHORT',                                 │
│      'xgboost': 'AVOID',                                      │
│      'random_forest': 'BUY_SHORT',                            │
│      'logistic_regression': 'BUY_SHORT'                       │
│    }                                                            │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

**Weighted Voting Logic:**

- **Higher Accuracy = Higher Weight** (performance-based)
- **Diversity** ensures different aspects are captured:
  - LightGBM: Sensitive, fast-changing trends
  - XGBoost: Stable, medium-term trends
  - Random Forest: Conservative, long-term stability
  - Logistic Regression: Macro risk factors

---

## ⚙️ Inference & Decision Engine

### Step 7: Multi-Timeframe Decision Logic

**Objective:** Convert ensemble predictions into actionable trading decisions

**Process:**

```
┌──────────────────────────────────────────────────────────────────┐
│  7. DECISION ENGINE (decision_engine.py)                        │
├──────────────────────────────────────────────────────────────────┤
│  INPUT: Ensemble predictions for each timeframe                │
│    predictions = {                                              │
│      'lightgbm': 'BUY_SHORT',                                  │
│      'xgboost': 'AVOID',                                       │
│      'random_forest': 'BUY_LONG',                              │
│      'logistic_regression': 'SELL'                             │
│    }                                                             │
│                                                                  │
│  DECISION RULES (Model-Specific Authority):                   │
│                                                                  │
│  ┌─ SCALP STRATEGY (5 days)                                   │
│  │  Boss: LightGBM (sensitive to short-term moves)            │
│  │  Veto Power: Logistic Regression (risk officer)            │
│  │                                                              │
│  │  Logic:                                                      │
│  │  IF LightGBM says "BUY":                                    │
│  │    IF LogReg says "SELL": → Signal = AVOID (veto)         │
│  │    ELSE: → Signal = BUY ✅                                 │
│  │  ELSE IF LightGBM says "SHORT": → Signal = SHORT          │
│  │  ELSE: → Signal = WAIT                                      │
│  │                                                              │
│  │  Output: {'scalp': {'signal': 'BUY', 'reason': '...'}}    │
│  │                                                              │
│  ├─ SWING STRATEGY (20 days)                                  │
│  │  Boss: XGBoost (captures medium trends)                     │
│  │  Confirmation: Random Forest (must agree)                   │
│  │                                                              │
│  │  Logic:                                                      │
│  │  IF XGBoost says "BUY":                                     │
│  │    IF RF says "BUY": → Signal = BUY ✅ (strong trend)      │
│  │    ELIF RF says "SELL": → Signal = WAIT (conflict)        │
│  │    ELSE: → Signal = BUY (possible trend)                   │
│  │                                                              │
│  ├─ INVEST STRATEGY (90 days)                                 │
│  │  Boss: Random Forest (stable, long-term)                    │
│  │  Veto: Logistic Regression (macro risk)                    │
│  │                                                              │
│  │  Logic:                                                      │
│  │  IF RF says "BUY":                                          │
│  │    IF LogReg says "SELL": → Signal = WAIT (high risk)     │
│  │    ELSE: → Signal = BUY ✅                                 │
│  │                                                              │
│  └─ MASTER SIGNAL                                              │
│     Combines all 3 timeframes for consensus:                  │
│     IF 2+ agree on direction: → MASTER = that direction      │
│     ELSE: → MASTER = SIDEWAYS (no consensus)                  │
│                                                                  │
│  OUTPUT:                                                         │
│  {                                                               │
│    'scalp': {'signal': 'AVOID', 'reason': '⛔ Risk Veto'},    │
│    'swing': {'signal': 'BUY', 'reason': '✅ Strong Trend'},   │
│    'invest': {'signal': 'WAIT', 'reason': '⚠️ Macro Risk'},  │
│    'master': 'SIDEWAYS',                                        │
│    'timestamp': '2025-01-15T10:30:00Z'                        │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## 🗞️ Sentiment Analysis

### Step 8: News-Based Sentiment Integration

**Objective:** Capture market sentiment from financial news to enhance predictions

**Process:**

```
┌────────────────────────────────────────────────────────────────────┐
│  8. SENTIMENT ANALYSIS (process_news.py)                          │
├────────────────────────────────────────────────────────────────────┤
│  DATA SOURCES:                                                     │
│  • MoneyControl RSS feed (business news)                           │
│  • Economic Times RSS feed (stock-specific news)                   │
│  • Custom IndianFinancialNews.csv (optional, historical)           │
│                                                                     │
│  SENTIMENT EXTRACTION PIPELINE:                                    │
│                                                                     │
│  1. RSS FEED PARSING (feedparser)                                 │
│     Input: RSS URL                                                 │
│     ├─ Extract: title, summary, published_date, link             │
│     └─ Filter: Only articles from last 24 hours                   │
│                                                                     │
│  2. STOCK MAPPING (keyword matching)                              │
│     For each article:                                              │
│     ├─ Search for stock keywords (case-insensitive)              │
│     ├─ Example: "HDFC Bank profit surge"                          │
│     │   → Matches keywords ['hdfc bank', 'hdfc'] → HDFCBANK     │
│     └─ Multiple matches allowed (sector news)                     │
│                                                                     │
│  3. SENTIMENT CLASSIFICATION                                       │
│     Two options:                                                   │
│                                                                     │
│     Option A: GROQ API (LLM-based, best accuracy)                │
│     ├─ Prompt: "Classify this news as Positive/Neutral/Negative" │
│     ├─ Model: Llama-3 (via Groq API)                             │
│     ├─ Cost: $0.0005 per article                                 │
│     └─ Accuracy: ~92%                                             │
│                                                                     │
│     Option B: DistilBERT (local, faster, no cost)                │
│     ├─ Pretrained: "distilbert-base-uncased"                     │
│     ├─ Fine-tuned on: 26K financial news labeled examples        │
│     ├─ Output: [Negative, Neutral, Positive] probabilities       │
│     └─ Inference: ~50ms per article                              │
│                                                                     │
│  4. DAILY AGGREGATION                                              │
│     Per stock, per day:                                            │
│     ├─ Collect all news articles                                  │
│     ├─ Average sentiment scores: sum / count                      │
│     ├─ Normalize to [-1, 1] range                                 │
│     │   -1.0 = all negative, 0 = neutral, +1.0 = all positive   │
│     └─ Store in Data/Processed/daily_sentiment.csv               │
│                                                                     │
│  5. FEATURE ENGINEERING (from sentiment)                           │
│     ├─ sentiment_score: daily average [-1, +1]                   │
│     ├─ sentiment_volatility: 7-day rolling std                    │
│     ├─ news_frequency: articles per day (momentum)                │
│     └─ sentiment_trend: moving average slope                      │
│                                                                     │
│  OUTPUT:                                                            │
│  data/processed/daily_sentiment_complete.csv:                     │
│  ┌─────────────┬─────────────┬──────────────┬──────────────────┐  │
│  │ Date        │ Ticker      │ sentiment    │ news_frequency   │  │
│  ├─────────────┼─────────────┼──────────────┼──────────────────┤  │
│  │ 2025-01-15  │ RELIANCE    │ +0.45        │ 3                │  │
│  │ 2025-01-15  │ TCS         │ -0.12        │ 1                │  │
│  │ 2025-01-15  │ HDFCBANK    │ +0.78        │ 5                │  │
│  └─────────────┴─────────────┴──────────────┴──────────────────┘  │
│                                                                     │
│  STOCK KEYWORDS MAPPING:                                           │
│  {                                                                  │
│    'RELIANCE': ['reliance', 'ril', 'jio'],                        │
│    'HDFCBANK': ['hdfc bank', 'hdfc'],                             │
│    'TCS': ['tcs', 'tata consultancy'],                            │
│    ... (50+ stocks with custom keywords)                          │
│  }                                                                  │
│                                                                     │
│  SENTIMENT TRAINING (DistilBERT):                                │
│  • Dataset: 26K labeled financial news                            │
│  • Labels: {0: Negative, 1: Neutral, 2: Positive}                │
│  • Train/Test: 80/20 split                                        │
│  • Epochs: 3                                                       │
│  • Batch Size: 32                                                  │
│  • Learning Rate: 2e-5                                             │
│  • Output: Models/sentiment/ (tokenizer + model)                  │
└────────────────────────────────────────────────────────────────────┘
```

---

## 🌐 API & Deployment

### Step 9: FastAPI Production Backend

**Objective:** Serve predictions via REST API with live data integration

**Process:**

```
┌─────────────────────────────────────────────────────────────┐
│  9. API BACKEND (src/api.py)                               │
├─────────────────────────────────────────────────────────────┤
│  FRAMEWORK: FastAPI + Uvicorn                              │
│  PORT: 8000                                                 │
│  DOCS: /docs (Swagger UI)                                  │
│                                                             │
│  CORE ENDPOINTS:                                            │
│                                                             │
│  1. POST /predict/{ticker}                                │
│     Request:                                               │
│     {                                                       │
│       "date": "2025-01-15"  (optional, defaults to today)  │
│     }                                                       │
│     Response:                                              │
│     {                                                       │
│       "ticker": "RELIANCE",                                │
│       "date": "2025-01-15",                                │
│       "ensemble_prediction": "BUY_SHORT",                  │
│       "confidence": 0.72,                                  │
│       "decisions": {                                       │
│         "scalp": {"signal": "BUY", "reason": "..."},      │
│         "swing": {"signal": "AVOID", "reason": "..."},    │
│         "invest": {"signal": "BUY", "reason": "..."},     │
│         "master": "BUY"                                    │
│       },                                                    │
│       "features_used": {                                   │
│         "BB_Width": 2.45,                                  │
│         "ATR_14": 50.2,                                    │
│         "RSI_14": 67.3,                                    │
│         ...                                                 │
│       },                                                    │
│       "live_price": 2850.50,                               │
│       "timestamp": "2025-01-15T10:30:00Z"                 │
│     }                                                       │
│                                                             │
│  2. GET /market-report                                    │
│     Returns top 10 buy, sell, and avoid signals            │
│                                                             │
│  3. GET /stocks                                            │
│     Returns list of supported tickers                      │
│                                                             │
│  4. GET /health                                            │
│     System health check                                     │
│                                                             │
│  LIVE DATA INTEGRATION:                                    │
│  • yfinance: Get live stock prices                         │
│  • feedparser: Real-time RSS feeds                         │
│  • Groq API: Sentiment analysis                            │
│                                                             │
│  CACHING:                                                   │
│  • Predictions cached for 1 hour                           │
│  • Models loaded at startup (not per-request)              │
│                                                             │
│  MIDDLEWARE:                                               │
│  • CORS: Allow cross-origin requests (web dashboard)       │
│  • Error Handling: 500s converted to readable JSON         │
│                                                             │
│  PRODUCTION DEPLOYMENT:                                    │
│  Command: uvicorn src.api:app --host 0.0.0.0 --port 8000 │
└─────────────────────────────────────────────────────────────┘
```

### Model Persistence & Loading

**Challenge:** Large Random Forest model (~2.5GB) exceeds GitHub limits

**Solution:** Chunking + Automatic Reassembly

```python
# Chunk the model into parts:
# rf_part_aa (500MB)
# rf_part_ab (500MB)
# ... (5 parts total)

# On API startup, reassemble automatically:
if not rf_model.pkl.exists():
    chunks = sorted(glob("rf_part_*"))
    with open("rf_model.pkl", "wb") as out:
        for chunk in chunks:
            out.write(open(chunk, "rb").read())
```

---

## 📈 Model Performance

### Evaluation Metrics (Test Set)

**Overall Ensemble Performance:**

| Model         | Accuracy   | Balanced Acc | Precision | Recall | F1 Score  |
| ------------- | ---------- | ------------ | --------- | ------ | --------- |
| LightGBM      | **49.29%** | 0.184        | 0.214     | 0.184  | 0.195     |
| XGBoost       | 47.16%     | 0.198        | 0.222     | 0.198  | 0.202     |
| Random Forest | 41.54%     | 0.166        | 0.218     | 0.166  | 0.141     |
| LogReg        | 48.71%     | **0.245**    | 0.223     | 0.245  | **0.230** |
| **Ensemble**  | **~50%**   | -            | -         | -      | -         |

**Key Observations:**

1. **Balanced Models:** No single model dominates; weighted ensemble captures diversity
2. **Class Imbalance:** AVOID is 60% of data; balanced accuracy shows true multi-class performance
3. **Logistic Regression Edge:** Best balanced accuracy (0.245) → good risk detection
4. **LightGBM Sensitivity:** Highest accuracy (49.3%) → good for scalping signals

### Per-Label Performance

| Label              | Precision | Recall | F1   | Support |
| ------------------ | --------- | ------ | ---- | ------- |
| AVOID              | 0.50      | 0.80   | 0.62 | 5,400   |
| BUY_SHORT          | 0.35      | 0.15   | 0.21 | 1,200   |
| SELL               | 0.25      | 0.05   | 0.08 | 800     |
| HIGH_RISK          | 0.20      | 0.02   | 0.04 | 600     |
| ... (other labels) | ...       | ...    | ...  | ...     |

**Insight:** Model is conservative (high AVOID precision), reduces false positives in live trading.

---

## ✅ Validation & Quality Assurance

### Step 10: Data Quality Suite

**Objective:** Ensure data integrity and model reliability

**Process:**

```
┌─────────────────────────────────────────────────────────────┐
│  10. QUALITY ASSURANCE (src/utils/quality_suite/)          │
├─────────────────────────────────────────────────────────────┤
│  VALIDATORS:                                                │
│                                                             │
│  1. Data Completeness (analyzer.py)                        │
│     ✓ No NaN values in features                            │
│     ✓ Date continuity (no gaps > 5 days)                   │
│     ✓ OHLCV relationships (High >= Low, Close in range)   │
│     ✓ Volume > 0 for all days                              │
│                                                             │
│  2. Statistical Validation (validator.py)                  │
│     ✓ Feature distributions are reasonable                 │
│     ✓ Returns follow expected patterns                     │
│     ✓ No extreme outliers (>5σ without cause)             │
│     ✓ Correlation matrix is sensible                       │
│                                                             │
│  3. Label Distribution (reporter.py)                       │
│     ✓ All 8 classes present                                │
│     ✓ Class ratio matches historical frequency             │
│     ✓ Temporal balance (each year similar distribution)    │
│                                                             │
│  4. Model Consistency (run_suite.py)                       │
│     ✓ Predictions are repeatable (same input, same output) │
│     ✓ Confidence scores sum to 1.0                         │
│     ✓ No NaN in predictions                                │
│                                                             │
│  OUTPUT:                                                    │
│  - quality_report.md (human-readable)                      │
│  - plots/ (visualizations)                                 │
│  - CSV export of metrics                                   │
│                                                             │
│  QUALITY GATES:                                            │
│  ✓ Pass: ≥95% data completeness                           │
│  ✓ Pass: All label classes present                         │
│  ✓ Pass: No model prediction errors                        │
│  ✗ Fail: Stop pipeline & alert                            │
└─────────────────────────────────────────────────────────────┘
```

### Backtesting Framework

**Not Implemented Yet** (planned for V3)

But the architecture supports it:

```python
# Pseudo-code
for date in test_set_dates:
    predictions = get_predictions(date)

    for stock, signal in predictions.items():
        entry_price = close_price(stock, date)
        exit_price = close_price(stock, date + horizon)

        if signal in ['BUY_SHORT', 'BUY_SWING', 'BUY_LONG']:
            pnl = (exit_price - entry_price) * quantity
        else:
            pnl = 0

        record_trade(stock, date, signal, pnl)

# Metrics:
# - Total Return
# - Sharpe Ratio
# - Max Drawdown
# - Win Rate
# - Risk-Adjusted Return
```

---

## 🔄 Data Flow Architecture

**End-to-End System Diagram:**

```
┌──────────────────┐
│  Yahoo Finance   │
│  (Live Prices)   │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐      ┌─────────────────┐
│  Data Pipeline           │◄────►│  RSS Feeds      │
│  (data_pipeline.py)      │      │  (News Data)    │
└────────┬─────────────────┘      └─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Data Alignment          │
│  (clean_and_align.py)    │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐      ┌─────────────────┐
│  Feature Engineering     │◄────►│  Sentiment      │
│  (feature_engineering.py)│      │  (process_news) │
└────────┬─────────────────┘      └─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Dataset Builder         │
│  (dataset_builder_v5.py) │
│  (Label Generation)      │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Model Training          │
│  (train_models_v5.py)    │
│  • LightGBM              │
│  • XGBoost               │
│  • Random Forest         │
│  • Logistic Regression   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Ensemble Prediction     │
│  (ensemble_predict.py)   │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Decision Engine         │
│  (decision_engine.py)    │
│  (3 timeframes)          │
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────┐      ┌──────────────────┐
│  FastAPI Backend         │◄────►│  Groq API        │
│  (src/api.py)            │      │  (AI Insights)   │
└────────┬─────────────────┘      └──────────────────┘
         │
         ▼
┌──────────────────────────┐
│  Web Dashboard           │
│  (web/index.html)        │
│  Dark Mode UI            │
└──────────────────────────┘
```

---

## 🛠️ Development & Testing

### Unit Testing

**Test Coverage:**

- `test_data_pipeline.py` - Data download & validation
- `test_feature_engineering.py` - Indicator calculation
- `test_dataset_builder.py` - Label generation logic
- `test_ensemble.py` - Model voting mechanism
- `test_api.py` - REST endpoint testing

### Integration Testing

1. **Full Pipeline:** Raw data → Predictions
2. **Model Stability:** Same input gives consistent predictions
3. **API Response Times:** <500ms per prediction
4. **Error Handling:** Graceful degradation when services down

---

## 📚 Dependencies

### Core Libraries

```
fastapi==0.104+
uvicorn==0.24+
pandas==2.0+
numpy==1.24+
scikit-learn==1.3+
lightgbm==4.0+
xgboost==2.0+
torch==2.0+
yfinance==0.2.32+
joblib==1.3+
feedparser==6.0+
requests==2.31+
python-dotenv==1.0+
transformers==4.30+  # DistilBERT for sentiment
```

---

## 🚀 Future Enhancements

1. **Deep Learning Models:** LSTM, Transformer architectures
2. **Backtesting Engine:** Full historical performance analysis
3. **Portfolio Optimization:** Multi-stock allocation
4. **Options Strategy:** Greeks calculation, IV analysis
5. **Risk Management:** Position sizing, stop-loss automation
6. **Real-Time Alerts:** Email/SMS for trading signals

---

## 📖 References & Sources

- **LightGBM Docs:** https://lightgbm.readthedocs.io
- **XGBoost Docs:** https://xgboost.readthedocs.io
- **DistilBERT:** https://huggingface.co/distilbert-base-uncased
- **Technical Analysis:** https://en.wikipedia.org/wiki/Technical_analysis
- **Yahoo Finance API:** https://github.com/ranaroussi/yfinance
- **Groq API:** https://console.groq.com

---

## 📞 Contact & Support

**Project Lead:** Aman  
**Repository:** github.com/thisisaman408/btp_project  
**License:** Proprietary (Quantitative Trading System)

---

**Last Updated:** December 6, 2025  
**Version:** 2.3.0
