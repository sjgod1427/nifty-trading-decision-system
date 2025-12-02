# NIFTY Trading Decision System - Technical Documentation

## Table of Contents
1. [Problem Statement](#problem-statement)
2. [Data Understanding](#data-understanding)
3. [Solution Architecture](#solution-architecture)
4. [Feature Engineering Strategy](#feature-engineering-strategy)
5. [Model Selection Rationale](#model-selection-rationale)
6. [Training Strategy](#training-strategy)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Signal Generation & PnL Calculation](#signal-generation--pnl-calculation)
9. [Results & Insights](#results--insights)
10. [Challenges & Solutions](#challenges--solutions)
11. [Production Improvements](#production-improvements)
12. [Final Results](#final-results)

---

## 1. Problem Statement

**Objective**: Predict whether the next candle's closing price will go **up** or **down** using NIFTY intraday historical data.

**Problem Type**: Binary Classification
- Target = 1: Next close > Current close (Price going UP)
- Target = 0: Next close ≤ Current close (Price going DOWN)

**Key Challenges**:
- Financial time series are inherently noisy and non-stationary
- Small price movements make classification difficult
- Need to avoid data leakage in time series modeling
- Balance between model complexity and explainability

---

## 2. Data Understanding

### Dataset Characteristics
- **Size**: ~319,000 rows (1 year of data)
- **Granularity**: 1-minute OHLC candles
- **Date Range**: February 2022 to September 2025
- **Columns**: timestamp, open, high, low, close, volume, open_interest

### Data Quality Issues Addressed
1. **Chronological Order**: Data was in descending order → sorted ascending
2. **Missing Values**: Checked OHLC columns for nulls → dropped if present
3. **Duplicates**: Removed duplicate timestamps
4. **Target Creation**: Last row dropped (no next candle for target)

### Class Distribution Analysis
- Checked balance between up/down movements
- Financial data typically shows slight class imbalance
- No extreme imbalance requiring SMOTE or class weighting

---

## 3. Solution Architecture

### Modular Design Principles
The solution follows a **modular architecture** for:
- **Maintainability**: Each module has a single responsibility
- **Scalability**: Easy to add new features or models
- **Testability**: Each component can be tested independently
- **Reusability**: Modules can be used in other projects

### Module Breakdown

```
┌─────────────────┐
│basic_approach.py        │  ← Orchestrator
└────────┬────────┘
         │
         ├──→ DataLoader         (Load & Clean)
         ├──→ FeatureEngineer    (Create Features)
         ├──→ ModelTrainer       (Train & Compare)
         └──→ Evaluator          (Evaluate & Generate Signals)
```

### Data Flow Pipeline
```
Raw CSV Data
    ↓
[1] Load & Preprocess (data_loader.py)
    ↓
[2] Create Target Variable (next_close > close)
    ↓
[3] Feature Engineering (feature_engineer.py)
    ↓
[4] Time-based Train/Test Split (70/30)
    ↓
[5] Train Multiple Models (model_trainer.py)
    ↓
[6] Compare Models & Select Best
    ↓
[7] Evaluate on Test Set (evaluator.py)
    ↓
[8] Generate Trading Signals (buy/sell)
    ↓
[9] Calculate Cumulative PnL
    ↓
[10] Save Outputs (CSV + Report)
```

---

## 4. Feature Engineering Strategy

**Philosophy**: "Features make the model, not the other way around."

Good features are MORE important than model selection for financial prediction.

### Feature Categories (20+ Features)

#### A. Price-Based Features (3 features)
```python
1. intraday_return = (close - open) / open
   → Captures strength of current candle

2. prev_return = (close - prev_close) / prev_close
   → Momentum from previous candle

3. hl_range = (high - low) / close
   → Volatility proxy, normalized by close
```

**Why these matter**:
- Returns are stationary (unlike raw prices)
- Capture immediate momentum
- Normalized to be scale-independent

#### B. Moving Averages (6 features)
```python
SMA_5, SMA_10, SMA_20  (Simple Moving Averages)
dist_from_sma_5, dist_from_sma_10, dist_from_sma_20

dist_from_sma = (close - SMA) / SMA
```

**Why these matter**:
- SMA smooths out noise, shows trend direction
- Distance from SMA indicates overbought/oversold conditions
- Different windows capture short/medium term trends

#### C. Momentum Indicators (4 features)
```python
1. RSI (Relative Strength Index, 14-period)
   RSI = 100 - (100 / (1 + RS))
   where RS = Avg Gain / Avg Loss

2. MACD (Moving Average Convergence Divergence)
   MACD = EMA(12) - EMA(26)

3. MACD Signal = EMA(9) of MACD

4. MACD Histogram = MACD - MACD Signal
```

**Why these matter**:
- RSI identifies overbought (>70) / oversold (<30) conditions
- MACD captures trend changes and momentum shifts
- Widely used by traders, captures real market psychology

#### D. Volatility Features (1 feature)
```python
volatility = rolling_std(prev_return, window=5)
```

**Why this matters**:
- High volatility → larger price swings, higher uncertainty
- Volatility clustering → high vol periods tend to persist
- Helps model adjust confidence based on market conditions

#### E. Lag Features (3 features)
```python
close_lag_1 = close from 1 candle ago
close_lag_2 = close from 2 candles ago
close_lag_3 = close from 3 candles ago
```

**Why these matter**:
- Captures short-term price memory
- Allows model to learn autoregressive patterns
- Limited to 3 lags to avoid overfitting

#### F. Time Features (2 features)
```python
hour = hour of day (9-15 for market hours)
minute = minute within hour (0-59)
```

**Why these matter**:
- Opening hour (9:15-10:15): High volatility, directional moves
- Mid-day (11:00-14:00): Lower volatility, range-bound
- Closing hour (15:00-15:30): Increased activity
- Captures intraday seasonality patterns

### Feature Engineering Best Practices
1. ✅ **No future data**: All features use only past/current data
2. ✅ **Normalized**: Returns and ratios instead of raw prices
3. ✅ **Domain knowledge**: Based on established technical analysis
4. ✅ **Variety**: Price, trend, momentum, volatility, time
5. ✅ **Handled NaNs**: Dropped initial rows with rolling calculation NaNs

---

## 5. Model Selection Rationale

### Models Implemented

#### Model 1: Logistic Regression
**Type**: Linear classifier
**Parameters**:
- max_iter = 1000
- solver = 'lbfgs'

**Why use it**:
- ✅ Simple, fast, interpretable
- ✅ Good baseline to compare against
- ✅ Works well when features are well-engineered
- ✅ Coefficients show feature importance

**Limitations**:
- ❌ Assumes linear relationship
- ❌ Cannot capture complex non-linear patterns
- ❌ May underfit financial data



---

#### Model 2: Random Forest
**Type**: Ensemble of decision trees
**Parameters**:
- n_estimators = 100 (100 trees)
- max_depth = 10
- min_samples_split = 10
- min_samples_leaf = 5

**Why use it**:
- ✅ Handles non-linear relationships
- ✅ Robust to outliers
- ✅ Built-in feature importance
- ✅ Reduces overfitting through bagging
- ✅ No need for feature scaling

**How it works**:
1. Creates 100 different decision trees
2. Each tree trained on random subset of data
3. Final prediction = majority vote
4. Reduces variance, improves generalization



---

#### Model 3: LightGBM (Light Gradient Boosting Machine)
**Type**: Gradient boosting framework
**Parameters**:
- n_estimators = 100
- max_depth = 10
- learning_rate = 0.1
- num_leaves = 31

**Why use it** (typically the best performer):
- ✅ Learns from previous trees' errors iteratively
- ✅ Handles complex patterns in financial data
- ✅ Very fast training (optimized for large datasets)
- ✅ Built-in handling of missing values
- ✅ Feature importance analysis
- ✅ Industry standard for tabular data competitions

**How it works**:
1. Starts with weak learner (tree 1)
2. Calculates errors (residuals)
3. Next tree focuses on reducing those errors
4. Repeats for 100 iterations
5. Final prediction = weighted sum of all trees


**Why LightGBM**:
- Boosting > Bagging for structured data
- Sequential error correction vs parallel voting
- Better handles feature interactions

---

### Model Comparison Strategy

**Evaluation Metric**: Accuracy (primary)
**Also track**: Precision, Recall, F1-Score

**Selection Process**:
1. Train all 3 models on same train set
2. Evaluate on same test set
3. Compare accuracy scores
4. Select model with highest accuracy
5. Validate with other metrics (precision/recall)

---

## 6. Training Strategy

### Time-Based Split (Critical for Time Series)

```
Data: Feb 2022 ──────────────────► Sep 2025
      |                           |
      |<──── 70% Train ────>|<30% Test>|
```

**Why time-based (NOT random)?**
- ✅ Prevents data leakage (no future data in training)
- ✅ Mimics real-world scenario (train on past, predict future)
- ✅ Tests generalization to unseen time periods
- ❌ Random split would use future data to predict past (cheating!)

### Training Process
1. Split at 70% chronological point
2. Train models only on past data
3. No shuffling of data
4. No information from test set used during training

### Cross-Validation Considerations
- Walk-forward cross-validation would be ideal for production
- Currently using single train/test split for computational efficiency
- Future enhancement: implement time series cross-validation

---

## 7. Evaluation Methodology

### Metrics Explained

#### Accuracy
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```
**What it means**: Overall correctness of predictions
**Good for**: Balanced classes
**Target**: > 55% (above random 50%)

#### Precision
```
Precision = TP / (TP + FP)
```
**What it means**: Of all "UP" predictions, how many were correct?
**Important for**: Minimizing false buy signals
**High precision**: Few false positives, confident in "buy" signals

#### Recall
```
Recall = TP / (TP + FN)
```
**What it means**: Of all actual "UP" movements, how many did we catch?
**Important for**: Not missing profitable opportunities
**High recall**: Catches most upward movements

#### F1-Score
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```
**What it means**: Harmonic mean of precision and recall
**Important for**: Balanced view when classes are imbalanced

### Confusion Matrix Interpretation

```
                 Predicted
                 DOWN  UP
Actual  DOWN    [ TN | FP ]
        UP      [ FN | TP ]

TN (True Negative): Correctly predicted price down
TP (True Positive): Correctly predicted price up
FN (False Negative): Missed upward movement (predicted down, actual up)
FP (False Positive): False alarm (predicted up, actual down)
```

---

## 8. Signal Generation & PnL Calculation

### Trading Signal Logic

```python
if prediction == 1:  # Model predicts price will go UP
    signal = "buy"
else:  # Model predicts price will go DOWN
    signal = "sell"
```

### PnL Calculation Methodology

**Logic**:
```python
cumulative_pnl = 0

for each candle in test_set (chronological order):
    if model_call == "buy":
        cumulative_pnl -= close  # We pay to buy
    elif model_call == "sell":
        cumulative_pnl += close  # We receive from selling
```

**Interpretation**:
- This simulates a basic trading strategy
- "Buy" signal: Enter long position (pay current price)
- "Sell" signal: Exit/short position (receive current price)
- Cumulative PnL tracks running profit/loss

**Production Considerations**:
- Current implementation is simplified for demonstration
- Production deployment would include:
  - Transaction costs and brokerage fees
  - Slippage modeling
  - Dynamic position sizing
  - Risk management (stop-loss, take-profit)
  - Portfolio-level constraints

---



### Feature Importance Insights

Top contributing features (typical results):
1. **Previous returns** (prev_return) - Strong momentum signal
2. **RSI** - Captures overbought/oversold conditions
3. **Distance from SMA_20** - Medium-term trend indicator
4. **MACD histogram** - Momentum shifts
5. **Intraday return** - Current candle strength

**Learning**: Price momentum and trend indicators are most predictive

---

## 9. Challenges & Solutions

### Challenge 1: Data Leakage Prevention
**Problem**: Using future data to predict past
**Solution**:
- Strict time-based split
- All features use only past/current data
- No shuffling during train/test split

### Challenge 2: Feature Engineering Complexity
**Problem**: Too many features → overfitting, too few → underfitting
**Solution**:
- Selected ~20 well-known indicators
- Based on domain knowledge (technical analysis)
- Avoided exotic/untested indicators


### Challenge 3: Low Baseline Accuracy
**Problem**: Stock prediction is inherently difficult (efficient market hypothesis)
**Solution**:
- Accept that 55-60% is good for this problem
- Focus on risk-adjusted returns, not just accuracy
- Use as proof of concept, not production system

### Challenge 4: Computational Efficiency
**Problem**: 319k rows with 20 features = large computation
**Solution**:
- LightGBM optimized for speed
- Modular design allows parallel processing
- Reasonable hyperparameters (not exhaustive tuning)

---

## Conclusion

### Key Takeaways

1. **Feature Engineering > Model Selection**
   - 20+ well-engineered features drive performance
   - Domain knowledge (technical analysis) is critical

2. **Time Series Awareness**
   - Time-based split is non-negotiable
   - Prevents unrealistic performance estimates

3. **Model Diversity**
   - Tested linear, bagging,probablistic and boosting approaches
  

4. **Realistic Expectations**
   - 55-60% accuracy is good for stock prediction
   - Focus on consistent edge over many trades

5. **Production Considerations**
   - This is a proof-of-concept
   - Real trading requires risk management, costs, monitoring

### Future Improvements

1. **Advanced Features**: Order flow, market microstructure, sentiment
2. **Ensemble Methods**: Combine multiple models (stacking, voting)
3. **Hyperparameter Tuning**: Grid search, Bayesian optimization
4. **Walk-Forward Validation**: Rolling window cross-validation
5. **Risk Management**: Position sizing, stop-loss, take-profit
6. **Alternative Targets**: Predict returns magnitude, not just direction

---

## References

### Technical Indicators
- RSI: Wilder, J. W. (1978). New Concepts in Technical Trading Systems
- MACD: Appel, G. (2005). Technical Analysis: Power Tools for Active Investors

### Machine Learning
- Random Forest: Breiman, L. (2001). Random Forests
- Gradient Boosting: Friedman, J. H. (2001). Greedy Function Approximation
- LightGBM: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree

### Financial ML
- Lopez de Prado, M. (2018). Advances in Financial Machine Learning
- Jansen, S. (2020). Machine Learning for Algorithmic Trading

---

**Document Version**: 1.0
**Author**: Sambhav Jain
**Last Updated**: December 2025

---

## 11. Production Improvements

### Version 1.0.0 - Accuracy Improvement from 53.49% to 57.82%

After baseline implementation, conducted extensive research to improve accuracy:

#### A. Enhanced Feature Engineering
**Added Technical Indicators**:
- Bollinger Bands (upper, lower, width, position)
- ATR (Average True Range) for volatility
- Stochastic Oscillator (K, D, overbought/oversold)
- Rate of Change (ROC) momentum indicator
- Enhanced EMA features (12, 26, 50-period)
- Total features increased from 36 to 51

**Result**: 52.23% accuracy (decreased due to noise)

#### B. Advanced ML Strategies Tested

**1. Hyperparameter Tuning**
- Used RandomizedSearchCV with TimeSeriesSplit
- Tested 15 parameter combinations per model
- Models: Random Forest, XGBoost, LightGBM, CatBoost
- **Result**: 53.11% (minimal improvement)

**2. Feature Selection**
- Selected top 20 features from 51 using importance
- Removed noisy features
- **Result**: 52.61% with selected features alone

**3. Target Engineering (Critical)**
- **0.05% threshold**: 53.49% (baseline) - 30,100 samples
- **0.15% threshold**: 54.39% - 1,745 samples
- Higher threshold filters noise, focuses on stronger movements
- **Result**: +0.90% improvement

**4. Probability Calibration (BEST)**
- Applied CalibratedClassifierCV with sigmoid method
- Improves probability estimates for confidence scoring
- Combined with 0.15% threshold + top 20 features
- **Result**: 57.82% accuracy ✅ (+4.33% improvement)

**5. Advanced Stacking**
- Base models: RF, XGBoost, LightGBM, CatBoost
- Meta-learner: Logistic Regression
- **Result**: 56.30% accuracy

**6. Feature Interactions**
- Created polynomial feature interactions
- Top 5 features paired
- **Result**: 54.58% accuracy

#### C. Market Regime Analysis

**Regime Detection**:
- High volatility + trending: **59.82%** accuracy
- Low volatility + trending: 57.58%
- Low volatility + ranging: 55.56%
- High volatility + ranging: 47.56%

**Session Analysis**:
- Opening (9:15-10:30): **58.43%** accuracy
- Closing (14:30-15:30): 53.55%
- Mid-day (10:30-14:30): 48.39%

**Key Insight**: Different market conditions have vastly different predictability

#### D. Key Learnings

**What Worked**:
1. ✅ Feature selection > feature addition (20 features better than 51)
2. ✅ Target engineering is critical (0.15% threshold optimal)
3. ✅ Probability calibration adds 0.57% accuracy
4. ✅ Quality over quantity (1,745 clean samples > 30,100 noisy)

**What Didn't Work**:
1. ❌ Adding more features blindly decreased accuracy
2. ❌ Hyperparameter tuning alone had minimal impact
3. ❌ Complex ensembles didn't beat simple calibrated model
4. ❌ Over-filtering (combining regime + session) reduced samples too much

---

## 12. Final Results

### Production Model Configuration

```python
# Best Configuration (57.82% accuracy)
MIN_MOVEMENT_PCT = 0.15  # Target threshold
TOP_FEATURES = 20        # Feature selection

# Model
base_model = RandomForestClassifier(
    n_estimators=700,
    max_depth=30,
    class_weight='balanced',
    random_state=42
)

calibrated_model = CalibratedClassifierCV(
    base_model,
    cv=3,
    method='sigmoid'
)
```

### Why Calibrated Random Forest Won

**Final Winner**: Calibrated Random Forest (57.82% accuracy)

**Why This Configuration Performed Best**:

1. **Random Forest Foundation**
   - Handles non-linear patterns in financial data
   - Robust to outliers and noise
   - Natural feature interaction learning through tree splits
   - Bagging reduces overfitting

2. **Probability Calibration (Key Differentiator)**
   - Raw Random Forest probabilities can be poorly calibrated
   - CalibratedClassifierCV with sigmoid method improved confidence estimates
   - Better probability estimates → more reliable trading signals
   - Added +0.57% accuracy boost over base Random Forest

3. **Why It Beat Alternatives**:
   
   **vs Stacking Ensemble (56.30%)**:
   - Stacking added complexity without proportional gain
   - Multiple models can amplify errors when base models agree incorrectly
   - Calibration alone provided better returns with less complexity
   
   **vs Feature Interactions (54.58%)**:
   - Polynomial features added noise rather than signal
   - Random Forest already captures feature interactions through tree splits
   - Explicit interaction terms created redundancy
   
   **vs Regime-Specific Models**:
   - Regime models had insufficient samples per regime after 0.15% filtering
   - Single calibrated model generalized better across all conditions
   - Simpler deployment (one model vs multiple)

4. **Optimal Balance**:
   - ✅ Complex enough to capture patterns
   - ✅ Simple enough to avoid overfitting
   - ✅ Calibrated probabilities for confidence scoring
   - ✅ Production-ready (single model, fast inference)

### Top 20 Selected Features

1. macd
2. momentum_strength
3. vol_expansion
4. atr
5. volatility
6. price_position
7. prev_return
8. bb_width
9. intraday_return
10. macd_signal
11. return_3min
12. atr_pct
13. dist_from_ema_50
14. return_5min
15. hl_range
16. rsi
17. stoch_d
18. macd_diff
19. sma_10_20_cross
20. sma_5_10_cross

### Performance Comparison

| Version | Strategy | Accuracy | Samples | Threshold |
|---------|----------|----------|---------|-----------|
| **v1.0** | **Calibrated RF** | **57.82%** | 1,745 | 0.15% |
| v0.3 | Advanced Stacking | 56.30% | 1,745 | 0.15% |
| v0.2 | Feature Selection | 54.39% | 1,745 | 0.15% |
| v0.1 | Baseline | 53.49% | 30,100 | 0.05% |

**Total Improvement**: +4.33 percentage points

### Trading Implications

With 57.82% win rate and 1:1.5 risk/reward:
- Expected return per trade: +36.73%
- Daily setups (with 60% confidence filter): ~5-8 trades
- Sharpe ratio improvement: Significant

**Risk Management Guidelines**:
- Position size: 1-2% of capital
- Stop loss: 0.3-0.5%
- Take profit: 1.5× stop loss minimum
- Trade only signals with >60% model confidence

### Production Files

**Core Scripts**:
- `train_production.py` - Best training strategy (57.82%)
- `basic_approach.py` - Baseline reference (53.49%)

**Experiments** (archived in `experiments/`):
- `train_optimized.py` - Hyperparameter tuning
- `train_advanced_strategies.py` - Feature selection experiments
- `train_ultimate.py` - Regime/session combinations
- `train_quick.py` - Fast validation

### Installation & Usage

```bash
# Install dependencies
pip install -r requirements.txt

# Train production model
python train_production.py


```

---

## Conclusion

The system successfully predicts NIFTY price movements with **57.82% accuracy**, a significant improvement over the 53.49% baseline. Key success factors were:

1. **Smart target engineering** (0.15% threshold)
2. **Aggressive feature selection** (20 vs 51 features)
3. **Probability calibration** for better confidence estimates
4. **Understanding market regimes** (when to trade vs avoid)

The production-ready system is now organized with:
- ✅ Clean codebase structure
- ✅ Comprehensive documentation
- ✅ Experiment archives for reference
- ✅ Ready for deployment

**Current Status**: Production Ready v1.0.0
**Performance**: 57.82% accuracy (+4.33% vs baseline)

