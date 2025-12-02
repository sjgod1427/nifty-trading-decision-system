"""
Configuration file for NIFTY Price Prediction Project
Contains all parameters and paths in one place for easy tuning

VERSION: 1.0.0 - Production Configuration
BEST ACCURACY: 57.82% (with 0.15% threshold + calibration)
"""

import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "nifty_data.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Output files
FINAL_PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "final_predictions.csv")
METRICS_REPORT_PATH = os.path.join(OUTPUT_DIR, "metrics_report.txt")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "feature_importance.png")

# Data processing parameters
TRAIN_TEST_SPLIT_RATIO = 0.7  # 70% train, 30% test
RANDOM_STATE = 42  # For reproducibility

# Target creation parameters
# PRODUCTION SETTING: 0.15 achieves 57.82% accuracy (tested extensively)
# Alternative: 0.05 gives 53.49% (baseline)
MIN_MOVEMENT_PCT = 0.15  # Filter movements below this threshold
USE_MULTICLASS = False   # Binary classification (Up/Down)
UP_THRESHOLD = 0.1       # For multiclass: threshold for "Strong Up" (not used)
DOWN_THRESHOLD = -0.1    # For multiclass: threshold for "Strong Down" (not used)

# Feature engineering parameters
SMA_WINDOWS = [5, 10, 20]  # Simple Moving Average windows
RSI_PERIOD = 14  # RSI period
VOLATILITY_WINDOW = 5  # Rolling volatility window
LAG_PERIODS = 3  # Number of lag features

# Model parameters
MODELS_TO_TRAIN = ['logistic_regression', 'random_forest', 'lightgbm', 'xgboost', 'lstm']

# Logistic Regression parameters (improved with better solver)
LR_PARAMS = {
    'max_iter': 2000,
    'random_state': RANDOM_STATE,
    'solver': 'saga',  # Better for large datasets
    'penalty': 'l2',
    'C': 1.0
}

# Random Forest parameters (improved - deeper trees, more estimators)
RF_PARAMS = {
    'n_estimators': 300,      # Increased from 100
    'max_depth': 20,          # Increased from 10
    'min_samples_split': 20,  # Increased from 10
    'min_samples_leaf': 10,   # Increased from 5
    'max_features': 'sqrt',
    'class_weight': 'balanced',  # NEW: Handle class imbalance
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LightGBM parameters (improved)
LGBM_PARAMS = {
    'n_estimators': 500,      # Increased from 100
    'max_depth': 15,          # Increased from 10
    'learning_rate': 0.05,    # Lower LR with more estimators
    'num_leaves': 63,         # Increased from 31
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'class_weight': 'balanced',
    'random_state': RANDOM_STATE,
    'verbose': -1
}

# XGBoost parameters (NEW!)
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 15,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'tree_method': 'hist',
    'eval_metric': 'logloss'
}

# Evaluation parameters
METRICS_TO_CALCULATE = ['accuracy', 'precision', 'recall', 'f1']

# Display settings
PRINT_DETAILED_METRICS = True
SAVE_FEATURE_IMPORTANCE = True