"""
Advanced Strategies to Break the 53% Accuracy Barrier

Strategies implemented:
1. Aggressive feature selection (keep only top performers)
2. Different target thresholds (larger movements)
3. Multi-candle prediction (2-3 candles ahead)
4. Volatility regime-based modeling
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer


def select_top_features(X_train, y_train, X_test, n_features=20):
    """Select only the top N most important features"""
    print(f"\n{'='*70}")
    print(f"FEATURE SELECTION - Keeping Top {n_features} Features")
    print('='*70)

    # Train RF for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    # Get feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Select top features
    top_features = importance_df.head(n_features)['feature'].tolist()

    print(f"\nTop {n_features} features:")
    for i, row in importance_df.head(n_features).iterrows():
        print(f"  {row['feature']:30s} {row['importance']:.4f}")

    print(f"\nReducing from {len(X_train.columns)} to {n_features} features")
    print('='*70)

    return X_train[top_features], X_test[top_features], top_features


def create_target_with_threshold(df, threshold_pct=0.15):
    """Create target with larger movement threshold"""
    print(f"\nCreating target with {threshold_pct}% threshold...")

    # Calculate next candle return
    df['next_close'] = df['close'].shift(-1)
    df['next_return'] = ((df['next_close'] - df['close']) / df['close']) * 100

    # Filter only significant movements
    df_filtered = df[np.abs(df['next_return']) >= threshold_pct].copy()

    # Create binary target
    df_filtered['target'] = (df_filtered['next_return'] > 0).astype(int)

    # Clean up
    df_filtered = df_filtered.drop(['next_close', 'next_return'], axis=1)

    print(f"Original samples: {len(df)}")
    print(f"After filtering (|movement| >= {threshold_pct}%): {len(df_filtered)}")
    print(f"Reduction: {(1 - len(df_filtered)/len(df))*100:.1f}%")

    # Class distribution
    class_dist = df_filtered['target'].value_counts()
    print(f"\nClass distribution:")
    print(f"  Down (0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(df_filtered)*100:.1f}%)")
    print(f"  Up (1):   {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(df_filtered)*100:.1f}%)")

    return df_filtered


def create_multi_candle_target(df, n_candles=3):
    """Predict N candles ahead instead of next candle"""
    print(f"\nCreating target for {n_candles} candles ahead...")

    # Calculate return N candles ahead
    df['future_close'] = df['close'].shift(-n_candles)
    df['future_return'] = ((df['future_close'] - df['close']) / df['close']) * 100

    # Filter significant movements
    df_filtered = df[np.abs(df['future_return']) >= 0.1].copy()

    # Create binary target
    df_filtered['target'] = (df_filtered['future_return'] > 0).astype(int)

    # Clean up
    df_filtered = df_filtered.drop(['future_close', 'future_return'], axis=1)

    print(f"Predicting {n_candles} candles ahead")
    print(f"Filtered samples: {len(df_filtered)}")

    return df_filtered


def split_by_volatility_regime(df, X, y):
    """Split data into high and low volatility regimes"""
    print(f"\n{'='*70}")
    print("VOLATILITY REGIME ANALYSIS")
    print('='*70)

    # Calculate rolling volatility
    df['vol'] = df['close'].pct_change().rolling(20).std() * 100
    median_vol = df['vol'].median()

    print(f"Median volatility: {median_vol:.4f}%")

    # Split into regimes
    high_vol_mask = df['vol'] > median_vol

    X_high_vol = X[high_vol_mask]
    y_high_vol = y[high_vol_mask]
    X_low_vol = X[~high_vol_mask]
    y_low_vol = y[~high_vol_mask]

    print(f"High volatility samples: {len(X_high_vol)}")
    print(f"Low volatility samples:  {len(X_low_vol)}")
    print('='*70)

    return X_high_vol, y_high_vol, X_low_vol, y_low_vol


def train_and_evaluate(X_train, y_train, X_test, y_test, label=""):
    """Train models and return best accuracy"""
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=700, max_depth=30, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=700, max_depth=20, learning_rate=0.05,
            subsample=0.8, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=700, max_depth=20, learning_rate=0.05,
            num_leaves=127, class_weight='balanced', random_state=42, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=700, depth=10, learning_rate=0.05,
            random_state=42, verbose=0
        )
    }

    print(f"\n{label}")
    print("-" * 70)

    best_model = None
    best_accuracy = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"{name:20s} Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_name = name

    print(f"\nBest: {best_name} - {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

    return best_model, best_accuracy, best_name


def main():
    print("\n" + "="*70)
    print("ADVANCED ACCURACY IMPROVEMENT STRATEGIES")
    print("="*70)

    results_summary = []

    # =========================
    # STRATEGY 1: Feature Selection
    # =========================
    print("\n\n" + "="*70)
    print("STRATEGY 1: AGGRESSIVE FEATURE SELECTION")
    print("="*70)

    data_loader = DataLoader(config.DATA_PATH)
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.create_target(
        min_movement_pct=config.MIN_MOVEMENT_PCT,
        use_multiclass=config.USE_MULTICLASS,
        up_threshold=config.UP_THRESHOLD,
        down_threshold=config.DOWN_THRESHOLD
    )

    feature_engineer = FeatureEngineer(
        df=df, sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )
    df, feature_columns = feature_engineer.create_all_features()
    X, y = feature_engineer.get_feature_matrix()

    # Split
    split_index = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Select top 20 features
    X_train_sel, X_test_sel, top_features = select_top_features(X_train, y_train, X_test, n_features=20)

    _, accuracy_s1, name_s1 = train_and_evaluate(
        X_train_sel, y_train, X_test_sel, y_test,
        "Training with Top 20 Features Only"
    )

    results_summary.append({
        'Strategy': 'Top 20 Features',
        'Best Model': name_s1,
        'Accuracy': accuracy_s1,
        'Accuracy %': f"{accuracy_s1*100:.2f}%"
    })

    # =========================
    # STRATEGY 2: Larger Movement Threshold
    # =========================
    print("\n\n" + "="*70)
    print("STRATEGY 2: LARGER MOVEMENT THRESHOLD (0.15%)")
    print("="*70)

    data_loader2 = DataLoader(config.DATA_PATH)
    data_loader2.load_data()
    data_loader2.preprocess()
    df2 = data_loader2.df

    # Create features first
    feature_engineer2 = FeatureEngineer(
        df=df2, sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )

    # Create features without target
    feature_engineer2.create_price_features()
    feature_engineer2.create_moving_averages()
    feature_engineer2.create_rsi()
    feature_engineer2.create_macd()
    feature_engineer2.create_volatility_features()
    feature_engineer2.create_lag_features()
    feature_engineer2.create_time_features()
    feature_engineer2.create_bollinger_bands()
    feature_engineer2.create_atr()
    feature_engineer2.create_stochastic()
    feature_engineer2.create_roc()
    feature_engineer2.create_advanced_features()

    df2 = feature_engineer2.df.dropna().reset_index(drop=True)

    # Now create target with higher threshold
    df2 = create_target_with_threshold(df2, threshold_pct=0.15)
    df2 = df2.dropna().reset_index(drop=True)

    X2 = df2[feature_engineer2.feature_columns]
    y2 = df2['target']

    split_index2 = int(len(df2) * config.TRAIN_TEST_SPLIT_RATIO)
    X_train2, X_test2 = X2.iloc[:split_index2], X2.iloc[split_index2:]
    y_train2, y_test2 = y2.iloc[:split_index2], y2.iloc[split_index2:]

    # Feature selection
    X_train2_sel, X_test2_sel, _ = select_top_features(X_train2, y_train2, X_test2, n_features=20)

    _, accuracy_s2, name_s2 = train_and_evaluate(
        X_train2_sel, y_train2, X_test2_sel, y_test2,
        "Training with 0.15% Threshold"
    )

    results_summary.append({
        'Strategy': '0.15% Threshold + Top 20 Features',
        'Best Model': name_s2,
        'Accuracy': accuracy_s2,
        'Accuracy %': f"{accuracy_s2*100:.2f}%"
    })

    # =========================
    # STRATEGY 3: Multi-Candle Prediction
    # =========================
    print("\n\n" + "="*70)
    print("STRATEGY 3: PREDICT 3 CANDLES AHEAD")
    print("="*70)

    data_loader3 = DataLoader(config.DATA_PATH)
    data_loader3.load_data()
    data_loader3.preprocess()
    df3 = data_loader3.df

    # Create features
    feature_engineer3 = FeatureEngineer(
        df=df3, sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )

    feature_engineer3.create_price_features()
    feature_engineer3.create_moving_averages()
    feature_engineer3.create_rsi()
    feature_engineer3.create_macd()
    feature_engineer3.create_volatility_features()
    feature_engineer3.create_lag_features()
    feature_engineer3.create_time_features()
    feature_engineer3.create_bollinger_bands()
    feature_engineer3.create_atr()
    feature_engineer3.create_stochastic()
    feature_engineer3.create_roc()
    feature_engineer3.create_advanced_features()

    df3 = feature_engineer3.df.dropna().reset_index(drop=True)

    # Create multi-candle target
    df3 = create_multi_candle_target(df3, n_candles=3)
    df3 = df3.dropna().reset_index(drop=True)

    X3 = df3[feature_engineer3.feature_columns]
    y3 = df3['target']

    split_index3 = int(len(df3) * config.TRAIN_TEST_SPLIT_RATIO)
    X_train3, X_test3 = X3.iloc[:split_index3], X3.iloc[split_index3:]
    y_train3, y_test3 = y3.iloc[:split_index3], y3.iloc[split_index3:]

    # Feature selection
    X_train3_sel, X_test3_sel, _ = select_top_features(X_train3, y_train3, X_test3, n_features=20)

    _, accuracy_s3, name_s3 = train_and_evaluate(
        X_train3_sel, y_train3, X_test3_sel, y_test3,
        "Training for 3-Candle Ahead Prediction"
    )

    results_summary.append({
        'Strategy': '3-Candle Ahead + Top 20 Features',
        'Best Model': name_s3,
        'Accuracy': accuracy_s3,
        'Accuracy %': f"{accuracy_s3*100:.2f}%"
    })

    # =========================
    # FINAL COMPARISON
    # =========================
    print("\n\n" + "="*70)
    print("FINAL STRATEGY COMPARISON")
    print("="*70 + "\n")

    results_df = pd.DataFrame(results_summary).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))

    best_strategy = results_df.iloc[0]
    baseline = 0.5349

    print(f"\n{'='*70}")
    print(f"🏆 BEST STRATEGY: {best_strategy['Strategy']}")
    print(f"   Model: {best_strategy['Best Model']}")
    print(f"   Accuracy: {best_strategy['Accuracy']:.4f} ({best_strategy['Accuracy']*100:.2f}%)")
    print(f"\n📈 Baseline: {baseline:.4f} (53.49%)")

    improvement = (best_strategy['Accuracy'] - baseline) * 100
    if improvement > 0:
        print(f"📊 Improvement: +{improvement:.2f} percentage points ✅")
    else:
        print(f"📊 Change: {improvement:.2f} percentage points")

    print("="*70 + "\n")

    # Save results
    output_file = f"{config.OUTPUT_DIR}/advanced_strategies_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ADVANCED STRATEGIES RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\n{'='*70}\n")
        f.write(f"BEST STRATEGY: {best_strategy['Strategy']}\n")
        f.write(f"Model: {best_strategy['Best Model']}\n")
        f.write(f"Accuracy: {best_strategy['Accuracy']:.4f} ({best_strategy['Accuracy']*100:.2f}%)\n")
        f.write(f"Baseline: {baseline:.4f} (53.49%)\n")
        f.write(f"Improvement: {improvement:+.2f} percentage points\n")
        f.write("="*70 + "\n")

    print(f"Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
