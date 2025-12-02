"""
ULTIMATE STRATEGY - Combining Best Insights

Combines:
1. High volatility + trending regime filter (59.82%)
2. Opening session filter (58.43%)
3. Probability calibration (57.82%)
4. 0.15% threshold
5. Top 20 features

Expected: 60%+ accuracy
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from catboost import CatBoostClassifier

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer


def detect_regime(df):
    """Detect if market is high vol + trending"""
    returns = df['close'].pct_change()
    df['vol'] = returns.rolling(20).std() * 100

    df['sma_20'] = df['close'].rolling(20).mean()
    df['trend_strength'] = np.abs((df['close'] - df['sma_20']) / df['sma_20']) * 100

    vol_median = df['vol'].median()
    trend_median = df['trend_strength'].median()

    # High vol + trending
    df['is_good_regime'] = (df['vol'] > vol_median) & (df['trend_strength'] > trend_median)

    return df


def detect_opening_session(df):
    """Detect if in opening session (9:15-10:30)"""
    hour = df['timestamp'].dt.hour
    minute = df['timestamp'].dt.minute
    time_decimal = hour + minute/60

    df['is_opening'] = (time_decimal >= 9.25) & (time_decimal < 10.5)

    return df


def select_top_features(X_train, y_train, X_test, n_features=20):
    """Select top features"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = importance_df.head(n_features)['feature'].tolist()

    print(f"\nTop {n_features} Features:")
    for i, row in importance_df.head(n_features).iterrows():
        print(f"  {i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

    return X_train[top_features], X_test[top_features], top_features


def main():
    print("\n" + "="*70)
    print("ULTIMATE STRATEGY - Best of All Worlds")
    print("="*70)

    # Load data
    print("\nStep 1: Loading data...")
    data_loader = DataLoader(config.DATA_PATH)
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.df

    # Create features
    print("\nStep 2: Feature engineering...")
    feature_engineer = FeatureEngineer(
        df=df, sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )

    feature_engineer.create_price_features()
    feature_engineer.create_moving_averages()
    feature_engineer.create_rsi()
    feature_engineer.create_macd()
    feature_engineer.create_volatility_features()
    feature_engineer.create_lag_features()
    feature_engineer.create_time_features()
    feature_engineer.create_bollinger_bands()
    feature_engineer.create_atr()
    feature_engineer.create_stochastic()
    feature_engineer.create_roc()
    feature_engineer.create_advanced_features()

    df = feature_engineer.df.dropna().reset_index(drop=True)

    # Create target (0.15% threshold)
    print("\nStep 3: Creating target with 0.15% threshold...")
    df['next_close'] = df['close'].shift(-1)
    df['next_return'] = ((df['next_close'] - df['close']) / df['close']) * 100
    df = df[np.abs(df['next_return']) >= 0.15].copy()
    df['target'] = (df['next_return'] > 0).astype(int)
    df = df.drop(['next_close', 'next_return'], axis=1).dropna().reset_index(drop=True)

    print(f"Samples after 0.15% filter: {len(df)}")

    # ============================================================
    # TEST 1: Baseline (no filters)
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 1: BASELINE (No filters)")
    print('='*70)

    X = df[feature_engineer.feature_columns]
    y = df['target']

    split_idx = int(len(df) * 0.7)
    X_train_full = X.iloc[:split_idx]
    X_test_full = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    X_train, X_test, top_features = select_top_features(X_train_full, y_train, X_test_full, 20)

    # Calibrated Random Forest
    rf_base = RandomForestClassifier(
        n_estimators=700, max_depth=30, class_weight='balanced',
        random_state=42, n_jobs=-1
    )

    calibrated_base = CalibratedClassifierCV(rf_base, cv=3, method='sigmoid')
    calibrated_base.fit(X_train, y_train)

    y_pred_base = calibrated_base.predict(X_test)
    acc_base = accuracy_score(y_test, y_pred_base)

    print(f"\nBaseline Accuracy: {acc_base:.4f} ({acc_base*100:.2f}%)")
    print(f"Test samples: {len(y_test)}")

    # ============================================================
    # TEST 2: High Vol + Trending Filter
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 2: HIGH VOLATILITY + TRENDING REGIME FILTER")
    print('='*70)

    df_regime = detect_regime(df.copy())
    df_good_regime = df_regime[df_regime['is_good_regime']].copy()

    print(f"\nOriginal samples: {len(df_regime)}")
    print(f"Good regime samples: {len(df_good_regime)} ({len(df_good_regime)/len(df_regime)*100:.1f}%)")

    if len(df_good_regime) > 100:
        X_regime = df_good_regime[feature_engineer.feature_columns]
        y_regime = df_good_regime['target']

        split_idx_regime = int(len(X_regime) * 0.7)
        X_train_regime_full = X_regime.iloc[:split_idx_regime]
        X_test_regime_full = X_regime.iloc[split_idx_regime:]
        y_train_regime = y_regime.iloc[:split_idx_regime]
        y_test_regime = y_regime.iloc[split_idx_regime:]

        X_train_regime, X_test_regime, _ = select_top_features(
            X_train_regime_full, y_train_regime, X_test_regime_full, 20
        )

        model_regime = CatBoostClassifier(
            iterations=500, depth=10, learning_rate=0.05,
            random_state=42, verbose=0
        )

        model_regime.fit(X_train_regime, y_train_regime)
        y_pred_regime = model_regime.predict(X_test_regime)
        acc_regime = accuracy_score(y_test_regime, y_pred_regime)

        print(f"\nRegime Filter Accuracy: {acc_regime:.4f} ({acc_regime*100:.2f}%)")
        print(f"Test samples: {len(y_test_regime)}")
    else:
        acc_regime = 0
        print("⚠️  Insufficient samples for regime filter")

    # ============================================================
    # TEST 3: Opening Session Filter
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 3: OPENING SESSION FILTER (9:15-10:30)")
    print('='*70)

    df_session = detect_opening_session(df.copy())
    df_opening = df_session[df_session['is_opening']].copy()

    print(f"\nOriginal samples: {len(df_session)}")
    print(f"Opening session samples: {len(df_opening)} ({len(df_opening)/len(df_session)*100:.1f}%)")

    if len(df_opening) > 100:
        X_opening = df_opening[feature_engineer.feature_columns]
        y_opening = df_opening['target']

        split_idx_opening = int(len(X_opening) * 0.7)
        X_train_opening_full = X_opening.iloc[:split_idx_opening]
        X_test_opening_full = X_opening.iloc[split_idx_opening:]
        y_train_opening = y_opening.iloc[:split_idx_opening]
        y_test_opening = y_opening.iloc[split_idx_opening:]

        X_train_opening, X_test_opening, _ = select_top_features(
            X_train_opening_full, y_train_opening, X_test_opening_full, 20
        )

        rf_opening = RandomForestClassifier(
            n_estimators=700, max_depth=30, class_weight='balanced',
            random_state=42, n_jobs=-1
        )

        calibrated_opening = CalibratedClassifierCV(rf_opening, cv=3, method='sigmoid')
        calibrated_opening.fit(X_train_opening, y_train_opening)

        y_pred_opening = calibrated_opening.predict(X_test_opening)
        acc_opening = accuracy_score(y_test_opening, y_pred_opening)

        print(f"\nOpening Session Accuracy: {acc_opening:.4f} ({acc_opening*100:.2f}%)")
        print(f"Test samples: {len(y_test_opening)}")
    else:
        acc_opening = 0
        print("⚠️  Insufficient samples for opening session")

    # ============================================================
    # TEST 4: COMBINED - Opening + High Vol Trending
    # ============================================================
    print(f"\n{'='*70}")
    print("TEST 4: ULTIMATE COMBO - Opening + High Vol + Trending")
    print('='*70)

    df_ultimate = detect_regime(df.copy())
    df_ultimate = detect_opening_session(df_ultimate)
    df_ultimate = df_ultimate[df_ultimate['is_good_regime'] & df_ultimate['is_opening']].copy()

    print(f"\nOriginal samples: {len(df)}")
    print(f"Ultimate filter samples: {len(df_ultimate)} ({len(df_ultimate)/len(df)*100:.1f}%)")

    if len(df_ultimate) > 50:
        X_ultimate = df_ultimate[feature_engineer.feature_columns]
        y_ultimate = df_ultimate['target']

        split_idx_ult = int(len(X_ultimate) * 0.7)
        X_train_ult_full = X_ultimate.iloc[:split_idx_ult]
        X_test_ult_full = X_ultimate.iloc[split_idx_ult:]
        y_train_ult = y_ultimate.iloc[:split_idx_ult]
        y_test_ult = y_ultimate.iloc[split_idx_ult:]

        if len(X_test_ult_full) >= 20:
            X_train_ult, X_test_ult, _ = select_top_features(
                X_train_ult_full, y_train_ult, X_test_ult_full, 20
            )

            rf_ult = RandomForestClassifier(
                n_estimators=700, max_depth=30, class_weight='balanced',
                random_state=42, n_jobs=-1
            )

            calibrated_ult = CalibratedClassifierCV(rf_ult, cv=3, method='sigmoid')
            calibrated_ult.fit(X_train_ult, y_train_ult)

            y_pred_ult = calibrated_ult.predict(X_test_ult)
            acc_ult = accuracy_score(y_test_ult, y_pred_ult)

            print(f"\nUltimate Combo Accuracy: {acc_ult:.4f} ({acc_ult*100:.2f}%)")
            print(f"Test samples: {len(y_test_ult)}")

            print(f"\nConfusion Matrix:")
            print(confusion_matrix(y_test_ult, y_pred_ult))

            print(f"\nClassification Report:")
            print(classification_report(y_test_ult, y_pred_ult, target_names=['Down', 'Up']))
        else:
            acc_ult = 0
            print("⚠️  Too few test samples for ultimate combo")
    else:
        acc_ult = 0
        print("⚠️  Insufficient samples for ultimate combo")

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print('='*70 + "\n")

    results = [
        {'Strategy': 'Baseline (0.15% threshold + calibration)', 'Accuracy': acc_base, 'Acc %': f"{acc_base*100:.2f}%"},
        {'Strategy': 'High Vol + Trending Filter', 'Accuracy': acc_regime, 'Acc %': f"{acc_regime*100:.2f}%"},
        {'Strategy': 'Opening Session Filter', 'Accuracy': acc_opening, 'Acc %': f"{acc_opening*100:.2f}%"},
        {'Strategy': 'ULTIMATE: Opening + High Vol + Trending', 'Accuracy': acc_ult, 'Acc %': f"{acc_ult*100:.2f}%"}
    ]

    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))

    best = results_df.iloc[0]
    baseline_orig = 0.5349

    print(f"\n{'='*70}")
    print(f"🏆 BEST STRATEGY: {best['Strategy']}")
    print(f"   Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)")
    print(f"\n📈 Original Baseline: {baseline_orig:.4f} (53.49%)")
    print(f"📊 Total Improvement: +{(best['Accuracy'] - baseline_orig)*100:.2f} percentage points")
    print("="*70 + "\n")

    # Save results
    output_file = f"{config.OUTPUT_DIR}/ultimate_strategy_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ULTIMATE STRATEGY RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write(results_df.to_string(index=False))
        f.write(f"\n\n{'='*70}\n")
        f.write(f"BEST STRATEGY: {best['Strategy']}\n")
        f.write(f"Accuracy: {best['Accuracy']:.4f} ({best['Accuracy']*100:.2f}%)\n")
        f.write(f"Original Baseline: {baseline_orig:.4f} (53.49%)\n")
        f.write(f"Improvement: +{(best['Accuracy'] - baseline_orig)*100:.2f} percentage points\n")
        f.write("="*70 + "\n")

    print(f"Results saved to: {output_file}\n")


if __name__ == "__main__":
    main()
