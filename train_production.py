


"""
Advanced ML Strategies (Classical ML Only - No Deep Learning)

Strategies:
1. Regime-Based Modeling (Volatility + Trend)
2. Multi-Timeframe Predictions
3. Probability Calibration
4. Session-Based Models (Opening/Mid/Closing hours)
5. Feature Interactions
6. Advanced Stacking
7. Gradient Boosting Ensemble
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer


def detect_market_regime(df):
    """
    Detect market regime based on volatility and trend
    Returns: 'high_vol_trending', 'high_vol_ranging', 'low_vol_trending', 'low_vol_ranging'
    """
    print(f"\n{'='*70}")
    print("DETECTING MARKET REGIMES")
    print('='*70)

    # Calculate volatility (20-period rolling std of returns)
    returns = df['close'].pct_change()
    df['regime_volatility'] = returns.rolling(20).std() * 100

    # Calculate trend strength (ADX-like indicator)
    window = 14
    df['high_low'] = df['high'] - df['low']
    df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close'] = np.abs(df['low'] - df['close'].shift(1))

    df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
    df['atr_regime'] = df['tr'].rolling(window).mean()

    # Price vs SMA distance as trend indicator
    df['sma_20_regime'] = df['close'].rolling(20).mean()
    df['trend_strength'] = np.abs((df['close'] - df['sma_20_regime']) / df['sma_20_regime']) * 100

    # Define regimes
    vol_median = df['regime_volatility'].median()
    trend_median = df['trend_strength'].median()

    print(f"Volatility median: {vol_median:.4f}%")
    print(f"Trend strength median: {trend_median:.4f}%")

    # Classify regimes
    conditions = [
        (df['regime_volatility'] > vol_median) & (df['trend_strength'] > trend_median),
        (df['regime_volatility'] > vol_median) & (df['trend_strength'] <= trend_median),
        (df['regime_volatility'] <= vol_median) & (df['trend_strength'] > trend_median),
        (df['regime_volatility'] <= vol_median) & (df['trend_strength'] <= trend_median)
    ]

    choices = ['high_vol_trending', 'high_vol_ranging', 'low_vol_trending', 'low_vol_ranging']
    df['regime'] = np.select(conditions, choices, default='low_vol_ranging')

    # Print regime distribution
    regime_counts = df['regime'].value_counts()
    print(f"\nRegime Distribution:")
    for regime, count in regime_counts.items():
        print(f"  {regime:25s}: {count:6d} ({count/len(df)*100:5.1f}%)")

    print('='*70)

    return df


def detect_session(df):
    """Detect trading session: opening (9:15-10:30), mid (10:30-14:30), closing (14:30-15:30)"""
    print(f"\n{'='*70}")
    print("DETECTING TRADING SESSIONS")
    print('='*70)

    hour = df['timestamp'].dt.hour
    minute = df['timestamp'].dt.minute
    time_decimal = hour + minute/60

    conditions = [
        (time_decimal >= 9.25) & (time_decimal < 10.5),   # Opening: 9:15-10:30
        (time_decimal >= 10.5) & (time_decimal < 14.5),   # Mid: 10:30-14:30
        (time_decimal >= 14.5) & (time_decimal <= 15.5)   # Closing: 14:30-15:30
    ]

    choices = ['opening', 'mid', 'closing']
    df['session'] = np.select(conditions, choices, default='mid')

    session_counts = df['session'].value_counts()
    print(f"\nSession Distribution:")
    for session, count in session_counts.items():
        print(f"  {session:10s}: {count:6d} ({count/len(df)*100:5.1f}%)")

    print('='*70)

    return df


def create_feature_interactions(X, top_features, n_interactions=5):
    """Create polynomial feature interactions for top features"""
    print(f"\n{'='*70}")
    print(f"CREATING FEATURE INTERACTIONS (Top {n_interactions} features)")
    print('='*70)

    # Select top N features for interaction
    interaction_features = top_features[:n_interactions]
    print(f"Features for interaction: {', '.join(interaction_features)}")

    X_subset = X[interaction_features]

    # Create polynomial features (degree 2 = feature pairs)
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_interactions = poly.fit_transform(X_subset)

    # Get feature names
    interaction_names = poly.get_feature_names_out(interaction_features)

    # Keep only interaction terms (not original features)
    interaction_cols = [name for name in interaction_names if '*' in name]
    interaction_idx = [i for i, name in enumerate(interaction_names) if '*' in name]

    X_interactions_only = X_interactions[:, interaction_idx]

    # Add to original dataframe
    X_combined = pd.concat([
        X.reset_index(drop=True),
        pd.DataFrame(X_interactions_only, columns=interaction_cols)
    ], axis=1)

    print(f"Created {len(interaction_cols)} interaction features")
    print(f"Total features: {len(X_combined.columns)}")
    print('='*70)

    return X_combined


def train_regime_specific_models(df, X, y, top_features):
    """Train separate models for each market regime"""
    print(f"\n{'='*70}")
    print("REGIME-BASED MODELING")
    print('='*70)

    regimes = df['regime'].unique()
    regime_models = {}
    regime_results = []

    for regime in regimes:
        print(f"\n{'-'*70}")
        print(f"Training model for: {regime}")
        print(f"{'-'*70}")

        # Filter data for this regime
        regime_mask = df['regime'] == regime
        X_regime = X[regime_mask][top_features]
        y_regime = y[regime_mask]

        # Split
        split_idx = int(len(X_regime) * 0.7)
        X_train = X_regime.iloc[:split_idx]
        X_test = X_regime.iloc[split_idx:]
        y_train = y_regime.iloc[:split_idx]
        y_test = y_regime.iloc[split_idx:]

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        if len(X_test) < 10:  # Skip if too few test samples
            print(f"⚠️  Skipping {regime} - insufficient test samples")
            continue

        # Train CatBoost (often best for regime-specific patterns)
        model = CatBoostClassifier(
            iterations=500,
            depth=10,
            learning_rate=0.05,
            random_state=42,
            verbose=0
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        regime_models[regime] = model

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        regime_results.append({
            'Regime': regime,
            'Train Size': len(X_train),
            'Test Size': len(X_test),
            'Accuracy': accuracy,
            'Accuracy %': f"{accuracy*100:.2f}%"
        })

    print(f"\n{'='*70}")
    print("REGIME MODEL SUMMARY")
    print('='*70)
    results_df = pd.DataFrame(regime_results)
    print(results_df.to_string(index=False))
    print('='*70)

    return regime_models, results_df


def train_session_specific_models(df, X, y, top_features):
    """Train separate models for each trading session"""
    print(f"\n{'='*70}")
    print("SESSION-BASED MODELING")
    print('='*70)

    sessions = ['opening', 'mid', 'closing']
    session_models = {}
    session_results = []

    for session in sessions:
        print(f"\n{'-'*70}")
        print(f"Training model for: {session.upper()} session")
        print(f"{'-'*70}")

        # Filter data for this session
        session_mask = df['session'] == session
        X_session = X[session_mask][top_features]
        y_session = y[session_mask]

        # Split
        split_idx = int(len(X_session) * 0.7)
        X_train = X_session.iloc[:split_idx]
        X_test = X_session.iloc[split_idx:]
        y_train = y_session.iloc[:split_idx]
        y_test = y_session.iloc[split_idx:]

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # Train model
        model = RandomForestClassifier(
            n_estimators=700,
            max_depth=30,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        session_models[session] = model

        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        session_results.append({
            'Session': session.capitalize(),
            'Train Size': len(X_train),
            'Test Size': len(X_test),
            'Accuracy': accuracy,
            'Accuracy %': f"{accuracy*100:.2f}%"
        })

    print(f"\n{'='*70}")
    print("SESSION MODEL SUMMARY")
    print('='*70)
    results_df = pd.DataFrame(session_results)
    print(results_df.to_string(index=False))
    print('='*70)

    return session_models, results_df


def train_calibrated_models(X_train, y_train, X_test, y_test):
    """Train probability-calibrated models"""
    print(f"\n{'='*70}")
    print("PROBABILITY CALIBRATION")
    print('='*70)

    base_models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=700, max_depth=30, class_weight='balanced',
            random_state=42, n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=500, depth=10, learning_rate=0.05,
            random_state=42, verbose=0
        )
    }

    results = []

    for name, model in base_models.items():
        print(f"\n{name}:")

        # Train base model
        model.fit(X_train, y_train)
        y_pred_base = model.predict(X_test)
        acc_base = accuracy_score(y_test, y_pred_base)

        # Calibrate
        calibrated = CalibratedClassifierCV(model, cv=3, method='sigmoid')
        calibrated.fit(X_train, y_train)
        y_pred_cal = calibrated.predict(X_test)
        acc_cal = accuracy_score(y_test, y_pred_cal)

        # Get probabilities for AUC
        try:
            y_proba = calibrated.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_proba)
        except:
            auc = 0

        print(f"  Base Accuracy:       {acc_base:.4f} ({acc_base*100:.2f}%)")
        print(f"  Calibrated Accuracy: {acc_cal:.4f} ({acc_cal*100:.2f}%)")
        print(f"  ROC-AUC:             {auc:.4f}")

        results.append({
            'Model': name,
            'Base Acc': acc_base,
            'Calibrated Acc': acc_cal,
            'Improvement': acc_cal - acc_base,
            'AUC': auc
        })

    print('='*70)
    return pd.DataFrame(results)


def train_advanced_stacking(X_train, y_train, X_test, y_test):
    """Advanced stacking with multiple meta-learners"""
    print(f"\n{'='*70}")
    print("ADVANCED STACKING ENSEMBLE")
    print('='*70)

    # Base estimators
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=500, max_depth=30, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('xgb', xgb.XGBClassifier(n_estimators=500, max_depth=20, learning_rate=0.05, random_state=42, n_jobs=-1)),
        ('lgb', lgb.LGBMClassifier(n_estimators=500, max_depth=20, learning_rate=0.05, class_weight='balanced', random_state=42, verbose=-1)),
        ('catboost', CatBoostClassifier(iterations=500, depth=10, learning_rate=0.05, random_state=42, verbose=0))
    ]

    # Test different meta-learners
    meta_learners = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'CatBoost': CatBoostClassifier(iterations=100, depth=5, learning_rate=0.1, random_state=42, verbose=0)
    }

    results = []

    for meta_name, meta_learner in meta_learners.items():
        print(f"\nStacking with {meta_name} meta-learner...")

        stacking_clf = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=3,
            n_jobs=-1
        )

        stacking_clf.fit(X_train, y_train)
        y_pred = stacking_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

        results.append({
            'Meta-Learner': meta_name,
            'Accuracy': accuracy,
            'Accuracy %': f"{accuracy*100:.2f}%"
        })

    print(f"\n{'='*70}")
    results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
    print(results_df.to_string(index=False))
    print('='*70)

    return results_df


def select_top_features(X_train, y_train, X_test, n_features=20):
    """Select top N features"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)

    top_features = importance_df.head(n_features)['feature'].tolist()

    return X_train[top_features], X_test[top_features], top_features


def main():
    print("\n" + "="*70)
    print("ADVANCED ML STRATEGIES (Classical ML Only)")
    print("="*70)

    all_results = []

    # Load and prepare data with 0.15% threshold (best from previous analysis)
    print("\nLoading data with 0.15% threshold...")
    data_loader = DataLoader(config.DATA_PATH)
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.df

    # Create features
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

    # Create target with 0.15% threshold
    df['next_close'] = df['close'].shift(-1)
    df['next_return'] = ((df['next_close'] - df['close']) / df['close']) * 100
    df = df[np.abs(df['next_return']) >= 0.15].copy()
    df['target'] = (df['next_return'] > 0).astype(int)
    df = df.drop(['next_close', 'next_return'], axis=1).dropna().reset_index(drop=True)

    print(f"Dataset size: {len(df)} samples")

    # Detect regimes and sessions
    df = detect_market_regime(df)
    df = detect_session(df)

    X = df[feature_engineer.feature_columns]
    y = df['target']

    # Split data
    split_idx = int(len(df) * 0.7)
    X_train_full = X.iloc[:split_idx]
    X_test_full = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]

    # Feature selection
    print(f"\n{'='*70}")
    print("FEATURE SELECTION")
    print('='*70)
    X_train, X_test, top_features = select_top_features(X_train_full, y_train, X_test_full, n_features=20)
    print(f"Selected top 20 features from {len(feature_engineer.feature_columns)}")
    print('='*70)

    # ============================================================
    # STRATEGY 1: Feature Interactions
    # ============================================================
    print(f"\n\n{'='*70}")
    print("STRATEGY 1: FEATURE INTERACTIONS")
    print('='*70)

    X_train_interact = create_feature_interactions(X_train, top_features, n_interactions=5)
    X_test_interact = create_feature_interactions(X_test, top_features, n_interactions=5)

    model_interact = CatBoostClassifier(
        iterations=500, depth=10, learning_rate=0.05,
        random_state=42, verbose=0
    )
    model_interact.fit(X_train_interact, y_train)
    y_pred = model_interact.predict(X_test_interact)
    acc_interact = accuracy_score(y_test, y_pred)

    print(f"\nCatBoost with interactions: {acc_interact:.4f} ({acc_interact*100:.2f}%)")

    all_results.append({
        'Strategy': 'Feature Interactions',
        'Model': 'CatBoost',
        'Accuracy': acc_interact,
        'Accuracy %': f"{acc_interact*100:.2f}%"
    })

    # ============================================================
    # STRATEGY 2: Regime-Based Models
    # ============================================================
    regime_models, regime_df = train_regime_specific_models(
        df.iloc[:split_idx], X_train_full, y_train, top_features
    )

    # Average accuracy
    avg_regime_acc = regime_df['Accuracy'].mean()
    all_results.append({
        'Strategy': 'Regime-Based (Average)',
        'Model': 'CatBoost per regime',
        'Accuracy': avg_regime_acc,
        'Accuracy %': f"{avg_regime_acc*100:.2f}%"
    })

    # ============================================================
    # STRATEGY 3: Session-Based Models
    # ============================================================
    session_models, session_df = train_session_specific_models(
        df.iloc[:split_idx], X_train_full, y_train, top_features
    )

    # Average accuracy
    avg_session_acc = session_df['Accuracy'].mean()
    all_results.append({
        'Strategy': 'Session-Based (Average)',
        'Model': 'Random Forest per session',
        'Accuracy': avg_session_acc,
        'Accuracy %': f"{avg_session_acc*100:.2f}%"
    })

    # ============================================================
    # STRATEGY 4: Probability Calibration
    # ============================================================
    calibration_df = train_calibrated_models(X_train, y_train, X_test, y_test)

    best_calibrated = calibration_df.loc[calibration_df['Calibrated Acc'].idxmax()]
    all_results.append({
        'Strategy': 'Probability Calibration',
        'Model': f"{best_calibrated['Model']} (Calibrated)",
        'Accuracy': best_calibrated['Calibrated Acc'],
        'Accuracy %': f"{best_calibrated['Calibrated Acc']*100:.2f}%"
    })

    # ============================================================
    # STRATEGY 5: Advanced Stacking
    # ============================================================
    stacking_df = train_advanced_stacking(X_train, y_train, X_test, y_test)

    best_stacking = stacking_df.iloc[0]
    all_results.append({
        'Strategy': 'Advanced Stacking',
        'Model': f"4 models + {best_stacking['Meta-Learner']}",
        'Accuracy': best_stacking['Accuracy'],
        'Accuracy %': best_stacking['Accuracy %']
    })

    # ============================================================
    # FINAL COMPARISON
    # ============================================================
    print(f"\n\n{'='*70}")
    print("FINAL COMPARISON OF ALL STRATEGIES")
    print('='*70 + "\n")

    final_df = pd.DataFrame(all_results).sort_values('Accuracy', ascending=False)
    print(final_df.to_string(index=False))

    best_strategy = final_df.iloc[0]
    baseline = 0.5349

    print(f"\n{'='*70}")
    print(f"🏆 BEST STRATEGY: {best_strategy['Strategy']}")
    print(f"   Model: {best_strategy['Model']}")
    print(f"   Accuracy: {best_strategy['Accuracy']:.4f} ({best_strategy['Accuracy']*100:.2f}%)")
    print(f"\n📈 Baseline: {baseline:.4f} (53.49%)")

    improvement = (best_strategy['Accuracy'] - baseline) * 100
    if improvement > 0:
        print(f"📊 Improvement: +{improvement:.2f} percentage points ✅")
    else:
        print(f"📊 Change: {improvement:.2f} percentage points")

    print("="*70 + "\n")

    # Save results
    output_file = f"{config.OUTPUT_DIR}/advanced_ml_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("ADVANCED ML STRATEGIES RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("ALL STRATEGIES:\n")
        f.write("-"*70 + "\n")
        f.write(final_df.to_string(index=False))
        f.write(f"\n\n{'='*70}\n")
        f.write(f"BEST STRATEGY: {best_strategy['Strategy']}\n")
        f.write(f"Model: {best_strategy['Model']}\n")
        f.write(f"Accuracy: {best_strategy['Accuracy']:.4f} ({best_strategy['Accuracy']*100:.2f}%)\n")
        f.write(f"Baseline: {baseline:.4f} (53.49%)\n")
        f.write(f"Improvement: {improvement:+.2f} percentage points\n")
        f.write("="*70 + "\n\n")

        f.write("REGIME-BASED MODELS:\n")
        f.write("-"*70 + "\n")
        f.write(regime_df.to_string(index=False))
        f.write("\n\n")

        f.write("SESSION-BASED MODELS:\n")
        f.write("-"*70 + "\n")
        f.write(session_df.to_string(index=False))
        f.write("\n\n")

        f.write("CALIBRATED MODELS:\n")
        f.write("-"*70 + "\n")
        f.write(calibration_df.to_string(index=False))
        f.write("\n\n")

        f.write("STACKING ENSEMBLES:\n")
        f.write("-"*70 + "\n")
        f.write(stacking_df.to_string(index=False))
        f.write("\n")

    print(f"Results saved to: {output_file}\n")

    # ============================================================
    # SAVE BEST MODEL
    # ============================================================
    # ============================================================
    # SAVE BEST MODEL AND GENERATE FINAL OUTPUT
    # ============================================================
    print(f"\n{'='*70}")
    print("SAVING BEST MODEL AND GENERATING FINAL OUTPUT")
    print('='*70)

    # Train the best model on full data and save it
    print(f"\nRetraining best model (Calibrated Random Forest) on full training data...")

    best_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )

    # Apply probability calibration
    calibrated_model = CalibratedClassifierCV(best_model, cv=3, method='sigmoid')
    calibrated_model.fit(X_train, y_train)

    # Test accuracy
    y_pred_final = calibrated_model.predict(X_test)
    final_accuracy = accuracy_score(y_test, y_pred_final)

    print(f"Final model accuracy on test set: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

    # Save model
    import joblib
    import os
    os.makedirs(config.MODELS_DIR, exist_ok=True)

    model_path = os.path.join(config.MODELS_DIR, 'production_model.pkl')
    joblib.dump(calibrated_model, model_path)

    print(f"✅ Model saved to: {model_path}")
    print(f"   Model type: Calibrated Random Forest")
    print(f"   Features used: Top 20 selected features")
    print(f"   Test accuracy: {final_accuracy:.4f}")
    print('='*70 + "\n")

    # ============================================================
    # CREATE FINAL OUTPUT USING EVALUATOR APPROACH
    # ============================================================
    print(f"\n{'='*70}")
    print("GENERATING FINAL OUTPUT (EVALUATOR STYLE)")
    print('='*70)

    # Create final_output directory in root
    final_output_dir = 'final_output'
    os.makedirs(final_output_dir, exist_ok=True)

    # Get test data portion of original df
    test_df_with_ohlc = df_test.copy()
    
    # Add predictions to test dataframe
    test_df_with_ohlc['predicted'] = y_pred_final
    
    # Generate model_call column (1 = buy, 0 = sell)
    test_df_with_ohlc['model_call'] = test_df_with_ohlc['predicted'].apply(
        lambda x: 'buy' if x == 1 else 'sell'
    )
    
    signal_counts = test_df_with_ohlc['model_call'].value_counts()
    print(f"\nTrading Signals:")
    print(f"  Buy signals:  {signal_counts.get('buy', 0)}")
    print(f"  Sell signals: {signal_counts.get('sell', 0)}")

    # Calculate cumulative PnL
    print("\nCalculating cumulative PnL...")
    
    # Ensure dataframe is sorted by timestamp
    test_df_with_ohlc = test_df_with_ohlc.sort_values('timestamp').reset_index(drop=True)
    
    # Initialize PnL column
    model_pnl_values = []
    cumulative_pnl = 0
    
    # Iterate through each row to calculate cumulative PnL
    for idx, row in test_df_with_ohlc.iterrows():
        if row['model_call'] == 'buy':
            cumulative_pnl -= row['close']  # Pay to buy
        else:  # sell
            cumulative_pnl += row['close']  # Receive from selling
        
        model_pnl_values.append(cumulative_pnl)
    
    # Add PnL column
    test_df_with_ohlc['model_pnl'] = model_pnl_values
    
    # Calculate PnL statistics
    final_pnl = model_pnl_values[-1]
    max_pnl = max(model_pnl_values)
    min_pnl = min(model_pnl_values)
    
    print(f"  Final PnL: {final_pnl:,.2f}")
    print(f"  Max PnL:   {max_pnl:,.2f}")
    print(f"  Min PnL:   {min_pnl:,.2f}")

    # Create final output with required columns
    output_columns = ['timestamp', 'close', 'predicted', 'model_call', 'model_pnl']
    final_predictions_df = test_df_with_ohlc[output_columns].copy()
    
    # Rename timestamp column to match expected format (capitalize)
    final_predictions_df.columns = ['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']
    
    # Save final predictions
    predictions_path = os.path.join(final_output_dir, 'final_predictions.csv')
    final_predictions_df.to_csv(predictions_path, index=False)
    
    print(f"\n✅ Final predictions saved to: {predictions_path}")
    print(f"   Total rows: {len(final_predictions_df)}")
    print(f"\nFirst few rows:")
    print(final_predictions_df.head(5).to_string(index=False))
    print(f"\nLast few rows:")
    print(final_predictions_df.tail(5).to_string(index=False))

    # Also save the test data with all features for reference
    test_data_df = X_test.copy()
    test_data_df['target'] = y_test.values
    test_data_df['predicted'] = y_pred_final
    
    test_data_path = os.path.join(final_output_dir, 'used_test_data.csv')
    test_data_df.to_csv(test_data_path, index=False)
    print(f"\n✅ Test data with features saved to: {test_data_path}")
    print(f"   Shape: {test_data_df.shape}")

    # Save comprehensive metrics report
    print(f"\n{'='*70}")
    print("SAVING METRICS REPORT")
    print('='*70)
    
    report_path = os.path.join(final_output_dir, 'model_evaluation_report.txt')
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NIFTY PRICE PREDICTION - MODEL EVALUATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        # Best Strategy Summary
        f.write("BEST STRATEGY SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Strategy: {best_strategy['Strategy']}\n")
        f.write(f"Model: {best_strategy['Model']}\n")
        f.write(f"Accuracy: {best_strategy['Accuracy']:.4f} ({best_strategy['Accuracy']*100:.2f}%)\n")
        f.write(f"Baseline: {baseline:.4f} (53.49%)\n")
        f.write(f"Improvement: {improvement:+.2f} percentage points\n")
        f.write("\n")
        
        # All Strategies Comparison
        f.write("ALL STRATEGIES COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(final_df.to_string(index=False))
        f.write("\n\n")
        
        # Test Set Metrics
        f.write("TEST SET PERFORMANCE\n")
        f.write("-"*70 + "\n")
        f.write(f"Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)\n")
        f.write(f"Test samples: {len(y_test)}\n")
        f.write("\n")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred_final)
        f.write("Confusion Matrix:\n")
        f.write(f"                 Predicted\n")
        f.write(f"                 0      1\n")
        f.write(f"Actual  0     {cm[0][0]:5d}  {cm[0][1]:5d}\n")
        f.write(f"        1     {cm[1][0]:5d}  {cm[1][1]:5d}\n")
        f.write("\n")
        
        # Trading Signals
        f.write("TRADING SIGNALS\n")
        f.write("-"*70 + "\n")
        f.write(f"Buy signals:  {signal_counts.get('buy', 0)}\n")
        f.write(f"Sell signals: {signal_counts.get('sell', 0)}\n")
        f.write("\n")
        
        # PnL Summary
        f.write("PnL SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"Final PnL: {final_pnl:,.2f}\n")
        f.write(f"Max PnL:   {max_pnl:,.2f}\n")
        f.write(f"Min PnL:   {min_pnl:,.2f}\n")
        f.write("\n")
        
        # Feature Information
        f.write("FEATURE INFORMATION\n")
        f.write("-"*70 + "\n")
        f.write(f"Total features available: {len(feature_engineer.feature_columns)}\n")
        f.write(f"Features used: {len(top_features)}\n")
        f.write("\nTop 20 features:\n")
        for i, feat in enumerate(top_features, 1):
            f.write(f"  {i:2d}. {feat}\n")
        f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"✅ Metrics report saved to: {report_path}")
    print('='*70 + "\n")

    return calibrated_model, top_features


if __name__ == "__main__":
    main()