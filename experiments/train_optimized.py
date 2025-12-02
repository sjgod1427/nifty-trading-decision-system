"""
NIFTY Trading System - Quick Optimization Script
Applies key improvements to boost accuracy
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer

# Import models
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier


def quick_tune_model(X_train, y_train, model_type='random_forest', n_iter=10):
    """Quick hyperparameter tuning"""
    print(f"\nTuning {model_type}...")
    
    tscv = TimeSeriesSplit(n_splits=3)
    
    if model_type == 'random_forest':
        param_dist = {
            'n_estimators': [300, 500, 700],
            'max_depth': [20, 30, None],
            'min_samples_split': [10, 20],
            'min_samples_leaf': [5, 10],
            'max_features': ['sqrt', None]
        }
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        
    elif model_type == 'xgboost':
        param_dist = {
            'n_estimators': [500, 700],
            'max_depth': [15, 20],
            'learning_rate': [0.05, 0.1],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
        
    elif model_type == 'lightgbm':
        param_dist = {
            'n_estimators': [500, 700],
            'max_depth': [15, 20],
            'learning_rate': [0.05, 0.1],
            'num_leaves': [63, 127]
        }
        base_model = lgb.LGBMClassifier(random_state=42, verbose=-1, class_weight='balanced')
        
    elif model_type == 'catboost':
        param_dist = {
            'iterations': [500, 700],
            'depth': [8, 10],
            'learning_rate': [0.05, 0.1]
        }
        base_model = CatBoostClassifier(random_state=42, verbose=0)
    
    search = RandomizedSearchCV(
        base_model, param_dist, n_iter=n_iter, cv=tscv,
        scoring='accuracy', n_jobs=-1, random_state=42, verbose=0
    )
    search.fit(X_train, y_train)
    
    print(f"  Best CV Score: {search.best_score_:.4f}")
    return search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics"""
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'predictions': y_pred
    }
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    return metrics


def main():
    print("\n" + "="*70)
    print("NIFTY TRADING SYSTEM - OPTIMIZED TRAINING")
    print("="*70 + "\n")
    
    # Load data
    print("Loading data...")
    data_loader = DataLoader(config.DATA_PATH)
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.create_target(
        min_movement_pct=config.MIN_MOVEMENT_PCT,
        use_multiclass=config.USE_MULTICLASS,
        up_threshold=config.UP_THRESHOLD,
        down_threshold=config.DOWN_THRESHOLD
    )
    
    # Feature engineering (with new indicators)
    print("\n" + "="*70)
    feature_engineer = FeatureEngineer(
        df=df,
        sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )
    df, feature_columns = feature_engineer.create_all_features()
    X, y = feature_engineer.get_feature_matrix()
    
    # Train/Test split
    print("="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    split_index = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
    
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    print(f"Training: {len(X_train)} samples | Testing: {len(X_test)} samples")
    print(f"Features: {len(feature_columns)}")
    
    # Hyperparameter tuning
    print("\n" + "="*70)
    print("HYPERPARAMETER TUNING (This will take a few minutes...)")
    print("="*70)
    
    models = {}
    
    print("\n[1/4] Random Forest...")
    models['Random Forest'] = quick_tune_model(X_train, y_train, 'random_forest', n_iter=10)
    
    print("\n[2/4] XGBoost...")
    models['XGBoost'] = quick_tune_model(X_train, y_train, 'xgboost', n_iter=10)
    
    print("\n[3/4] LightGBM...")
    models['LightGBM'] = quick_tune_model(X_train, y_train, 'lightgbm', n_iter=10)
    
    print("\n[4/4] CatBoost...")
    models['CatBoost'] = quick_tune_model(X_train, y_train, 'catboost', n_iter=10)
    
    # Evaluate all models
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = {}
    for name, model in models.items():
        results[name] = evaluate_model(model, X_test, y_test, name)
    
    # Create ensemble
    print("\n" + "="*70)
    print("BUILDING ENSEMBLE")
    print("="*70)
    
    estimators = [(name, model) for name, model in models.items()]
    voting_clf = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
    print("\nTraining Voting Classifier...")
    voting_clf.fit(X_train, y_train)
    results['Voting Ensemble'] = evaluate_model(voting_clf, X_test, y_test, 'Voting Ensemble')
    
    # Final comparison
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70 + "\n")
    
    comparison = pd.DataFrame([
        {
            'Model': name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1']
        }
        for name, metrics in results.items()
    ]).sort_values('Accuracy', ascending=False)
    
    print(comparison.to_string(index=False))
    
    # Best model details
    best_model_name = comparison.iloc[0]['Model']
    best_accuracy = comparison.iloc[0]['Accuracy']
    
    print(f"\n{'='*70}")
    print(f"🏆 BEST MODEL: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    # Detailed report for best model
    y_pred = results[best_model_name]['predictions']
    
    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
    
    print("="*70)
    
    # Save results
    output_file = f"{config.OUTPUT_DIR}/optimized_results.txt"
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("OPTIMIZED MODEL RESULTS\n")
        f.write("="*70 + "\n\n")
        f.write("MODEL COMPARISON\n")
        f.write("-"*70 + "\n")
        f.write(comparison.to_string(index=False))
        f.write(f"\n\n{'='*70}\n")
        f.write(f"BEST MODEL: {best_model_name}\n")
        f.write(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)\n")
        f.write("="*70 + "\n\n")
        f.write("CONFUSION MATRIX:\n")
        f.write(str(cm) + "\n\n")
        f.write("CLASSIFICATION REPORT:\n")
        f.write(classification_report(y_test, y_pred, target_names=['Down (0)', 'Up (1)']))
    
    print(f"\nResults saved to: {output_file}")
    print("\n" + "="*70)
    print("OPTIMIZATION COMPLETE!")
    print("="*70 + "\n")
    
    # Show improvement
    baseline_accuracy = 0.5349
    improvement = (best_accuracy - baseline_accuracy) * 100
    improvement_pct = (improvement / baseline_accuracy) * 100
    
    print(f"Baseline Accuracy: {baseline_accuracy:.4f} (53.49%)")
    print(f"New Accuracy:      {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print(f"Improvement:       {improvement:+.2f} percentage points ({improvement_pct:+.1f}%)")
    print()


if __name__ == "__main__":
    main()
