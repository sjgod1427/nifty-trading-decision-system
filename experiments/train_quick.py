"""
Quick Test - See immediate improvements with new features (no tuning)
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

print("\n" + "="*70)
print("QUICK ACCURACY TEST - With New Features")
print("="*70 + "\n")

# Load data
data_loader = DataLoader(config.DATA_PATH)
data_loader.load_data()
data_loader.preprocess()
df = data_loader.create_target(
    min_movement_pct=config.MIN_MOVEMENT_PCT,
    use_multiclass=config.USE_MULTICLASS,
    up_threshold=config.UP_THRESHOLD,
    down_threshold=config.DOWN_THRESHOLD
)

# Feature engineering with NEW indicators
feature_engineer = FeatureEngineer(
    df=df,
    sma_windows=config.SMA_WINDOWS,
    rsi_period=config.RSI_PERIOD,
    volatility_window=config.VOLATILITY_WINDOW,
    lag_periods=config.LAG_PERIODS
)
df, feature_columns = feature_engineer.create_all_features()
X, y = feature_engineer.get_feature_matrix()

print(f"Total features: {len(feature_columns)}")

# Split data
split_index = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Test models with improved configs
models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=500, max_depth=30, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=500, max_depth=20, learning_rate=0.05,
        random_state=42, n_jobs=-1
    ),
    'LightGBM': lgb.LGBMClassifier(
        n_estimators=500, max_depth=20, learning_rate=0.05,
        class_weight='balanced', random_state=42, verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=10, learning_rate=0.05,
        random_state=42, verbose=0
    )
}

print("\n" + "="*70)
print("TRAINING & EVALUATION")
print("="*70)

results = []
for name, model in models.items():
    print(f"\n{name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Accuracy %': f"{accuracy*100:.2f}%"
    })

# Results table
print("\n" + "="*70)
print("RESULTS COMPARISON")
print("="*70 + "\n")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)
print(results_df.to_string(index=False))

best_accuracy = results_df.iloc[0]['Accuracy']
baseline = 0.5349

print(f"\n{'='*70}")
print(f"🏆 BEST: {results_df.iloc[0]['Model']} - {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"📈 Baseline: {baseline:.4f} (53.49%)")
print(f"📊 Improvement: {(best_accuracy - baseline)*100:+.2f} percentage points")
print("="*70 + "\n")
