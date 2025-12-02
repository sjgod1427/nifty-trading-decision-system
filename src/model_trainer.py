# """
# Model Trainer Module
# Trains multiple ML models and compares their performance
# """

# import numpy as np
# import pandas as pd
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.preprocessing import StandardScaler
# import lightgbm as lgb
# import xgboost as xgb
# import pickle
# import os
# import warnings
# warnings.filterwarnings('ignore')

# class ModelTrainer:
#     """
#     Trains and compares multiple ML models for binary classification
#     """

#     def __init__(self, X_train, X_test, y_train, y_test, models_dir, test_df=None):
#         """
#         Initialize ModelTrainer

#         Args:
#             X_train: Training features
#             X_test: Testing features
#             y_train: Training labels
#             y_test: Testing labels
#             models_dir: Directory to save trained models
#             test_df: Test dataframe with timestamp and close price (for saving predictions)
#         """
#         self.X_train = X_train
#         self.X_test = X_test
#         self.y_train = y_train
#         self.y_test = y_test
#         self.models_dir = models_dir
#         self.test_df = test_df
#         self.models = {}
#         self.results = {}

#     def train_logistic_regression(self, params, use_scaling=True):
#         """
#         Train Logistic Regression model with optional feature scaling

#         Args:
#             params (dict): Model parameters
#             use_scaling (bool): Whether to scale features (recommended for LR)

#         Returns:
#             LogisticRegression: Trained model
#         """
#         print("\n" + "-"*60)
#         print("Training Logistic Regression...")
#         print("-"*60)

#         if use_scaling:
#             print("Scaling features for Logistic Regression...")
#             scaler = StandardScaler()
#             X_train_scaled = scaler.fit_transform(self.X_train)
#             X_test_scaled = scaler.transform(self.X_test)
#         else:
#             X_train_scaled = self.X_train
#             X_test_scaled = self.X_test

#         model = LogisticRegression(**params)
#         model.fit(X_train_scaled, self.y_train)

#         # Store scaler for predictions
#         self.scaler = scaler if use_scaling else None

#         # Predictions
#         y_pred = model.predict(X_test_scaled)

#         # Calculate metrics
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred, zero_division=0)
#         recall = recall_score(self.y_test, y_pred, zero_division=0)
#         f1 = f1_score(self.y_test, y_pred, zero_division=0)

#         self.models['logistic_regression'] = model
#         self.results['logistic_regression'] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'predictions': y_pred
#         }

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")

#         return model

#     def train_random_forest(self, params):
#         """
#         Train Random Forest model

#         Args:
#             params (dict): Model parameters

#         Returns:
#             RandomForestClassifier: Trained model
#         """
#         print("\n" + "-"*60)
#         print("Training Random Forest...")
#         print("-"*60)

#         model = RandomForestClassifier(**params)
#         model.fit(self.X_train, self.y_train)

#         # Predictions
#         y_pred = model.predict(self.X_test)

#         # Calculate metrics
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred, zero_division=0)
#         recall = recall_score(self.y_test, y_pred, zero_division=0)
#         f1 = f1_score(self.y_test, y_pred, zero_division=0)

#         self.models['random_forest'] = model
#         self.results['random_forest'] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'predictions': y_pred
#         }

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")

#         return model

#     def train_lightgbm(self, params):
#         """
#         Train LightGBM model

#         Args:
#             params (dict): Model parameters

#         Returns:
#             lgb.LGBMClassifier: Trained model
#         """
#         print("\n" + "-"*60)
#         print("Training LightGBM...")
#         print("-"*60)

#         model = lgb.LGBMClassifier(**params)
#         model.fit(self.X_train, self.y_train)

#         # Predictions
#         y_pred = model.predict(self.X_test)

#         # Calculate metrics
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred, zero_division=0)
#         recall = recall_score(self.y_test, y_pred, zero_division=0)
#         f1 = f1_score(self.y_test, y_pred, zero_division=0)

#         self.models['lightgbm'] = model
#         self.results['lightgbm'] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'predictions': y_pred
#         }

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")

#         return model

#     def train_xgboost(self, params):
#         """
#         Train XGBoost model

#         Args:
#             params (dict): Model parameters

#         Returns:
#             xgb.XGBClassifier: Trained model
#         """
#         print("\n" + "-"*60)
#         print("Training XGBoost...")
#         print("-"*60)

#         model = xgb.XGBClassifier(**params)
#         model.fit(self.X_train, self.y_train)

#         # Predictions
#         y_pred = model.predict(self.X_test)

#         # Calculate metrics
#         accuracy = accuracy_score(self.y_test, y_pred)
#         precision = precision_score(self.y_test, y_pred, zero_division=0, average='weighted')
#         recall = recall_score(self.y_test, y_pred, zero_division=0, average='weighted')
#         f1 = f1_score(self.y_test, y_pred, zero_division=0, average='weighted')

#         self.models['xgboost'] = model
#         self.results['xgboost'] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'predictions': y_pred
#         }

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")

#         return model

#     def train_lstm(self, params=None, sequence_length=10):
#         """
#         Train LSTM model for time series prediction

#         Args:
#             params (dict): Model parameters
#             sequence_length (int): Number of time steps to look back

#         Returns:
#             Keras Model: Trained LSTM model
#         """
#         print("\n" + "-"*60)
#         print("Training LSTM...")
#         print("-"*60)

#         try:
#             from tensorflow import keras
#             from tensorflow.keras.models import Sequential
#             from tensorflow.keras.layers import LSTM, Dense, Dropout
#             from tensorflow.keras.optimizers import Adam
#         except ImportError:
#             print("❌ TensorFlow not installed. Skipping LSTM.")
#             print("   Install with: pip install tensorflow")
#             return None

#         # Default parameters
#         if params is None:
#             params = {
#                 'units': 64,
#                 'dropout': 0.2,
#                 'learning_rate': 0.001,
#                 'epochs': 50,
#                 'batch_size': 32
#             }

#         # Prepare sequences
#         def create_sequences(X, y, seq_length):
#             X_seq, y_seq = [], []
#             for i in range(len(X) - seq_length):
#                 X_seq.append(X[i:i + seq_length])
#                 y_seq.append(y[i + seq_length])
#             return np.array(X_seq), np.array(y_seq)

#         # Scale features
#         print("Scaling features for LSTM...")
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(self.X_train)
#         X_test_scaled = scaler.transform(self.X_test)

#         # Create sequences
#         print(f"Creating sequences (length={sequence_length})...")
#         X_train_seq, y_train_seq = create_sequences(
#             X_train_scaled, self.y_train.values, sequence_length
#         )
#         X_test_seq, y_test_seq = create_sequences(
#             X_test_scaled, self.y_test.values, sequence_length
#         )

#         print(f"Training sequences: {X_train_seq.shape}")
#         print(f"Testing sequences: {X_test_seq.shape}")

#         # Build LSTM model
#         model = Sequential([
#             LSTM(params['units'], return_sequences=True,
#                  input_shape=(sequence_length, X_train_scaled.shape[1])),
#             Dropout(params['dropout']),
#             LSTM(params['units'] // 2, return_sequences=False),
#             Dropout(params['dropout']),
#             Dense(32, activation='relu'),
#             Dense(1, activation='sigmoid')
#         ])

#         model.compile(
#             optimizer=Adam(learning_rate=params['learning_rate']),
#             loss='binary_crossentropy',
#             metrics=['accuracy']
#         )

#         # Train model
#         print("Training LSTM model...")
#         model.fit(
#             X_train_seq, y_train_seq,
#             epochs=params['epochs'],
#             batch_size=params['batch_size'],
#             validation_split=0.2,
#             verbose=0
#         )

#         # Predictions
#         y_pred_proba = model.predict(X_test_seq, verbose=0)
#         y_pred = (y_pred_proba > 0.5).astype(int).flatten()

#         # Calculate metrics
#         accuracy = accuracy_score(y_test_seq, y_pred)
#         precision = precision_score(y_test_seq, y_pred, zero_division=0)
#         recall = recall_score(y_test_seq, y_pred, zero_division=0)
#         f1 = f1_score(y_test_seq, y_pred, zero_division=0)

#         # Store model and results
#         self.models['lstm'] = model
#         self.results['lstm'] = {
#             'accuracy': accuracy,
#             'precision': precision,
#             'recall': recall,
#             'f1': f1,
#             'predictions': y_pred,
#             'scaler': scaler,
#             'sequence_length': sequence_length
#         }

#         print(f"Accuracy:  {accuracy:.4f}")
#         print(f"Precision: {precision:.4f}")
#         print(f"Recall:    {recall:.4f}")
#         print(f"F1-Score:  {f1:.4f}")

#         return model

#     def save_model_predictions(self, model_name, output_dir='output'):
#         """
#         Save predictions for a specific model to CSV

#         Args:
#             model_name (str): Name of the model
#             output_dir (str): Directory to save predictions
#         """
#         if self.test_df is None:
#             print(f"⚠️ Cannot save predictions: test_df not provided")
#             return

#         if model_name not in self.results:
#             print(f"⚠️ Model {model_name} not found in results")
#             return

#         os.makedirs(output_dir, exist_ok=True)

#         # Get predictions
#         predictions = self.results[model_name]['predictions']

#         # Create predictions dataframe
#         pred_df = self.test_df.copy()
#         pred_df['predicted'] = predictions
#         pred_df['model_call'] = pred_df['predicted'].apply(
#             lambda x: 'buy' if x == 1 else 'sell'
#         )

#         # Calculate PnL
#         pred_df = pred_df.sort_values('timestamp').reset_index(drop=True)
#         model_pnl_values = []
#         cumulative_pnl = 0

#         for idx, row in pred_df.iterrows():
#             if row['model_call'] == 'buy':
#                 cumulative_pnl -= row['close']
#             else:
#                 cumulative_pnl += row['close']
#             model_pnl_values.append(cumulative_pnl)

#         pred_df['model_pnl'] = model_pnl_values

#         # Select output columns
#         output_df = pred_df[['timestamp', 'close', 'predicted', 'model_call', 'model_pnl']].copy()
#         output_df.columns = ['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']

#         # Save to CSV
#         output_path = os.path.join(output_dir, f'predictions_{model_name}.csv')
#         output_df.to_csv(output_path, index=False)

#         print(f"✅ Predictions saved: {output_path}")

#     def compare_models(self):
#         """
#         Compare all trained models and select the best one

#         Returns:
#             tuple: (best_model_name, best_model, comparison_df)
#         """
#         print("\n" + "="*60)
#         print("MODEL COMPARISON")
#         print("="*60)

#         # Create comparison dataframe
#         comparison_data = []
#         for model_name, metrics in self.results.items():
#             comparison_data.append({
#                 'Model': model_name.replace('_', ' ').title(),
#                 'Accuracy': metrics['accuracy'],
#                 'Precision': metrics['precision'],
#                 'Recall': metrics['recall'],
#                 'F1-Score': metrics['f1']
#             })

#         comparison_df = pd.DataFrame(comparison_data)
#         comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

#         print(comparison_df.to_string(index=False))
#         print("="*60)

#         # Select best model based on accuracy
#         best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
#         best_model = self.models[best_model_name]

#         print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
#         print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
#         print("="*60 + "\n")

#         return best_model_name, best_model, comparison_df

#     def save_model(self, model, model_name):
#         """
#         Save trained model to disk

#         Args:
#             model: Trained model object
#             model_name (str): Name of the model
#         """
#         os.makedirs(self.models_dir, exist_ok=True)
#         model_path = os.path.join(self.models_dir, f"{model_name}.pkl")

#         with open(model_path, 'wb') as f:
#             pickle.dump(model, f)

#         print(f"Model saved: {model_path}")

#     def get_feature_importance(self, model, model_name, feature_names, top_n=15):
#         """
#         Get feature importance for tree-based models

#         Args:
#             model: Trained model
#             model_name (str): Name of the model
#             feature_names (list): List of feature names
#             top_n (int): Number of top features to return

#         Returns:
#             pd.DataFrame: Feature importance dataframe
#         """
#         if model_name in ['random_forest', 'lightgbm', 'xgboost']:
#             importances = model.feature_importances_
#             importance_df = pd.DataFrame({
#                 'feature': feature_names,
#                 'importance': importances
#             })
#             importance_df = importance_df.sort_values('importance', ascending=False)
#             return importance_df.head(top_n)
#         else:
#             return None


"""
Model Trainer Module
Trains multiple ML models and compares their performance
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    Trains and compares multiple ML models for binary classification
    """

    def __init__(self, X_train, X_test, y_train, y_test, models_dir, test_df=None):
        """
        Initialize ModelTrainer

        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            models_dir: Directory to save trained models
            test_df: Test dataframe with timestamp and close price (for saving predictions)
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_dir = models_dir
        self.test_df = test_df
        self.models = {}
        self.results = {}

    def train_logistic_regression(self, params, use_scaling=True):
        """
        Train Logistic Regression model with optional feature scaling

        Args:
            params (dict): Model parameters
            use_scaling (bool): Whether to scale features (recommended for LR)

        Returns:
            LogisticRegression: Trained model
        """
        print("\n" + "-"*60)
        print("Training Logistic Regression...")
        print("-"*60)

        if use_scaling:
            print("Scaling features for Logistic Regression...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.X_train)
            X_test_scaled = scaler.transform(self.X_test)
        else:
            X_train_scaled = self.X_train
            X_test_scaled = self.X_test

        model = LogisticRegression(**params)
        model.fit(X_train_scaled, self.y_train)

        # Store scaler for predictions
        self.scaler = scaler if use_scaling else None

        # Predictions
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_random_forest(self, params):
        """
        Train Random Forest model

        Args:
            params (dict): Model parameters

        Returns:
            RandomForestClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training Random Forest...")
        print("-"*60)

        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_lightgbm(self, params):
        """
        Train LightGBM model

        Args:
            params (dict): Model parameters

        Returns:
            lgb.LGBMClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training LightGBM...")
        print("-"*60)

        model = lgb.LGBMClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def train_xgboost(self, params):
        """
        Train XGBoost model

        Args:
            params (dict): Model parameters

        Returns:
            xgb.XGBClassifier: Trained model
        """
        print("\n" + "-"*60)
        print("Training XGBoost...")
        print("-"*60)

        model = xgb.XGBClassifier(**params)
        model.fit(self.X_train, self.y_train)

        # Predictions
        y_pred = model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0, average='weighted')
        recall = recall_score(self.y_test, y_pred, zero_division=0, average='weighted')
        f1 = f1_score(self.y_test, y_pred, zero_division=0, average='weighted')

        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': y_pred
        }

        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")

        return model

    def save_model_predictions(self, model_name, output_dir='output'):
        """
        Save predictions for a specific model to CSV

        Args:
            model_name (str): Name of the model
            output_dir (str): Directory to save predictions
        """
        if self.test_df is None:
            print(f"⚠️ Cannot save predictions: test_df not provided")
            return

        if model_name not in self.results:
            print(f"⚠️ Model {model_name} not found in results")
            return

        os.makedirs(output_dir, exist_ok=True)

        # Get predictions
        predictions = self.results[model_name]['predictions']

        # Create predictions dataframe
        pred_df = self.test_df.copy()
        
        # Verify length match
        if len(predictions) != len(pred_df):
            print(f"⚠️ Length mismatch: predictions={len(predictions)}, dataframe={len(pred_df)}")
            print(f"   Skipping {model_name} predictions save.")
            return
        
        pred_df['predicted'] = predictions
        pred_df['model_call'] = pred_df['predicted'].apply(
            lambda x: 'buy' if x == 1 else 'sell'
        )

        # Calculate PnL
        pred_df = pred_df.sort_values('timestamp').reset_index(drop=True)
        model_pnl_values = []
        cumulative_pnl = 0

        for idx, row in pred_df.iterrows():
            if row['model_call'] == 'buy':
                cumulative_pnl -= row['close']
            else:
                cumulative_pnl += row['close']
            model_pnl_values.append(cumulative_pnl)

        pred_df['model_pnl'] = model_pnl_values

        # Select output columns
        output_df = pred_df[['timestamp', 'close', 'predicted', 'model_call', 'model_pnl']].copy()
        output_df.columns = ['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']

        # Save to CSV
        output_path = os.path.join(output_dir, f'predictions_{model_name}.csv')
        output_df.to_csv(output_path, index=False)

        print(f"✅ Predictions saved: {output_path}")

    def compare_models(self):
        """
        Compare all trained models and select the best one

        Returns:
            tuple: (best_model_name, best_model, comparison_df)
        """
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)

        # Create comparison dataframe
        comparison_data = []
        for model_name, metrics in self.results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1']
            })

        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Accuracy', ascending=False)

        print(comparison_df.to_string(index=False))
        print("="*60)

        # Select best model based on accuracy
        best_model_name = comparison_df.iloc[0]['Model'].lower().replace(' ', '_')
        best_model = self.models[best_model_name]

        print(f"\nBest Model: {comparison_df.iloc[0]['Model']}")
        print(f"Best Accuracy: {comparison_df.iloc[0]['Accuracy']:.4f}")
        print("="*60 + "\n")

        return best_model_name, best_model, comparison_df

    def save_model(self, model, model_name):
        """
        Save trained model to disk

        Args:
            model: Trained model object
            model_name (str): Name of the model
        """
        os.makedirs(self.models_dir, exist_ok=True)
        model_path = os.path.join(self.models_dir, f"{model_name}.pkl")

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"Model saved: {model_path}")

    def get_feature_importance(self, model, model_name, feature_names, top_n=15):
        """
        Get feature importance for tree-based models

        Args:
            model: Trained model
            model_name (str): Name of the model
            feature_names (list): List of feature names
            top_n (int): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if model_name in ['random_forest', 'lightgbm', 'xgboost']:
            importances = model.feature_importances_
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            })
            importance_df = importance_df.sort_values('importance', ascending=False)
            return importance_df.head(top_n)
        else:
            return None