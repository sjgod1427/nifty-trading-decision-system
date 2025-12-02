"""
LSTM Price Prediction Model
Standalone implementation for NIFTY price prediction using LSTM
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import pickle
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping


class LSTMPricePredictor:
    """
    LSTM model for price prediction
    """
    
    def __init__(self, data_path, time_step=100, train_ratio=0.70):
        """
        Initialize LSTM predictor
        
        Args:
            data_path (str): Path to CSV file
            time_step (int): Number of time steps to look back
            train_ratio (float): Ratio of training data
        """
        self.data_path = data_path
        self.time_step = time_step
        self.train_ratio = train_ratio
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess the data
        """
        print("\n" + "="*60)
        print("LOADING AND PREPROCESSING DATA")
        print("="*60)
        
        # Load data
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {len(self.df)} rows")
        
        # Drop unnecessary columns
        if 'id' in self.df.columns:
            self.df = self.df.drop(['id', 'symbol', 'exchange'], axis=1)
        
        # Convert timestamp
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        
        # Extract close prices
        close_prices = self.df['close'].values.reshape(-1, 1)
        
        # Scale data
        scaled_data = self.scaler.fit_transform(close_prices)
        
        # Split into train and test
        training_size = int(len(scaled_data) * self.train_ratio)
        self.train_data = scaled_data[:training_size]
        self.test_data = scaled_data[training_size:]
        
        print(f"Training samples: {len(self.train_data)} ({self.train_ratio*100:.0f}%)")
        print(f"Testing samples: {len(self.test_data)} ({(1-self.train_ratio)*100:.0f}%)")
        print("="*60 + "\n")
        
    def create_dataset(self, dataset):
        """
        Create dataset with time steps
        
        Args:
            dataset: Scaled data array
            
        Returns:
            tuple: (X, y) arrays
        """
        dataX, dataY = [], []
        for i in range(len(dataset) - self.time_step - 1):
            a = dataset[i:(i + self.time_step), 0]
            dataX.append(a)
            dataY.append(dataset[i + self.time_step, 0])
        return np.array(dataX), np.array(dataY)
    
    def prepare_sequences(self):
        """
        Prepare training and testing sequences
        """
        print("="*60)
        print("PREPARING SEQUENCES")
        print("="*60)
        
        # Create datasets
        self.X_train, self.y_train = self.create_dataset(self.train_data)
        self.X_test, self.y_test = self.create_dataset(self.test_data)
        
        # Reshape for LSTM [samples, time_steps, features]
        self.X_train = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[1], 1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[1], 1)
        
        print(f"X_train shape: {self.X_train.shape}")
        print(f"y_train shape: {self.y_train.shape}")
        print(f"X_test shape: {self.X_test.shape}")
        print(f"y_test shape: {self.y_test.shape}")
        print("="*60 + "\n")
        
    def build_model(self):
        """
        Build LSTM model architecture
        """
        print("="*60)
        print("BUILDING LSTM MODEL")
        print("="*60)
        
        self.model = Sequential()
        self.model.add(LSTM(50, return_sequences=True, input_shape=(self.time_step, 1)))
        self.model.add(LSTM(50, ))
        # self.model.add(LSTM(50))
        self.model.add(Dense(1))
        
        self.model.compile(loss='mean_squared_error', optimizer='adam')
        
        print("Model architecture:")
        self.model.summary()
        print("="*60 + "\n")
        
    def train_model(self, epochs=100, batch_size=64):
        """
        Train the LSTM model
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
        """
        print("="*60)
        print("TRAINING MODEL")
        print("="*60)
        
        # Early stopping to prevent overfitting
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = self.model.fit(
            self.X_train, 
            self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,
            callbacks=[early_stop]
        )
        
        print("\nTraining completed!")
        print("="*60 + "\n")
        
        return history
    
    def make_predictions(self):
        """
        Make predictions on test data
        
        Returns:
            tuple: (predictions, actual_values)
        """
        print("="*60)
        print("MAKING PREDICTIONS")
        print("="*60)
        
        # Predict
        train_predict = self.model.predict(self.X_train, verbose=0)
        test_predict = self.model.predict(self.X_test, verbose=0)
        
        # Inverse transform predictions
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        y_train_actual = self.scaler.inverse_transform(self.y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
        
        print("Predictions completed!")
        print("="*60 + "\n")
        
        return train_predict, test_predict, y_train_actual, y_test_actual
    
    def calculate_metrics(self, y_actual, y_pred):
        """
        Calculate performance metrics
        
        Args:
            y_actual: Actual values
            y_pred: Predicted values
            
        Returns:
            dict: Dictionary of metrics
        """
        mse = mean_squared_error(y_actual, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        # Calculate accuracy as percentage (1 - MAPE)
        mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
        accuracy = 100 - mape
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'accuracy': accuracy
        }
    
    def generate_trading_signals(self, test_predict, y_test_actual):
        """
        Generate trading signals based on predictions
        
        Args:
            test_predict: Predicted prices
            y_test_actual: Actual prices
            
        Returns:
            pd.DataFrame: DataFrame with trading signals and PnL
        """
        # Get corresponding timestamps from test data
        test_start_idx = int(len(self.df) * self.train_ratio) + self.time_step + 1
        test_timestamps = self.df['timestamp'].iloc[test_start_idx:test_start_idx + len(test_predict)]
        
        # Create predictions dataframe
        pred_df = pd.DataFrame({
            'Timestamp': test_timestamps.values,
            'Close': y_test_actual.flatten(),
            'Predicted': test_predict.flatten()
        })
        
        # Generate trading signals: if predicted > current, buy (1), else sell (0)
        # Shift predicted to compare with next close
        pred_df['Predicted_shift'] = pred_df['Predicted'].shift(1)
        pred_df['Signal'] = (pred_df['Predicted_shift'] > pred_df['Close'].shift(1)).astype(int)
        
        # For first row, default to buy
        pred_df.loc[0, 'Signal'] = 1
        
        # Convert to model_call
        pred_df['model_call'] = pred_df['Signal'].apply(lambda x: 'buy' if x == 1 else 'sell')
        
        # Calculate PnL
        cumulative_pnl = 0
        pnl_values = []
        
        for idx, row in pred_df.iterrows():
            if row['model_call'] == 'buy':
                cumulative_pnl -= row['Close']
            else:
                cumulative_pnl += row['Close']
            pnl_values.append(cumulative_pnl)
        
        pred_df['model_pnl'] = pnl_values
        
        # Select final columns
        final_df = pred_df[['Timestamp', 'Close', 'Signal', 'model_call', 'model_pnl']].copy()
        final_df.rename(columns={'Signal': 'Predicted'}, inplace=True)
        
        return final_df
    
    def save_model(self, model_dir='models'):
        """
        Save trained model and scaler
        
        Args:
            model_dir (str): Directory to save model
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # Save Keras model
        model_path = os.path.join(model_dir, 'lstm_model.h5')
        self.model.save(model_path)
        print(f"✅ Model saved: {model_path}")
        
        # Save scaler
        scaler_path = os.path.join(model_dir, 'lstm_scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"✅ Scaler saved: {scaler_path}")
        
    def save_predictions(self, predictions_df, output_dir='output'):
        """
        Save predictions to CSV
        
        Args:
            predictions_df: DataFrame with predictions
            output_dir (str): Directory to save predictions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'predictions_lstm.csv')
        predictions_df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved: {output_path}")
        
    def save_metrics(self, train_metrics, test_metrics, output_dir='output'):
        """
        Save accuracy metrics to text file
        
        Args:
            train_metrics (dict): Training metrics
            test_metrics (dict): Testing metrics
            output_dir (str): Directory to save metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_path = os.path.join(output_dir, 'lstm_accuracy.txt')
        
        with open(metrics_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("LSTM MODEL - PERFORMANCE METRICS\n")
            f.write("="*60 + "\n\n")
            
            f.write("TRAINING SET METRICS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean Squared Error (MSE):     {train_metrics['mse']:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {train_metrics['rmse']:.4f}\n")
            f.write(f"Mean Absolute Error (MAE):    {train_metrics['mae']:.4f}\n")
            f.write(f"R² Score:                     {train_metrics['r2']:.4f}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {train_metrics['mape']:.2f}%\n")
            f.write(f"Accuracy:                     {train_metrics['accuracy']:.2f}%\n")
            f.write("\n")
            
            f.write("TESTING SET METRICS:\n")
            f.write("-"*60 + "\n")
            f.write(f"Mean Squared Error (MSE):     {test_metrics['mse']:.4f}\n")
            f.write(f"Root Mean Squared Error (RMSE): {test_metrics['rmse']:.4f}\n")
            f.write(f"Mean Absolute Error (MAE):    {test_metrics['mae']:.4f}\n")
            f.write(f"R² Score:                     {test_metrics['r2']:.4f}\n")
            f.write(f"Mean Absolute Percentage Error (MAPE): {test_metrics['mape']:.2f}%\n")
            f.write(f"Accuracy:                     {test_metrics['accuracy']:.2f}%\n")
            f.write("\n")
            f.write("="*60 + "\n")
        
        print(f"✅ Metrics saved: {metrics_path}")
        
    def run_pipeline(self, epochs=100, batch_size=64, model_dir='models', output_dir='output'):
        """
        Run the complete LSTM pipeline
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            model_dir (str): Directory to save model
            output_dir (str): Directory to save outputs
        """
        print("\n" + "="*60)
        print("LSTM PRICE PREDICTION PIPELINE")
        print("="*60 + "\n")
        
        # Step 1: Load and preprocess
        self.load_and_preprocess_data()
        
        # Step 2: Prepare sequences
        self.prepare_sequences()
        
        # Step 3: Build model
        self.build_model()
        
        # Step 4: Train model
        self.train_model(epochs=epochs, batch_size=batch_size)
        
        # Step 5: Make predictions
        train_predict, test_predict, y_train_actual, y_test_actual = self.make_predictions()
        
        # Step 6: Calculate metrics
        print("="*60)
        print("CALCULATING METRICS")
        print("="*60)
        
        train_metrics = self.calculate_metrics(y_train_actual, train_predict)
        test_metrics = self.calculate_metrics(y_test_actual, test_predict)
        
        print("\nTraining Set Metrics:")
        print(f"  RMSE: {train_metrics['rmse']:.4f}")
        print(f"  MAE:  {train_metrics['mae']:.4f}")
        print(f"  R²:   {train_metrics['r2']:.4f}")
        print(f"  Accuracy: {train_metrics['accuracy']:.2f}%")
        
        print("\nTesting Set Metrics:")
        print(f"  RMSE: {test_metrics['rmse']:.4f}")
        print(f"  MAE:  {test_metrics['mae']:.4f}")
        print(f"  R²:   {test_metrics['r2']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.2f}%")
        print("="*60 + "\n")
        
        # Step 7: Generate trading signals
        predictions_df = self.generate_trading_signals(test_predict, y_test_actual)
        
        print("="*60)
        print("TRADING SIGNALS SUMMARY")
        print("="*60)
        print(f"Total predictions: {len(predictions_df)}")
        print(f"Buy signals: {(predictions_df['model_call'] == 'buy').sum()}")
        print(f"Sell signals: {(predictions_df['model_call'] == 'sell').sum()}")
        print(f"Final PnL: {predictions_df['model_pnl'].iloc[-1]:,.2f}")
        print("="*60 + "\n")
        
        # Step 8: Save everything
        print("="*60)
        print("SAVING OUTPUTS")
        print("="*60)
        self.save_model(model_dir=model_dir)
        self.save_predictions(predictions_df, output_dir=output_dir)
        self.save_metrics(train_metrics, test_metrics, output_dir=output_dir)
        print("="*60 + "\n")
        
        print("="*60)
        print("LSTM PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60 + "\n")
        
        return predictions_df, train_metrics, test_metrics


def main():
    """
    Main execution function
    """
    # Configuration
    DATA_PATH = 'nifty_data.csv'  # Update this path
    TIME_STEP = 100
    TRAIN_RATIO = 0.70
    EPOCHS = 10
    BATCH_SIZE = 1000
    MODEL_DIR = 'models'
    OUTPUT_DIR = 'output'
    
    # Initialize and run
    predictor = LSTMPricePredictor(
        data_path=DATA_PATH,
        time_step=TIME_STEP,
        train_ratio=TRAIN_RATIO
    )
    
    predictor.run_pipeline(
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR
    )


if __name__ == "__main__":
    main()