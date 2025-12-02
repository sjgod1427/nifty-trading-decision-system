"""
Evaluator Module
Generates trading signals and calculates PnL based on model predictions
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

class Evaluator:
    """
    Evaluates model performance and generates trading signals with PnL
    """

    def __init__(self, model, X_test, y_test, test_df):
        """
        Initialize Evaluator

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            test_df: Original test dataframe with OHLC data
        """
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.test_df = test_df.copy()
        self.predictions = None
        self.metrics = {}

    def evaluate_model(self):
        """
        Evaluate model performance with detailed metrics

        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        print("\n" + "="*60)
        print("MODEL EVALUATION ON TEST SET")
        print("="*60)

        # Generate predictions
        self.predictions = self.model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)
        precision = precision_score(self.y_test, self.predictions, zero_division=0)
        recall = recall_score(self.y_test, self.predictions, zero_division=0)
        f1 = f1_score(self.y_test, self.predictions, zero_division=0)

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        # Print metrics
        print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
        print(f"F1-Score:  {f1:.4f}")

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.predictions)
        print("\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 0      1")
        print(f"Actual  0     {cm[0][0]:5d}  {cm[0][1]:5d}")
        print(f"        1     {cm[1][0]:5d}  {cm[1][1]:5d}")

        # Classification Report
        print("\nDetailed Classification Report:")
        print(classification_report(self.y_test, self.predictions,
                                   target_names=['Down (0)', 'Up (1)']))
        print("="*60 + "\n")

        return self.metrics

    def generate_signals(self):
        """
        Generate trading signals based on model predictions
        - prediction = 1 (price up) → "buy"
        - prediction = 0 (price down) → "sell"

        Returns:
            pd.DataFrame: Test dataframe with signals
        """
        print("Generating trading signals...")

        # Add predictions to test dataframe
        self.test_df['predicted'] = self.predictions

        # Generate model_call column
        self.test_df['model_call'] = self.test_df['predicted'].apply(
            lambda x: 'buy' if x == 1 else 'sell'
        )

        signal_counts = self.test_df['model_call'].value_counts()
        print(f"Buy signals:  {signal_counts.get('buy', 0)}")
        print(f"Sell signals: {signal_counts.get('sell', 0)}")

        return self.test_df

    def calculate_pnl(self):
        """
        Calculate cumulative PnL based on model signals
        Logic:
        - If model_call = 'buy': Subtract current close price (we pay to buy)
        - If model_call = 'sell': Add current close price (we receive from selling)
        - Iterate row by row in timestamp ascending order
        - Keep updating cumulative running PnL

        Returns:
            pd.DataFrame: Test dataframe with PnL column
        """
        print("\nCalculating cumulative PnL...")

        # Ensure dataframe is sorted by timestamp
        self.test_df = self.test_df.sort_values('timestamp').reset_index(drop=True)

        # Initialize PnL column
        model_pnl_values = []
        cumulative_pnl = 0

        # Iterate through each row to calculate cumulative PnL
        for idx, row in self.test_df.iterrows():
            if row['model_call'] == 'buy':
                cumulative_pnl -= row['close']  # Pay to buy
            else:  # sell
                cumulative_pnl += row['close']  # Receive from selling

            model_pnl_values.append(cumulative_pnl)

        # Add PnL column
        self.test_df['model_pnl'] = model_pnl_values

        # Calculate some PnL statistics
        final_pnl = model_pnl_values[-1]
        max_pnl = max(model_pnl_values)
        min_pnl = min(model_pnl_values)

        print(f"Final PnL:   {final_pnl:,.2f}")
        print(f"Max PnL:     {max_pnl:,.2f}")
        print(f"Min PnL:     {min_pnl:,.2f}")

        return self.test_df

    def create_final_output(self, output_path):
        """
        Create final output CSV with required columns:
        - Timestamp
        - Close
        - Predicted
        - model_call
        - model_pnl

        Args:
            output_path (str): Path to save the output CSV

        Returns:
            pd.DataFrame: Final output dataframe
        """
        print("\n" + "="*60)
        print("CREATING FINAL OUTPUT")
        print("="*60)

        # Select required columns
        output_columns = ['timestamp', 'close', 'predicted', 'model_call', 'model_pnl']
        final_df = self.test_df[output_columns].copy()

        # Rename timestamp column to match expected format (capitalize)
        final_df.columns = ['Timestamp', 'Close', 'Predicted', 'model_call', 'model_pnl']

        # Save to CSV
        final_df.to_csv(output_path, index=False)

        print(f"Final predictions saved to: {output_path}")
        print(f"Total rows: {len(final_df)}")
        print(f"\nFirst few rows of output:")
        print(final_df.head(10).to_string(index=False))
        print(f"\nLast few rows of output:")
        print(final_df.tail(5).to_string(index=False))
        print("="*60 + "\n")

        return final_df

    def save_metrics_report(self, report_path, model_name, comparison_df):
        """
        Save detailed metrics report to a text file

        Args:
            report_path (str): Path to save the report
            model_name (str): Name of the best model
            comparison_df (pd.DataFrame): Model comparison dataframe
        """
        with open(report_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("NIFTY PRICE PREDICTION - MODEL EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")

            # Model Comparison
            f.write("MODEL COMPARISON\n")
            f.write("-"*60 + "\n")
            f.write(comparison_df.to_string(index=False))
            f.write("\n\n")

            # Best Model
            f.write(f"BEST MODEL: {model_name.replace('_', ' ').title()}\n")
            f.write("-"*60 + "\n")
            f.write(f"Accuracy:  {self.metrics['accuracy']:.4f} ({self.metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision: {self.metrics['precision']:.4f} ({self.metrics['precision']*100:.2f}%)\n")
            f.write(f"Recall:    {self.metrics['recall']:.4f} ({self.metrics['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:  {self.metrics['f1']:.4f}\n")
            f.write("\n")

            # Confusion Matrix
            cm = confusion_matrix(self.y_test, self.predictions)
            f.write("Confusion Matrix:\n")
            f.write(f"                 Predicted\n")
            f.write(f"                 0      1\n")
            f.write(f"Actual  0     {cm[0][0]:5d}  {cm[0][1]:5d}\n")
            f.write(f"        1     {cm[1][0]:5d}  {cm[1][1]:5d}\n")
            f.write("\n")

            # Classification Report
            f.write("Detailed Classification Report:\n")
            f.write(classification_report(self.y_test, self.predictions,
                                         target_names=['Down (0)', 'Up (1)']))
            f.write("\n")

            # PnL Summary
            f.write("PnL SUMMARY\n")
            f.write("-"*60 + "\n")
            final_pnl = self.test_df['model_pnl'].iloc[-1]
            max_pnl = self.test_df['model_pnl'].max()
            min_pnl = self.test_df['model_pnl'].min()
            f.write(f"Final PnL:   {final_pnl:,.2f}\n")
            f.write(f"Max PnL:     {max_pnl:,.2f}\n")
            f.write(f"Min PnL:     {min_pnl:,.2f}\n")
            f.write("\n")

            f.write("="*60 + "\n")

        print(f"Metrics report saved to: {report_path}")
