"""
Data Loader Module
Handles loading, cleaning, and preparing the NIFTY dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataLoader:
    """
    Loads and preprocesses NIFTY intraday data
    """

    def __init__(self, data_path):
        """
        Initialize DataLoader with path to CSV file

        Args:
            data_path (str): Path to the CSV file
        """
        self.data_path = data_path
        self.df = None

    def load_data(self):
        """
        Load CSV file and perform basic preprocessing

        Returns:
            pd.DataFrame: Loaded and preprocessed dataframe
        """
        print("Loading data from CSV...")
        self.df = pd.read_csv(self.data_path)
        print(f"Data loaded: {len(self.df)} rows, {len(self.df.columns)} columns")

        return self.df

    def preprocess(self):
        """
        Preprocess the data:
        - Convert timestamp to datetime
        - Sort by timestamp ascending (chronological order)
        - Remove duplicates
        - Handle missing values

        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        print("Preprocessing data...")

        # Convert timestamp to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Sort by timestamp in ascending order (oldest first)
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        print(f"Data sorted chronologically: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")

        # Remove duplicates based on timestamp
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['timestamp'], keep='first')
        removed_duplicates = initial_count - len(self.df)
        if removed_duplicates > 0:
            print(f"Removed {removed_duplicates} duplicate rows")

        # Check for missing values in key columns
        key_columns = ['open', 'high', 'low', 'close']
        missing_counts = self.df[key_columns].isnull().sum()
        if missing_counts.sum() > 0:
            print(f"Missing values found:\n{missing_counts}")
            print("Dropping rows with missing values in OHLC columns...")
            self.df = self.df.dropna(subset=key_columns).reset_index(drop=True)

        print(f"Preprocessing complete: {len(self.df)} rows remaining")
        return self.df

    def create_target(self, min_movement_pct=0.08, use_multiclass=False,
                      up_threshold=0.1, down_threshold=-0.1):
        """
        Create target column with noise filtering:
        - Option 1 (Binary): Filter noise, then predict up/down on significant movements
        - Option 2 (Multi-class): Strong Up / Neutral / Strong Down

        Args:
            min_movement_pct (float): Minimum movement to consider (filters noise)
            use_multiclass (bool): Use 3-class target instead of binary
            up_threshold (float): Threshold for "Strong Up" class
            down_threshold (float): Threshold for "Strong Down" class

        Returns:
            pd.DataFrame: Dataframe with target column
        """
        print("Creating target variable...")

        # Shift close price to get next candle's close
        self.df['next_close'] = self.df['close'].shift(-1)

        # Calculate percentage change
        self.df['pct_change'] = ((self.df['next_close'] - self.df['close']) / self.df['close']) * 100

        if use_multiclass:
            print(f"Using MULTI-CLASS target (Up: >{up_threshold}%, Down: <{down_threshold}%)")

            # Create 3-class target
            self.df['target'] = 1  # Default: neutral
            self.df.loc[self.df['pct_change'] > up_threshold, 'target'] = 2  # Strong Up
            self.df.loc[self.df['pct_change'] < down_threshold, 'target'] = 0  # Strong Down

            # Remove neutral class to focus on clear signals
            initial_count = len(self.df)
            self.df = self.df[self.df['target'] != 1].reset_index(drop=True)
            removed_neutral = initial_count - len(self.df)
            print(f"Removed {removed_neutral} neutral movements ({removed_neutral/initial_count*100:.2f}%)")

        else:
            print(f"Using BINARY target with noise filtering (min movement: {min_movement_pct}%)")

            # Filter out noise - keep only significant movements
            initial_count = len(self.df)
            self.df = self.df[self.df['pct_change'].abs() >= min_movement_pct].reset_index(drop=True)
            removed_noise = initial_count - len(self.df)
            print(f"Filtered {removed_noise} noisy movements ({removed_noise/initial_count*100:.2f}%)")

            # Create binary target on meaningful movements
            self.df['target'] = (self.df['pct_change'] > 0).astype(int)

        # Drop rows with NaN
        self.df = self.df.dropna(subset=['next_close']).reset_index(drop=True)

        # Check class distribution
        class_distribution = self.df['target'].value_counts().sort_index()
        print(f"\nTarget variable created. Class distribution:")
        if use_multiclass:
            print(f"  0 (Strong Down): {class_distribution.get(0, 0)} ({class_distribution.get(0, 0)/len(self.df)*100:.2f}%)")
            print(f"  2 (Strong Up):   {class_distribution.get(2, 0)} ({class_distribution.get(2, 0)/len(self.df)*100:.2f}%)")
        else:
            print(f"  0 (Price Down): {class_distribution[0]} ({class_distribution[0]/len(self.df)*100:.2f}%)")
            print(f"  1 (Price Up):   {class_distribution[1]} ({class_distribution[1]/len(self.df)*100:.2f}%)")
        print(f"Final dataset: {len(self.df)} rows")

        return self.df

    def get_processed_data(self):
        """
        Execute complete data loading pipeline

        Returns:
            pd.DataFrame: Fully processed dataframe ready for feature engineering
        """
        self.load_data()
        self.preprocess()
        self.create_target()

        return self.df
