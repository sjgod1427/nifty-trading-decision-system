# """
# NIFTY Trading Decision System - Main Script
# Predicts whether the next candle's closing price will go up or down

# Author: Sambhav Jain
# """

# import argparse
# import warnings
# warnings.filterwarnings('ignore')

# import config
# from src.data_loader import DataLoader
# from src.feature_engineer import FeatureEngineer
# from src.model_trainer import ModelTrainer
# from src.evaluator import Evaluator


# def parse_arguments():
#     """
#     Parse command line arguments

#     Returns:
#         argparse.Namespace: Parsed arguments
#     """
#     parser = argparse.ArgumentParser(
#         description='NIFTY Price Prediction using Machine Learning'
#     )
#     parser.add_argument(
#         '--model',
#         type=str,
#         choices=['all', 'logistic_regression', 'random_forest', 'lightgbm', 'xgboost'],
#         default='all',
#         help='Model to train (default: all)'
#     )
#     return parser.parse_args()


# def main():
#     """
#     Main execution function
#     Orchestrates the complete ML pipeline
#     """
#     print("\n" + "="*60)
#     print("NIFTY TRADING DECISION SYSTEM")
#     print("="*60 + "\n")

#     # Parse arguments
#     args = parse_arguments()

#     # Step 1: Load and preprocess data
#     print("STEP 1: DATA LOADING & PREPROCESSING")
#     print("-"*60)
#     data_loader = DataLoader(config.DATA_PATH)
#     # Load, preprocess, and create target with noise filtering
#     data_loader.load_data()
#     data_loader.preprocess()
#     df = data_loader.create_target(
#         min_movement_pct=config.MIN_MOVEMENT_PCT,
#         use_multiclass=config.USE_MULTICLASS,
#         up_threshold=config.UP_THRESHOLD,
#         down_threshold=config.DOWN_THRESHOLD
#     )

#     # Step 2: Feature Engineering
#     feature_engineer = FeatureEngineer(
#         df=df,
#         sma_windows=config.SMA_WINDOWS,
#         rsi_period=config.RSI_PERIOD,
#         volatility_window=config.VOLATILITY_WINDOW,
#         lag_periods=config.LAG_PERIODS
#     )
#     df, feature_columns = feature_engineer.create_all_features()
#     X, y = feature_engineer.get_feature_matrix()

#     # Step 3: Train/Test Split (Time-based)
#     print("="*60)
#     print("TRAIN/TEST SPLIT")
#     print("="*60)
#     split_index = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)

#     X_train = X.iloc[:split_index]
#     X_test = X.iloc[split_index:]
#     y_train = y.iloc[:split_index]
#     y_test = y.iloc[split_index:]

#     train_df = df.iloc[:split_index]
#     test_df = df.iloc[split_index:]

#     print(f"Total samples: {len(df)}")
#     print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
#     print(f"Testing set:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
#     print(f"Features: {len(feature_columns)}")
#     print(f"\nTrain period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
#     print(f"Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
#     print("="*60)

#     # Step 4: Model Training
#     print("\n" + "="*60)
#     print("MODEL TRAINING")
#     print("="*60)

#     trainer = ModelTrainer(X_train, X_test, y_train, y_test, config.MODELS_DIR, test_df)

#     # Train selected models
#     if args.model == 'all' or args.model == 'logistic_regression':
#         trainer.train_logistic_regression(config.LR_PARAMS, use_scaling=False)  # Disable scaling for now

#     if args.model == 'all' or args.model == 'random_forest':
#         trainer.train_random_forest(config.RF_PARAMS)

#     if args.model == 'all' or args.model == 'lightgbm':
#         trainer.train_lightgbm(config.LGBM_PARAMS)

#     if args.model == 'all' or args.model == 'xgboost':
#         trainer.train_xgboost(config.XGB_PARAMS)



#     # Step 4.5: Save predictions for all trained models
#     print("\n" + "="*60)
#     print("SAVING MODEL-SPECIFIC PREDICTIONS")
#     print("="*60)
#     for model_name in trainer.models.keys():
#         trainer.save_model_predictions(model_name, config.OUTPUT_DIR)
#     print("="*60)

#     # Step 5: Model Comparison and Selection
#     best_model_name, best_model, comparison_df = trainer.compare_models()

#     # Save best model
#     trainer.save_model(best_model, best_model_name)

#     # Step 6: Feature Importance (if applicable)
#     if config.SAVE_FEATURE_IMPORTANCE:
#         feature_importance = trainer.get_feature_importance(
#             best_model,
#             best_model_name,
#             feature_columns
#         )
#         if feature_importance is not None:
#             print("\nTop 10 Most Important Features:")
#             print(feature_importance.head(10).to_string(index=False))

#     # Step 7: Model Evaluation
#     evaluator = Evaluator(best_model, X_test, y_test, test_df)
#     metrics = evaluator.evaluate_model()

#     # Step 8: Generate Trading Signals
#     print("="*60)
#     print("SIGNAL GENERATION")
#     print("="*60)
#     evaluator.generate_signals()

#     # Step 9: Calculate PnL
#     evaluator.calculate_pnl()

#     # Step 10: Create Final Output
#     final_df = evaluator.create_final_output(config.FINAL_PREDICTIONS_PATH)

#     # Step 11: Save Metrics Report
#     evaluator.save_metrics_report(
#         config.METRICS_REPORT_PATH,
#         best_model_name,
#         comparison_df
#     )

#     # Final Summary
#     print("="*60)
#     print("EXECUTION COMPLETE")
#     print("="*60)
#     print(f"\nOutputs saved:")
#     print(f"1. Predictions: {config.FINAL_PREDICTIONS_PATH}")
#     print(f"2. Metrics:     {config.METRICS_REPORT_PATH}")
#     print(f"3. Best Model:  {config.MODELS_DIR}/{best_model_name}.pkl")
#     print(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
#     print(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
#     print("\n" + "="*60)
#     print("Thank you for using NIFTY Trading Decision System!")
#     print("="*60 + "\n")


# if __name__ == "__main__":
#     main()


"""
NIFTY Trading Decision System - Main Script
Predicts whether the next candle's closing price will go up or down

Author: Sambhav Jain
"""

import argparse
import warnings
warnings.filterwarnings('ignore')

import config
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.evaluator import Evaluator


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description='NIFTY Price Prediction using Machine Learning'
    )
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'logistic_regression', 'random_forest', 'lightgbm', 'xgboost'],
        default='all',
        help='Model to train (default: all)'
    )
    return parser.parse_args()


def main():
    """
    Main execution function
    Orchestrates the complete ML pipeline
    """
    print("\n" + "="*60)
    print("NIFTY TRADING DECISION SYSTEM")
    print("="*60 + "\n")

    # Parse arguments
    args = parse_arguments()

    # Step 1: Load and preprocess data
    print("STEP 1: DATA LOADING & PREPROCESSING")
    print("-"*60)
    data_loader = DataLoader(config.DATA_PATH)
    # Load, preprocess, and create target with noise filtering
    data_loader.load_data()
    data_loader.preprocess()
    df = data_loader.create_target(
        min_movement_pct=config.MIN_MOVEMENT_PCT,
        use_multiclass=config.USE_MULTICLASS,
        up_threshold=config.UP_THRESHOLD,
        down_threshold=config.DOWN_THRESHOLD
    )

    # Step 2: Feature Engineering
    feature_engineer = FeatureEngineer(
        df=df,
        sma_windows=config.SMA_WINDOWS,
        rsi_period=config.RSI_PERIOD,
        volatility_window=config.VOLATILITY_WINDOW,
        lag_periods=config.LAG_PERIODS
    )
    df, feature_columns = feature_engineer.create_all_features()
    X, y = feature_engineer.get_feature_matrix()

    # Step 3: Train/Test Split (Time-based)
    print("="*60)
    print("TRAIN/TEST SPLIT")
    print("="*60)
    split_index = int(len(df) * config.TRAIN_TEST_SPLIT_RATIO)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Total samples: {len(df)}")
    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Testing set:  {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    print(f"Features: {len(feature_columns)}")
    print(f"\nTrain period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
    print(f"Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")
    print("="*60)

    # Step 4: Model Training
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60)

    trainer = ModelTrainer(X_train, X_test, y_train, y_test, config.MODELS_DIR, test_df)

    # Train selected models
    if args.model == 'all' or args.model == 'logistic_regression':
        trainer.train_logistic_regression(config.LR_PARAMS, use_scaling=False)  # Disable scaling for now

    if args.model == 'all' or args.model == 'random_forest':
        trainer.train_random_forest(config.RF_PARAMS)

    if args.model == 'all' or args.model == 'lightgbm':
        trainer.train_lightgbm(config.LGBM_PARAMS)

    if args.model == 'all' or args.model == 'xgboost':
        trainer.train_xgboost(config.XGB_PARAMS)



    # Step 4.5: Save predictions for all trained models
    print("\n" + "="*60)
    print("SAVING ALL MODEL PREDICTIONS WITH TEST DATA")
    print("="*60)
    for model_name in trainer.models.keys():
        trainer.save_model_predictions(model_name, config.OUTPUT_DIR)
    print("="*60)

    # Step 5: Model Comparison and Selection
    best_model_name, best_model, comparison_df = trainer.compare_models()

    # Save best model
    trainer.save_model(best_model, best_model_name)

    # Step 6: Feature Importance (if applicable)
    if config.SAVE_FEATURE_IMPORTANCE:
        feature_importance = trainer.get_feature_importance(
            best_model,
            best_model_name,
            feature_columns
        )
        if feature_importance is not None:
            print("\nTop 10 Most Important Features:")
            print(feature_importance.head(10).to_string(index=False))

    # Step 7: Model Evaluation
    evaluator = Evaluator(best_model, X_test, y_test, test_df)
    metrics = evaluator.evaluate_model()

    # Step 8: Generate Trading Signals
    print("="*60)
    print("SIGNAL GENERATION")
    print("="*60)
    evaluator.generate_signals()

    # Step 9: Calculate PnL
    evaluator.calculate_pnl()

    # Step 11: Save Metrics Report
    evaluator.save_metrics_report(
        config.METRICS_REPORT_PATH,
        best_model_name,
        comparison_df
    )

    # Final Summary
    print("="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"\nOutputs saved:")
    print(f"1. Model Predictions: {config.OUTPUT_DIR}/<model_name>_predictions.csv (for each model)")
    print(f"2. Metrics:     {config.METRICS_REPORT_PATH}")
    print(f"3. Best Model:  {config.MODELS_DIR}/{best_model_name}.pkl")
    print(f"\nBest Model: {best_model_name.replace('_', ' ').title()}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print("\n" + "="*60)
    print("Thank you for using NIFTY Trading Decision System!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()