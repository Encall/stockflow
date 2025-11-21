import os
import DataLoader  
from Hyperparameter.HyperparameterTuner import HyperparameterTuner
from Hyperparameter.StagedTuning import StagedTuning


def main():
    """Main function to run staged hyperparameter tuning"""
    
    print("="*80)
    print("STARTING STAGED HYPERPARAMETER TUNING")
    print("="*80)
    
    # MLflow settings from environment variables
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "127.0.0.1:8080")
    use_mlflow = os.environ.get("USE_MLFLOW", "true").lower() == "true"
    
    print(f"\nMLflow Settings:")
    print(f"  Tracking URI: {mlflow_uri}")
    print(f"  Use MLflow: {use_mlflow}")
    
    # Load real data from MinIO using DataLoader
    print("\nLoading data from MinIO...")
    data_loader = DataLoader.DataLoader("DIG")
    data = data_loader.get_data()
    
    if data is None:
        print("❌ Failed to load data from DataLoader.")
        return
    
    print(f"✅ Data loaded successfully")
    print(f"   Data shape: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    # Define features and target (using actual column names from gold layer)
    feature_cols = ["open", "high", "low", "volume"]
    target_col = "close"
    
    # Check if required columns exist
    missing_cols = [col for col in feature_cols + [target_col] if col not in data.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        print(f"   Available columns: {list(data.columns)}")
        return
    
    # Initialize tuner
    print(f"\nInitializing hyperparameter tuner...")
    tuner = HyperparameterTuner(mlflow_tracking_uri=mlflow_uri, use_mlflow=use_mlflow)
    
    # Initialize staged tuning orchestrator
    print("Initializing staged tuning orchestrator...")
    staged_tuning = StagedTuning(tuner)
    
    # Models to tune
    models_to_tune = ["LSTM", "GRU", "NBERT", "Transformer"]
    
    print(f"\nModels to tune: {models_to_tune}")
    print("\nStaged tuning strategy:")
    print("  Stage 1: Find best scaler (5 runs)")
    print("  Stage 2: Find best seq_len (5 runs)")
    print("  Stage 3: Find best model params per model (~20-30 runs each)")
    print("  Stage 4: Fine-tune training params (~15 runs)")
    print("  Final: Train all models with best configs (4 runs)")
    print(f"  Estimated total: ~50-60 runs")
    print(f"  (vs 384,000 with full grid search!)")
    
    # Run staged grid search
    print("\n" + "="*80)
    print("Starting staged optimization...")
    print("="*80)
    
    results = staged_tuning.run_staged_search(
        model_types=models_to_tune,
        data=data,
        feature_cols=feature_cols,
        target_col=target_col,
        experiment_name=None  # Will auto-generate name with timestamp
    )
    
    print("\n" + "="*80)
    print("STAGED HYPERPARAMETER TUNING COMPLETED")
    print("="*80)
    print(f"\nResults saved to JSON files in current directory")
    print(f"Total runs completed: {len(results)}")
    print(f"\nNext steps:")
    print(f"  1. Review results: results_final.json")
    print(f"  2. Run analysis: python analyze.py")
    print(f"  3. Check MLflow UI for detailed tracking")


if __name__ == "__main__":
    main()