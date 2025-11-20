import os
import GetDummies
from src.Hyperparameter.HyperparameterTuner import HyperparameterTuner
from src.Hyperparameter.StagedTuning import StagedTuning


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
    
    # Generate dummy data for testing
    print("\nGenerating dummy data...")
    data = GetDummies.get_dummy(
        spec={
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Volume": "int"
        },
        n_rows=1000
    )
    print(f"Data shape: {data.shape}")
    
    # Define features and target
    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = "Close"
    
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
    
    # Show quick summary of best models
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        final_runs = [r for r in successful if r.get("tuning_stage") == "final"]
        if final_runs:
            print("\n" + "-"*80)
            print("BEST MODELS (Final Training):")
            print("-"*80)
            sorted_final = sorted(final_runs, key=lambda x: x["best_val_loss"])
            for idx, result in enumerate(sorted_final, 1):
                print(f"{idx}. {result['model_type']}: {result['best_val_loss']:.6f}")


if __name__ == "__main__":
    main()
