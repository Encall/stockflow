"""
Main script to run hyperparameter tuning for stock prediction models.
"""
import argparse
from dotenv import load_dotenv
import pandas as pd
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Hyperparameter.Tuner import ModelTuner
import DataLoader

load_dotenv()
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Ensure MLFLOW_S3_ENDPOINT_URL is set if AWS_S3_ENDPOINT_URL is present
# This is required for MLflow to work with MinIO/S3-compatible storage
aws_endpoint = os.getenv("AWS_S3_ENDPOINT_URL")
if aws_endpoint and not os.getenv("MLFLOW_S3_ENDPOINT_URL"):
    if not aws_endpoint.startswith(("http://", "https://")):
        aws_endpoint = f"https://{aws_endpoint}"
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = aws_endpoint

def main():
    parser = argparse.ArgumentParser(description="Run hyperparameter tuning for stock prediction models")
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="stock_hyperparameter_tuning",
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10,
        help="Number of hyperparameter trials to run"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training"
    )
    parser.add_argument(
        "--model-types",
        type=str,
        nargs="+",
        default=None,
        help="Model types to tune (LSTM, GRU, NBERT, Transformer). If not specified, all models will be tried."
    )
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow tracking"
    )
    parser.add_argument(
        "--mlflow-uri",
        type=str,
        default=MLFLOW_TRACKING_URI,
        help="MLflow tracking URI"
    )
    parser.add_argument(
        "--feature-cols",
        type=str,
        nargs="+",
        default=["open", "high", "low", "volume"],
        help="Feature columns to use"
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default="close",
        help="Target column to predict"
    )
    parser.add_argument(
        "--exhaustive",
        action="store_true",
        help="Try ALL possible parameter combinations (ignores n-trials)"
    )
    
    args = parser.parse_args()

    # If MLFLOW_TRACKING_URI is set, force use_mlflow to True
    if args.mlflow_uri:
        args.use_mlflow = True
    
    print("=" * 60)
    print("Multi-Stock Hyperparameter Tuning")
    print("=" * 60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Mode: {'EXHAUSTIVE (all combinations)' if args.exhaustive else f'RANDOM ({args.n_trials} trials)'}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model types: {args.model_types or 'All (LSTM, GRU, NBERT, Transformer)'}")
    print(f"MLflow enabled: {args.use_mlflow}")
    if args.use_mlflow:
        print(f"MLflow URI: {args.mlflow_uri}")
    print(f"Feature columns: {args.feature_cols}")
    print(f"Target column: {args.target_col}")
    print("=" * 60)
    
    # Load data for all stocks
    all_stocks = ["DIG", "DJP", "EDC", "ERC", "EUM", 
                  "FEZ", "FUND", "GLD", "IAE", "MIDU", 
                  "PBP", "PID", "TSM", "TV", "UGA", 
                  "VOO", "VSS", "XBI", "XLI", "XLP"]
    all_data = dict()
    
    print(f"\n📥 Loading data for {len(all_stocks)} stocks...")
    for stock in all_stocks:
        data_loader = DataLoader.DataLoader(stock)
        data = data_loader.get_data()
        if data is None:
            print(f"❌ Failed to load data for stock {stock}")
            sys.exit(1)
        all_data[stock] = data
        print(f"✅ Loaded {stock}: {len(data)} rows")
    
    print(f"\n✅ Successfully loaded all {len(all_stocks)} stocks")
    
    # Initialize tuner with experiment name
    print(f"\n🔧 Initializing tuner with experiment: {args.experiment_name}")
    tuner = ModelTuner(
        data=all_data,
        feature_cols=args.feature_cols,
        target_col=[args.target_col] if isinstance(args.target_col, str) else args.target_col,
        experiment_name=args.experiment_name
    )
    
    # Run tuning
    print(f"\n🚀 Starting hyperparameter tuning...\n")
    best_config = tuner.tune(
        model_types=args.model_types,
        n_trials=args.n_trials,
        batch_size=args.batch_size,
        use_mlflow=args.use_mlflow,
        mlflow_tracking_uri=args.mlflow_uri if args.use_mlflow else None,
        exhaustive=args.exhaustive
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("🏆 Best Configuration Found")
    print("=" * 60)
    print(f"Model: {best_config['model_type']}")
    print(f"Validation Loss: {best_config['val_loss']:.6f}")
    print(f"\nModel Parameters:")
    for k, v in best_config['model_params'].items():
        print(f"  {k}: {v}")
    print(f"\nDataset Parameters:")
    print(f"  seq_len: {best_config['seq_len']}")
    print(f"  scaler: {best_config['scaler']}")
    print(f"\nTraining Parameters:")
    print(f"  lr: {best_config['lr']}")
    print(f"  batch_size: {best_config['batch_size']}")
    print("=" * 60)
    
    # Register best model as production if MLflow is enabled
    if args.use_mlflow and 'run_id' in best_config:
        try:
            import mlflow
            from mlflow.tracking import MlflowClient
            import os
            
            # Set shorter timeout for model registry operations
            os.environ['MLFLOW_HTTP_REQUEST_TIMEOUT'] = '30'
            
            client = MlflowClient()
            model_name = f"{args.experiment_name}_best_model"
            run_id = best_config['run_id']
            best_model_type = best_config['model_type']
            
            print(f"\n📦 Registering best model to MLflow Model Registry...")
            print(f"   Model name: {model_name}")
            print(f"   Run ID: {run_id}")
            print(f"   Model Type: {best_model_type}")
            
            # Verify the run has model artifacts before registering
            run = client.get_run(run_id)
            model_uri = f"runs:/{run_id}/model"
            
            # List artifacts to verify model exists
            artifacts = client.list_artifacts(run_id, "model")
            if not artifacts:
                print(f"   ⚠️  Warning: No model artifacts found for run {run_id}")
                print(f"   Skipping model registration")
            else:
                print(f"   ✓ Model artifacts found: {len(artifacts)} files")
                
                # Check if there's already a production model with different model_type
                try:
                    # Use aliases instead of deprecated stages
                    all_versions = client.search_model_versions(f"name='{model_name}'")
                    production_versions = [v for v in all_versions if "production" in [alias.lower() for alias in v.aliases]]
                    should_promote_to_production = True
                    
                    if production_versions:
                        # Get the current production model's run to check its model_type tag
                        current_prod_version = production_versions[0]
                        current_prod_run_id = current_prod_version.run_id
                        current_prod_run = client.get_run(current_prod_run_id)
                        current_model_type = current_prod_run.data.tags.get("model_type", "")
                        
                        print(f"\n   Current Production Model Type: {current_model_type}")
                        
                        if current_model_type and current_model_type != best_model_type:
                            should_promote_to_production = False
                            print(f"   ⚠️  Model type changed: {current_model_type} → {best_model_type}")
                            print(f"   Will register with 'champion' alias instead of 'production'")
                        else:
                            print(f"   ✓ Same model type, will promote to production")
                except Exception as e:
                    # No production version exists yet, safe to promote
                    print(f"   No existing production version found")
                    should_promote_to_production = True
                
                # Register the model with timeout handling
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError("Model registration timed out")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(30)
                
                try:
                    # Register without waiting for creation to complete
                    registered_model = mlflow.register_model(
                        model_uri, 
                        model_name,
                        await_registration_for=10  # Only wait 10 seconds
                    )
                    signal.alarm(0)
                    
                    # Use aliases instead of stages
                    version = registered_model.version
                except TimeoutError:
                    signal.alarm(0)  # Cancel the alarm
                    print(f"   ⚠️  Model registration timed out")
                    print(f"   The model may still be registering in the background")
                    # Don't raise, just skip alias assignment
                    version = None
                
                # Only set aliases if registration completed
                if version is not None:
                    if should_promote_to_production:
                        # Set as production
                        client.set_registered_model_alias(model_name, "production", version)
                        client.set_registered_model_alias(model_name, "champion", version)
                        target_alias = "production"
                    else:
                        # Set as champion (candidate for production)
                        client.set_registered_model_alias(model_name, "champion", version)
                        target_alias = "champion"
                    
                    # Add description
                    client.update_model_version(
                        name=model_name,
                        version=version,
                        description=f"Best model from hyperparameter tuning. "
                                   f"Model: {best_config['model_type']}, "
                                   f"Val Loss: {best_config['val_loss']:.6f}, "
                                   f"Seq Len: {best_config['seq_len']}, "
                                   f"LR: {best_config['lr']}, "
                                   f"Scaler: {best_config['scaler']}"
                    )
                    
                    print(f"✅ Model registered successfully!")
                    print(f"   Name: {model_name}")
                    print(f"   Version: {version}")
                    print(f"   Alias: {target_alias}")
                    if target_alias == "production":
                        print(f"   Load with: mlflow.pyfunc.load_model('models:/{model_name}@production')")
                    else:
                        print(f"   Load with: mlflow.pyfunc.load_model('models:/{model_name}@champion')")
                        print(f"   💡 Review and manually set 'production' alias if desired")
                else:
                    print(f"   ⚠️  Skipping alias assignment due to timeout")
            
        except Exception as e:
            print(f"❌ Failed to register model: {e}")
            import traceback
            traceback.print_exc()
    
    if args.use_mlflow:
        print(f"\n📊 MLflow tracking: {args.mlflow_uri}")
        print(f"   Experiment: {args.experiment_name}")
    
    print("\n✅ Tuning complete!")
    
    # Ensure all MLflow runs are ended and clean up connections
    if args.use_mlflow:
        import mlflow
        try:
            # End any active run
            while mlflow.active_run():
                mlflow.end_run()
        except:
            pass
    
    # Force exit to prevent hanging
    sys.exit(0)


if __name__ == "__main__":
    main()
