import torch
import torch.nn as nn
import os
import json
import traceback
import pickle
from datetime import datetime
from itertools import product
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available - running without experiment tracking")

import model.LSTM as LSTMModel
import model.Transformer as TransformerModel
import model.GRU as GRUModel
import model.NBERT as NBERTModel
import StockDataset
import GetDummies
from src.Hyperparameter.HyperparameterConfig import (
    SCALER_OPTIONS,
    DATASET_PARAMS,
    MODEL_PARAMS,
    TRAINING_PARAMS
)


class HyperparameterTuner:
    def __init__(self, mlflow_tracking_uri: str = "http://127.0.0.1:8080", use_mlflow: bool = False):
        """Initialize hyperparameter tuner with optional MLflow tracking"""
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow:
            # Fix tracking URI format
            if not mlflow_tracking_uri.startswith(('http://', 'https://')):
                mlflow_tracking_uri = f"http://{mlflow_tracking_uri}"
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"MLflow tracking enabled: {mlflow_tracking_uri}")
        else:
            print("Running without MLflow tracking")
        self.results = []
        
    def get_loss_function(self, loss_name: str):
        """Get loss function by name"""
        loss_functions = {
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "Huber": nn.HuberLoss()
        }
        return loss_functions.get(loss_name, nn.MSELoss())
    
    def create_model(self, model_type: str, model_params: Dict[str, Any], input_size: int, seq_len: int = None):
        """Create model instance based on type and parameters"""
        try:
            if model_type == "LSTM":
                return LSTMModel.LSTM(
                    input_size=input_size,
                    **model_params
                )
            elif model_type == "GRU":
                return GRUModel.GRU(
                    input_size=input_size,
                    **model_params
                )
            elif model_type == "NBERT":
                return NBERTModel.NBERT(
                    input_size=input_size,
                    seq_len=seq_len,
                    **model_params
                )
            elif model_type == "Transformer":
                return TransformerModel.Transformer(
                    input_size=input_size,
                    **model_params
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        except Exception as e:
            print(f"Error creating model {model_type}: {str(e)}")
            raise
    
    def train_single_epoch(self, model, dataLoader, optimizer, loss_fn, device):
        """Train model for one epoch"""
        model.train()
        total_loss = 0
        num_batches = 0
        
        try:
            for batch_idx, (data, target) in enumerate(dataLoader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = loss_fn(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            return avg_loss
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return float('inf')
    
    def evaluate_model(self, model, dataLoader, loss_fn, device):
        """Evaluate model on validation/test data"""
        model.eval()
        total_loss = 0
        num_batches = 0
        
        try:
            with torch.no_grad():
                for data, target in dataLoader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = loss_fn(output, target)
                    total_loss += loss.item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
            return avg_loss
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return float('inf')
    
    def train_with_config(
        self,
        model_type: str,
        model_params: Dict[str, Any],
        dataset_params: Dict[str, Any],
        training_params: Dict[str, Any],
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        device: torch.device
    ) -> Dict[str, Any]:
        """Train model with specific configuration"""
        
        run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = None
        
        try:
            # Start MLflow run if available
            if self.use_mlflow:
                mlflow_run = mlflow.start_run(run_name=run_name)
                run_id = mlflow_run.info.run_id
                # Log all parameters
                mlflow.log_param("model_type", model_type)
                mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
                mlflow.log_params({f"dataset_{k}": v for k, v in dataset_params.items()})
                mlflow.log_params({f"training_{k}": v for k, v in training_params.items()})
            
            # Prepare dataset
            scaler = SCALER_OPTIONS[dataset_params["scaler"]]
            seq_len = dataset_params["seq_len"]
            
            stock_data = StockDataset.MultiFeaturePriceDataset(
                data=data,
                feature_cols=feature_cols,
                target_col=target_col,
                seq_len=seq_len,
                scaler=scaler
            )
            
            # Split data (80% train, 20% validation)
            train_size = int(0.8 * len(stock_data))
            val_size = len(stock_data) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                stock_data, [train_size, val_size]
            )
            
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=training_params["batch_size"],
                shuffle=True
            )
            
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=training_params["batch_size"],
                shuffle=False
            )
            
            # Create model
            model = self.create_model(
                model_type=model_type,
                model_params=model_params,
                input_size=len(feature_cols),
                seq_len=seq_len
            )
            model = model.to(device)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])
            loss_fn = self.get_loss_function(training_params["loss_fn"])
            
            # Training loop
            best_val_loss = float('inf')
            best_epoch = 0
            patience = 10
            patience_counter = 0
            
            for epoch in range(training_params["epochs"]):
                train_loss = self.train_single_epoch(model, train_loader, optimizer, loss_fn, device)
                val_loss = self.evaluate_model(model, val_loader, loss_fn, device)
                
                # Log metrics
                if self.use_mlflow:
                    mlflow.log_metric("train_loss", train_loss, step=epoch)
                    mlflow.log_metric("val_loss", val_loss, step=epoch)
                
                print(f"Epoch {epoch+1}/{training_params['epochs']} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Log best metrics and artifacts
            if self.use_mlflow:
                mlflow.log_metric("best_val_loss", best_val_loss)
                mlflow.log_metric("best_epoch", best_epoch)
                
                # Save and log model artifacts
                model_dir = "temp_models"
                os.makedirs(model_dir, exist_ok=True)
                
                # 1. Log PyTorch model
                mlflow.pytorch.log_model(model, "model")
                
                # 2. Save and log model state dict (.pth)
                model_pth_path = f"{model_dir}/{run_name}_best_model.pth"
                torch.save(model.state_dict(), model_pth_path)
                mlflow.log_artifact(model_pth_path, "model_checkpoints")
                
                # 3. Save and log complete model as pickle (.pkl)
                model_pkl_path = f"{model_dir}/{run_name}_model.pkl"
                with open(model_pkl_path, 'wb') as f:
                    pickle.dump(model, f)
                mlflow.log_artifact(model_pkl_path, "model_checkpoints")
                
                # Log configuration as artifact
                config_dict = {
                    "model_params": model_params,
                    "dataset_params": dataset_params,
                    "training_params": training_params,
                    "best_val_loss": float(best_val_loss),
                    "best_epoch": best_epoch
                }
                config_path = f"{model_dir}/{run_name}_config.json"
                with open(config_path, 'w') as f:
                    json.dump(config_dict, f, indent=2)
                mlflow.log_artifact(config_path, "configs")
                
                # Log training summary
                summary_dict = {
                    "model_type": model_type,
                    "final_train_loss": float(train_loss),
                    "final_val_loss": float(val_loss),
                    "best_val_loss": float(best_val_loss),
                    "best_epoch": best_epoch,
                    "total_epochs": epoch + 1
                }
                summary_path = f"{model_dir}/{run_name}_summary.json"
                with open(summary_path, 'w') as f:
                    json.dump(summary_dict, f, indent=2)
                mlflow.log_artifact(summary_path, "summaries")
                
                # Clean up temp files
                for temp_file in [model_pth_path, model_pkl_path, config_path, summary_path]:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                
                # Set tags for easy filtering
                mlflow.set_tag("status", "success")
                mlflow.set_tag("model_type", model_type)
                mlflow.set_tag("best_val_loss", f"{best_val_loss:.6f}")
                mlflow.set_tag("scaler", dataset_params["scaler"])
                mlflow.set_tag("loss_function", training_params["loss_fn"])
                
                mlflow.end_run()
            
            # Create result summary
            result = {
                "run_id": run_id,
                "run_name": run_name,
                "model_type": model_type,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model_params": model_params,
                "dataset_params": dataset_params,
                "training_params": training_params,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
                
        except Exception as e:
            error_msg = f"Error training model: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            
            if self.use_mlflow:
                try:
                    mlflow.log_param("error", str(e))
                    mlflow.set_tag("status", "failed")
                    mlflow.end_run()
                except:
                    pass
            
            return {
                "run_id": run_id,
                "run_name": run_name,
                "model_type": model_type,
                "best_val_loss": float('inf'),
                "best_epoch": -1,
                "model_params": model_params,
                "dataset_params": dataset_params,
                "training_params": training_params,
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def grid_search(
        self,
        model_types: List[str],
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        max_combinations: int = 100,
        random_sample: bool = True,
        experiment_name: str = None
    ):
        """Perform grid search over hyperparameters"""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create or use existing experiment
        if experiment_name is None:
            experiment_name = f"hyperparameter_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.use_mlflow:
            try:
                experiment = mlflow.set_experiment(experiment_name)
                print(f"\nMLflow Experiment: {experiment_name}")
                print(f"Experiment ID: {experiment.experiment_id}")
                print(f"Artifact Location: {experiment.artifact_location}")
            except Exception as e:
                print(f"Warning: Could not set MLflow experiment: {str(e)}")
                print("Continuing without MLflow tracking...")
                self.use_mlflow = False
        
        all_results = []
        
        for model_type in model_types:
            print(f"\n{'='*80}")
            print(f"Tuning hyperparameters for {model_type}")
            print(f"{'='*80}\n")
            
            # Get parameter grids
            model_param_grid = MODEL_PARAMS[model_type]
            dataset_param_grid = DATASET_PARAMS
            training_param_grid = TRAINING_PARAMS
            
            # Generate all combinations
            model_combinations = list(product(*[[(k, v) for v in values] for k, values in model_param_grid.items()]))
            model_combinations = [{k: v for k, v in combo} for combo in model_combinations]
            
            dataset_combinations = list(product(*[[(k, v) for v in values] for k, values in dataset_param_grid.items()]))
            dataset_combinations = [{k: v for k, v in combo} for combo in dataset_combinations]
            
            training_combinations = list(product(*[[(k, v) for v in values] for k, values in training_param_grid.items()]))
            training_combinations = [{k: v for k, v in combo} for combo in training_combinations]
            
            # Create all possible configurations
            all_configs = []
            for m_params in model_combinations:
                for d_params in dataset_combinations:
                    for t_params in training_combinations:
                        all_configs.append((m_params, d_params, t_params))
            
            print(f"Total possible combinations for {model_type}: {len(all_configs)}")
            
            # Sample configurations if too many
            if random_sample and len(all_configs) > max_combinations:
                print(f"Randomly sampling {max_combinations} combinations")
                import random
                all_configs = random.sample(all_configs, max_combinations)
            else:
                all_configs = all_configs[:max_combinations]
            
            # Train each configuration
            for idx, (m_params, d_params, t_params) in enumerate(all_configs):
                print(f"\n[{idx+1}/{len(all_configs)}] Training configuration:")
                print(f"  Model params: {m_params}")
                print(f"  Dataset params: {d_params}")
                print(f"  Training params: {t_params}")
                
                result = self.train_with_config(
                    model_type=model_type,
                    model_params=m_params,
                    dataset_params=d_params,
                    training_params=t_params,
                    data=data,
                    feature_cols=feature_cols,
                    target_col=target_col,
                    device=device
                )
                
                all_results.append(result)
                
                # Save intermediate results
                self.save_results(all_results, f"results_{model_type}_intermediate.json")
        
        # Save final results
        self.results = all_results
        self.save_results(all_results, "results_final.json")
        
        # Print summary
        self.print_summary()
        
        return all_results
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
    
    def print_summary(self):
        """Print summary of tuning results"""
        if not self.results:
            print("No results to summarize")
            return
        
        # Filter successful results
        successful_results = [r for r in self.results if r["status"] == "success"]
        
        if not successful_results:
            print("\nNo successful training runs")
            return
        
        # Sort by validation loss
        successful_results.sort(key=lambda x: x["best_val_loss"])
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*80)
        print(f"\nTotal runs: {len(self.results)}")
        print(f"Successful runs: {len(successful_results)}")
        print(f"Failed runs: {len(self.results) - len(successful_results)}")
        
        print("\n" + "-"*80)
        print("TOP 10 CONFIGURATIONS")
        print("-"*80)
        
        for idx, result in enumerate(successful_results[:10]):
            print(f"\n{idx+1}. Model: {result['model_type']}")
            print(f"   Best Val Loss: {result['best_val_loss']:.6f}")
            print(f"   Best Epoch: {result['best_epoch']}")
            print(f"   Run ID: {result['run_id']}")
            print(f"   Model Params: {result['model_params']}")
            print(f"   Dataset Params: {result['dataset_params']}")
            print(f"   Training Params: {result['training_params']}")


def main():
    """Main function to run hyperparameter tuning"""
    
    print("="*80)
    print("STARTING HYPERPARAMETER TUNING")
    print("="*80)
    
    # Set MLflow tracking URI (optional)
    mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "127.0.0.1:8080")
    use_mlflow = os.environ.get("USE_MLFLOW", "true").lower() == "true"  # Default to true
    
    print(f"\nMLflow Settings:")
    print(f"  Tracking URI: {mlflow_uri}")
    print(f"  Use MLflow: {use_mlflow}")
    print(f"  MLflow Available: {MLFLOW_AVAILABLE}")
    
    # Generate dummy data
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
    
    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = "Close"
    
    # Initialize tuner
    print(f"\nInitializing tuner (MLflow: {use_mlflow})...")
    tuner = HyperparameterTuner(mlflow_tracking_uri=mlflow_uri, use_mlflow=use_mlflow)
    
    # Run grid search
    # You can select which models to tune
    # models_to_tune = ["LSTM", "GRU", "NBERT", "Transformer"]
    models_to_tune = ["LSTM", "GRU", "NBERT", "Transformer"]
    
    print(f"\nModels to tune: {models_to_tune}")
    print(f"Max combinations per model: 10")
    
    results = tuner.grid_search(
        model_types=models_to_tune,
        data=data,
        feature_cols=feature_cols,
        target_col=target_col,
        max_combinations=10,  # Limit combinations per model for testing
        random_sample=True     # Randomly sample configurations
    )
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()