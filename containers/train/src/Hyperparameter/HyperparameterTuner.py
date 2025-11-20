"""
Core HyperparameterTuner class for training models with different configurations
"""
import torch
import torch.nn as nn
import os
import json
import traceback
import pickle
from datetime import datetime
from typing import Dict, Any, List
import pandas as pd

# Try to import MLflow, but make it optional
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

import model.LSTM as LSTMModel
import model.Transformer as TransformerModel
import model.GRU as GRUModel
import model.NBERT as NBERTModel
import StockDataset
from src.Hyperparameter.HyperparameterConfig import SCALER_OPTIONS


class HyperparameterTuner:
    """Main tuner class for training and evaluating models"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://127.0.0.1:8080", use_mlflow: bool = False):
        """Initialize hyperparameter tuner with optional MLflow tracking"""
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow:
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
                return LSTMModel.LSTM(input_size=input_size, **model_params)
            elif model_type == "GRU":
                return GRUModel.GRU(input_size=input_size, **model_params)
            elif model_type == "NBERT":
                return NBERTModel.NBERT(input_size=input_size, seq_len=seq_len, **model_params)
            elif model_type == "Transformer":
                return TransformerModel.Transformer(input_size=input_size, **model_params)
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
            
            return total_loss / num_batches if num_batches > 0 else float('inf')
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
            
            return total_loss / num_batches if num_batches > 0 else float('inf')
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return float('inf')
    
    def _prepare_data(self, data: pd.DataFrame, feature_cols: List[str], target_col: str,
                     dataset_params: Dict, training_params: Dict, device: torch.device):
        """Prepare dataset and dataloaders"""
        scaler = SCALER_OPTIONS[dataset_params["scaler"]]
        seq_len = dataset_params["seq_len"]
        
        stock_data = StockDataset.MultiFeaturePriceDataset(
            data=data, feature_cols=feature_cols, target_col=target_col,
            seq_len=seq_len, scaler=scaler
        )
        
        # Split 80/20
        train_size = int(0.8 * len(stock_data))
        val_size = len(stock_data) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(stock_data, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=training_params["batch_size"], shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=training_params["batch_size"], shuffle=False
        )
        
        return train_loader, val_loader, seq_len
    
    def _train_loop(self, model, train_loader, val_loader, optimizer, loss_fn, device, epochs: int):
        """Main training loop with early stopping"""
        best_val_loss = float('inf')
        best_epoch = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_single_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = self.evaluate_model(model, val_loader, loss_fn, device)
            
            if self.use_mlflow:
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            
            print(f"Epoch {epoch+1}/{epochs} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
            
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
        
        return best_val_loss, best_epoch, epoch + 1
    
    def _log_mlflow_artifacts(self, model, run_name: str, model_params: Dict, 
                             dataset_params: Dict, training_params: Dict,
                             best_val_loss: float, best_epoch: int, total_epochs: int,
                             train_loss: float, val_loss: float, model_type: str, stage: str):
        """Log model and artifacts to MLflow"""
        if not self.use_mlflow:
            return
        
        model_dir = "temp_models"
        os.makedirs(model_dir, exist_ok=True)
        
        # Log PyTorch model
        mlflow.pytorch.log_model(model, "model")
        
        # Save state dict
        model_pth_path = f"{model_dir}/{run_name}_best_model.pth"
        torch.save(model.state_dict(), model_pth_path)
        mlflow.log_artifact(model_pth_path, "model_checkpoints")
        
        # Save pickle
        model_pkl_path = f"{model_dir}/{run_name}_model.pkl"
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(model, f)
        mlflow.log_artifact(model_pkl_path, "model_checkpoints")
        
        # Save config
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
        
        # Save summary
        summary_dict = {
            "model_type": model_type,
            "tuning_stage": stage,
            "final_train_loss": float(train_loss),
            "final_val_loss": float(val_loss),
            "best_val_loss": float(best_val_loss),
            "best_epoch": best_epoch,
            "total_epochs": total_epochs
        }
        summary_path = f"{model_dir}/{run_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_dict, f, indent=2)
        mlflow.log_artifact(summary_path, "summaries")
        
        # Clean up
        for temp_file in [model_pth_path, model_pkl_path, config_path, summary_path]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Set tags
        mlflow.log_metric("best_val_loss", best_val_loss)
        mlflow.log_metric("best_epoch", best_epoch)
        mlflow.set_tag("status", "success")
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("tuning_stage", stage)
        mlflow.set_tag("best_val_loss", f"{best_val_loss:.6f}")
        mlflow.set_tag("scaler", dataset_params["scaler"])
        mlflow.set_tag("loss_function", training_params["loss_fn"])
    
    def train_with_config(
        self,
        model_type: str,
        model_params: Dict[str, Any],
        dataset_params: Dict[str, Any],
        training_params: Dict[str, Any],
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        device: torch.device,
        stage: str = "full"
    ) -> Dict[str, Any]:
        """Train model with specific configuration"""
        
        run_name = f"{model_type}_{stage}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        run_id = None
        
        try:
            # Start MLflow run
            if self.use_mlflow:
                mlflow_run = mlflow.start_run(run_name=run_name)
                run_id = mlflow_run.info.run_id
                mlflow.log_param("model_type", model_type)
                mlflow.log_param("tuning_stage", stage)
                mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
                mlflow.log_params({f"dataset_{k}": v for k, v in dataset_params.items()})
                mlflow.log_params({f"training_{k}": v for k, v in training_params.items()})
            
            # Prepare data
            train_loader, val_loader, seq_len = self._prepare_data(
                data, feature_cols, target_col, dataset_params, training_params, device
            )
            
            # Create model
            model = self.create_model(model_type, model_params, len(feature_cols), seq_len)
            model = model.to(device)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=training_params["lr"])
            loss_fn = self.get_loss_function(training_params["loss_fn"])
            
            # Train
            best_val_loss, best_epoch, total_epochs = self._train_loop(
                model, train_loader, val_loader, optimizer, loss_fn, device, training_params["epochs"]
            )
            
            # Get final losses
            train_loss = self.train_single_epoch(model, train_loader, optimizer, loss_fn, device)
            val_loss = self.evaluate_model(model, val_loader, loss_fn, device)
            
            # Log artifacts
            self._log_mlflow_artifacts(
                model, run_name, model_params, dataset_params, training_params,
                best_val_loss, best_epoch, total_epochs, train_loss, val_loss, model_type, stage
            )
            
            if self.use_mlflow:
                mlflow.end_run()
            
            return {
                "run_id": run_id,
                "run_name": run_name,
                "model_type": model_type,
                "tuning_stage": stage,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
                "model_params": model_params,
                "dataset_params": dataset_params,
                "training_params": training_params,
                "status": "success",
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
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
                "tuning_stage": stage,
                "best_val_loss": float('inf'),
                "best_epoch": -1,
                "model_params": model_params,
                "dataset_params": dataset_params,
                "training_params": training_params,
                "status": "failed",
                "error": error_msg,
                "timestamp": datetime.now().isoformat()
            }
    
    def save_results(self, results: List[Dict], filename: str):
        """Save results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n💾 Results saved to {filename}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
