import model.LSTM as LSTMModel
import model.Transformer as TransformerModel
import model.GRU as GRUModel
import model.NBERT as NBERTModel
import torch
import torch.nn as nn

try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from typing import Dict, Any, List, Optional
import time


class ModelTrainer:
    def __init__(
        self,
        train_data,
        test_data,
        model_name: str,
        model_params: Dict[str, Any],
        input_size: int,
        seq_len: int = None,
        learning_rate: float = 0.001,
        epochs: int = 30,
        loss_name: str = "MSE",
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        model_experiment_name: str = "hyperparameter_tuning"
    ):
        self.train_data = train_data
        self.test_data = test_data
        self.model_name = model_name
        self.model_params = model_params
        self.model_experiment_name = model_experiment_name
        self.input_size = input_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.model = self._create_model()
        self.loss_name = loss_name
        self.loss_fn = self._get_loss_function()

        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        if self.use_mlflow and mlflow_tracking_uri:
            if not mlflow_tracking_uri.startswith(("http://", "https://")):
                mlflow_tracking_uri = f"http://{mlflow_tracking_uri}"
            mlflow.set_tracking_uri(mlflow_tracking_uri)
            print(f"MLflow tracking enabled: {mlflow_tracking_uri}")
        elif use_mlflow and not MLFLOW_AVAILABLE:
            print("MLflow requested but not available (missing dependency)")
        
        
    def _get_loss_function(self):
        """Get loss function by name"""
        loss_functions = {
            "MSE": nn.MSELoss(),
            "MAE": nn.L1Loss(),
            "Huber": nn.HuberLoss()
        }
        return loss_functions.get(self.loss_name, nn.MSELoss())

    def _create_model(self):
        """Create model instance based on type and parameters"""
        try:
            if self.model_name == "LSTM":
                return LSTMModel.LSTM(input_size=self.input_size, **self.model_params)
            elif self.model_name == "GRU":
                return GRUModel.GRU(input_size=self.input_size, **self.model_params)
            elif self.model_name == "NBERT":
                return NBERTModel.NBERT(input_size=self.input_size, seq_len=self.seq_len, **self.model_params)
            elif self.model_name == "Transformer":
                return TransformerModel.Transformer(input_size=self.input_size, **self.model_params)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
        except Exception as e:
            print(f"Error creating model {self.model_name}: {str(e)}")
            raise
        
    def _validate_model(self, loss_fn=torch.nn.MSELoss()):
        """Evaluate model on validation set."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for data, target in self.test_data:
                output = self.model(data)
                loss = loss_fn(output, target)
                total_loss += loss.item()

        avg_loss = total_loss / len(self.test_data)
        return avg_loss
    
    def train_model(self, log_interval=None):
        """Train model and validate every epoch."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Check if MLflow run is already active (started by Tuner)
        run_already_active = False
        if self.use_mlflow and MLFLOW_AVAILABLE:
            try:
                active_run = mlflow.active_run()
                run_already_active = active_run is not None
                if not run_already_active:
                    # Only start a new run if none is active
                    run_name = f"{self.model_name}_{self.loss_name}_{int(time.time())}"
                    mlflow.start_run(run_name=run_name)
                    mlflow.log_param("model_name", self.model_name)
                    mlflow.log_param("model_type", self.model_name)
                    mlflow.log_param("learning_rate", self.learning_rate)
                    mlflow.log_param("epochs", self.epochs)
                    mlflow.log_param("loss_fn", self.loss_name)
                    mlflow.log_param("loss_function", self.loss_name)
                    for k, v in self.model_params.items():
                        mlflow.log_param(f"model_{k}", v)
                    # Set tags
                    mlflow.set_tag("model_type", self.model_name)
                    mlflow.set_tag("loss_function", self.loss_name)
            except Exception as e:
                print(f"Warning: MLflow setup issue: {e}")

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (data, target) in enumerate(self.train_data):
                optimizer.zero_grad()

                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                if log_interval is not None:
                    if (batch_idx + 1) % log_interval == 0:
                        print(
                            f"Epoch [{epoch + 1}/{self.epochs}], "
                            f"Step [{batch_idx + 1}/{len(self.train_data)}], "
                            f"Loss: {loss.item():.6f}"
                        )

            try:
                avg_train_loss = total_loss / len(self.train_data)
            except Exception:
                avg_train_loss = float('inf')
            avg_val_loss = self._validate_model(self.loss_fn)

            if self.use_mlflow:
                try:
                    mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                    mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                except Exception:
                    pass

            print(
                f"Epoch {epoch + 1}/{self.epochs} "
                f"Train Loss: {avg_train_loss:.6f} "
                f"Val Loss: {avg_val_loss:.6f}"
            )

        if self.use_mlflow:
            try:
                # Get a sample input for signature inference
                sample_batch = next(iter(self.train_data))
                sample_input = sample_batch[0][:1]  # Take first sample from batch
                
                # Log model with signature
                mlflow.pytorch.log_model(
                    self.model, 
                    artifact_path="model",
                    input_example=sample_input.numpy()
                )
                
                # Save and log model state dict as artifact
                import tempfile
                import os
                with tempfile.TemporaryDirectory() as tmpdir:
                    model_path = os.path.join(tmpdir, "model.pth")
                    torch.save(self.model.state_dict(), model_path)
                    mlflow.log_artifact(model_path, artifact_path="checkpoints")
                    
                    # Also save full model
                    full_model_path = os.path.join(tmpdir, "full_model.pth")
                    torch.save(self.model, full_model_path)
                    mlflow.log_artifact(full_model_path, artifact_path="checkpoints")
                
                # Log final validation loss as artifact
                mlflow.log_metric("final_val_loss", avg_val_loss)
                mlflow.log_metric("final_train_loss", avg_train_loss)
                
                # Log model architecture as text artifact
                import io
                model_summary = io.StringIO()
                model_summary.write(f"Model: {self.model_name}\n")
                model_summary.write(f"Input Size: {self.input_size}\n")
                if self.seq_len:
                    model_summary.write(f"Sequence Length: {self.seq_len}\n")
                model_summary.write(f"\nModel Parameters:\n")
                for k, v in self.model_params.items():
                    model_summary.write(f"  {k}: {v}\n")
                model_summary.write(f"\nTraining Config:\n")
                model_summary.write(f"  Learning Rate: {self.learning_rate}\n")
                model_summary.write(f"  Epochs: {self.epochs}\n")
                model_summary.write(f"  Loss Function: {self.loss_name}\n")
                model_summary.write(f"\nFinal Results:\n")
                model_summary.write(f"  Final Train Loss: {avg_train_loss:.6f}\n")
                model_summary.write(f"  Final Val Loss: {avg_val_loss:.6f}\n")
                model_summary.write(f"\nModel Architecture:\n{str(self.model)}\n")
                
                mlflow.log_text(model_summary.getvalue(), "model_summary.txt")
                
            except Exception as e:
                print(f"Warning: failed to log model to MLflow: {e}")
            
            # Only end run if we started it (not if Tuner started it)
            if not run_already_active:
                try:
                    mlflow.end_run()
                except Exception:
                    pass

        return self.model
    
