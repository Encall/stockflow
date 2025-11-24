import random
import json
import os
from typing import Dict, Any, Optional

import pandas as pd

from Hyperparameter.HyperparameterConfig import (
    SCALER_OPTIONS,
    DATASET_PARAMS,
    MODEL_PARAMS,
    TRAINING_PARAMS,
    DEFAULT_TRAINING_CONFIG,
)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:
    mlflow = None
    MLFLOW_AVAILABLE = False

from ModelTrainer import ModelTrainer
import StockDataset
import torch


class ModelTuner:
    def __init__(self, data, feature_cols: list, target_col, experiment_name: str = "hyperparameter_tuning"):
        """Simple hyperparameter tuner.

        Args:
            data: pandas DataFrame (single asset) or dict of DataFrames (multi-asset).
            feature_cols: list of feature column names.
            target_col: name of target column (str) or list for multi-asset.
            experiment_name: name of the experiment for tracking purposes.
        """
        self.data = data
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.experiment_name = experiment_name
        self.is_multi_asset = isinstance(data, dict)
    def _sample_model_params(self, model_type: str) -> Dict[str, Any]:
        params = {}
        param_space = MODEL_PARAMS[model_type]
        for k, v in param_space.items():
            params[k] = random.choice(v)
        return params

    def tune(
        self,
        model_types: Optional[list] = None,
        n_trials: int = 10,
        batch_size: int = 128,
        use_mlflow: bool = False,
        mlflow_tracking_uri: Optional[str] = None,
        out_json: str = "tuning_results.json",
        exhaustive: bool = False,
    ) -> Dict[str, Any]:
        """Run random-search tuning over the parameter spaces and return best config.

        Args:
            exhaustive: If True, tries all possible parameter combinations for each model type.
                       Each model type gets its own MLflow experiment.
        """
        if model_types is None:
            model_types = list(MODEL_PARAMS.keys())

        results = []
        best = {"val_loss": float("inf")}

        # Set up MLflow tracking URI
        if use_mlflow and MLFLOW_AVAILABLE and mlflow is not None and mlflow_tracking_uri:
            try:
                if not mlflow_tracking_uri.startswith(("http://", "https://")):
                    mlflow_tracking_uri = f"http://{mlflow_tracking_uri}"
                mlflow.set_tracking_uri(mlflow_tracking_uri)
            except Exception as e:
                print(f"Warning: could not set up MLflow: {e}")

        for model_type in model_types:
            # Create separate experiment for each model type
            if use_mlflow and MLFLOW_AVAILABLE and mlflow is not None:
                try:
                    experiment_name = f"{self.experiment_name}_{model_type}"
                    mlflow.set_experiment(experiment_name)
                    print(f"\n{'='*60}")
                    print(f"MLflow Experiment: {experiment_name}")
                    print(f"{'='*60}")
                except Exception as e:
                    print(f"Warning: could not set experiment: {e}")
            
            if exhaustive:
                # Generate all possible combinations
                configs = self._generate_all_configs(model_type)
                print(f"\n{'='*60}")
                print(f"Testing ALL {len(configs)} parameter combinations for {model_type}")
                print(f"{'='*60}\n")
            else:
                # Random sampling
                trials_for_model = n_trials // len(model_types)
                configs = [self._sample_config(model_type, batch_size) for _ in range(trials_for_model)]
                print(f"\n{'='*60}")
                print(f"Testing {len(configs)} random configurations for {model_type}")
                print(f"{'='*60}\n")
            
            for trial_idx, config in enumerate(configs, 1):
                model_params = config["model_params"]
                seq_len = config["seq_len"]
                scaler_key = config["scaler"]
                scaler = SCALER_OPTIONS[scaler_key]
                lr = config["lr"]
                epochs = config["epochs"]
                bs = config["batch_size"]

                # Create dataset based on data type
                if self.is_multi_asset:
                    dataset = StockDataset.MultiAssetDataset(
                        data=self.data,
                        feature_cols=self.feature_cols,
                        target_col=self.target_col,
                        seq_len=seq_len,
                        scaler=scaler,
                    )
                else:
                    dataset = StockDataset.SingleAssetDataset(
                        data=self.data,
                        feature_cols=self.feature_cols,
                        target_col=self.target_col,
                        seq_len=seq_len,
                        scaler=scaler,
                    )

                if len(dataset) < 2:
                    print(f"Skipping trial {trial_idx}: not enough data for seq_len={seq_len}")
                    continue

                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=bs, shuffle=True)
                val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=bs, shuffle=False)

                # Get input size and seq_len from dataset
                sample_x, _ = dataset[0]
                input_size = sample_x.shape[-1]
                actual_seq_len = sample_x.shape[0] if len(sample_x.shape) > 1 else seq_len

                # Start MLflow run
                if use_mlflow and MLFLOW_AVAILABLE and mlflow is not None:
                    try:
                        loss_fn_name = DEFAULT_TRAINING_CONFIG["loss_fn"]
                        run_name = f"{model_type}_{loss_fn_name}_seq{seq_len}_lr{lr}_bs{bs}"
                        param_snippet = "".join([f"_{k}{v}" for k, v in list(model_params.items())[:2]])
                        run_name = run_name + param_snippet
                        mlflow.start_run(run_name=run_name)
                        try:
                            # Log parameters
                            mlflow.log_params({f"model_{k}": v for k, v in model_params.items()})
                            mlflow.log_param("seq_len", seq_len)
                            mlflow.log_param("scaler", scaler_key)
                            mlflow.log_param("lr", lr)
                            mlflow.log_param("batch_size", bs)
                            mlflow.log_param("model_type", model_type)
                            mlflow.log_param("loss_function", loss_fn_name)
                            
                            # Set tags for filtering
                            mlflow.set_tag("model_type", model_type)
                            mlflow.set_tag("loss_function", loss_fn_name)
                        except Exception:
                            pass
                    except Exception as e:
                        print(f"Warning: could not start MLflow run: {e}")

                trainer = ModelTrainer(
                    train_loader,
                    val_loader,
                    model_name=model_type,
                    model_params=model_params,
                    input_size=input_size,
                    seq_len=actual_seq_len,
                    learning_rate=lr,
                    epochs=epochs,
                    use_mlflow=use_mlflow,
                    mlflow_tracking_uri=mlflow_tracking_uri,
                )

                print(f"Trial {trial_idx}/{len(configs)}: seq_len={seq_len} scaler={scaler_key} lr={lr} bs={bs} params={model_params}")
                model = trainer.train_model(log_interval=20)

                val_loss = trainer._validate_model(trainer.loss_fn)

                # Capture run_id BEFORE ending the run
                current_run_id = None
                if use_mlflow and MLFLOW_AVAILABLE and mlflow is not None:
                    try:
                        active_run = mlflow.active_run()
                        if active_run:
                            current_run_id = active_run.info.run_id
                    except Exception:
                        pass

                # End MLflow run
                if use_mlflow and MLFLOW_AVAILABLE and mlflow is not None:
                    try:
                        mlflow.end_run()
                    except Exception:
                        pass

                record = {
                    "trial": trial_idx,
                    "model_type": model_type,
                    "model_params": model_params,
                    "seq_len": seq_len,
                    "scaler": scaler_key,
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": epochs,
                    "val_loss": float(val_loss),
                }
                
                # Add run_id if we captured it
                if current_run_id:
                    record["run_id"] = current_run_id
                
                results.append(record)

                if val_loss < best["val_loss"]:
                    best = record.copy()
                    print(f"🏆 New best! {model_type} with val_loss={val_loss:.6f}")

        return best
    
    def _sample_config(self, model_type: str, batch_size: int) -> Dict[str, Any]:
        """Sample a random configuration for the given model type."""
        model_params = self._sample_model_params(model_type)
        seq_len = random.choice(DATASET_PARAMS["seq_len"])
        scaler_key = random.choice(DATASET_PARAMS["scaler"])
        lr = random.choice(TRAINING_PARAMS["lr"]) if "lr" in TRAINING_PARAMS else DEFAULT_TRAINING_CONFIG["lr"]
        epochs = DEFAULT_TRAINING_CONFIG["epochs"]
        
        return {
            "model_params": model_params,
            "seq_len": seq_len,
            "scaler": scaler_key,
            "lr": lr,
            "batch_size": batch_size,
            "epochs": epochs,
        }
    
    def _generate_all_configs(self, model_type: str) -> list:
        """Generate all possible parameter combinations for the given model type."""
        import itertools
        
        configs = []
        
        # Get parameter spaces
        model_param_space = MODEL_PARAMS[model_type]
        seq_lens = DATASET_PARAMS["seq_len"]
        scalers = DATASET_PARAMS["scaler"]
        lrs = TRAINING_PARAMS.get("lr", [DEFAULT_TRAINING_CONFIG["lr"]])
        batch_sizes = TRAINING_PARAMS.get("batch_size", [DEFAULT_TRAINING_CONFIG["batch_size"]])
        epochs = DEFAULT_TRAINING_CONFIG["epochs"]
        
        # Generate all model param combinations
        param_names = list(model_param_space.keys())
        param_values = [model_param_space[name] for name in param_names]
        
        for model_param_combo in itertools.product(*param_values):
            model_params = dict(zip(param_names, model_param_combo))
            
            # Combine with dataset and training params
            for seq_len, scaler, lr, bs in itertools.product(seq_lens, scalers, lrs, batch_sizes):
                configs.append({
                    "model_params": model_params,
                    "seq_len": seq_len,
                    "scaler": scaler,
                    "lr": lr,
                    "batch_size": bs,
                    "epochs": epochs,
                })
        
        return configs
