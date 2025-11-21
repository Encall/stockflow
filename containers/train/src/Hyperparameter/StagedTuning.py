"""
Staged tuning strategies for hyperparameter optimization
"""
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from typing import Dict, Any, List, Tuple
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from Hyperparameter.HyperparameterConfig import (
    DATASET_PARAMS,
    MODEL_PARAMS,
    TRAINING_PARAMS,
    DEFAULT_DATASET_CONFIG,
    DEFAULT_TRAINING_CONFIG,
    get_default_model_config
)
from Hyperparameter.HyperparameterTuner import HyperparameterTuner


class StagedTuning:
    """Staged tuning strategy to reduce total combinations"""
    
    def __init__(self, tuner: HyperparameterTuner):
        self.tuner = tuner
    
    def _find_best_result(self, results: List[Dict]) -> Dict:
        """Find best result from a list based on validation loss"""
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            return None
        return min(successful, key=lambda x: x["best_val_loss"])
    
    def _setup_experiment(self, experiment_name: str = None) -> str:
        """Setup MLflow experiment"""
        if experiment_name is None:
            from datetime import datetime
            experiment_name = f"staged_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        if self.tuner.use_mlflow:
            try:
                experiment = mlflow.set_experiment(experiment_name)
                print(f"\nMLflow Experiment: {experiment_name}")
                print(f"Experiment ID: {experiment.experiment_id}")
            except Exception as e:
                print(f"Warning: Could not set MLflow experiment: {str(e)}")
                self.tuner.use_mlflow = False
        
        return experiment_name
    
    def stage1_find_best_scaler(
        self, data: pd.DataFrame, feature_cols: List[str], 
        target_col: str, device: torch.device
    ) -> Tuple[str, List[Dict]]:
        """Stage 1: Find best scaler"""
        print("\n" + "="*80)
        print("STAGE 1: Finding Best Scaler")
        print("="*80)
        
        results = []
        default_model = "LSTM"
        default_config = get_default_model_config(default_model)
        
        for scaler_name in DATASET_PARAMS["scaler"]:
            print(f"\nTesting scaler: {scaler_name}")
            result = self.tuner.train_with_config(
                model_type=default_model,
                model_params=default_config,
                dataset_params={"seq_len": DEFAULT_DATASET_CONFIG["seq_len"], "scaler": scaler_name},
                training_params=DEFAULT_TRAINING_CONFIG,
                data=data, feature_cols=feature_cols, target_col=target_col,
                device=device, stage="stage1_scaler"
            )
            results.append(result)
        
        best = self._find_best_result(results)
        if not best:
            print("\n❌ Stage 1 failed - using default")
            return DEFAULT_DATASET_CONFIG["scaler"], results
        
        best_scaler = best["dataset_params"]["scaler"]
        print(f"\n✅ Best Scaler: {best_scaler} (Val Loss: {best['best_val_loss']:.6f})")
        return best_scaler, results
    
    def stage2_find_best_seq_len(
        self, best_scaler: str, data: pd.DataFrame, 
        feature_cols: List[str], target_col: str, device: torch.device
    ) -> Tuple[int, List[Dict]]:
        """Stage 2: Find best sequence length"""
        print("\n" + "="*80)
        print("STAGE 2: Finding Best Sequence Length")
        print("="*80)
        
        results = []
        default_model = "LSTM"
        default_config = get_default_model_config(default_model)
        
        for seq_len in DATASET_PARAMS["seq_len"]:
            print(f"\nTesting seq_len: {seq_len}")
            result = self.tuner.train_with_config(
                model_type=default_model,
                model_params=default_config,
                dataset_params={"seq_len": seq_len, "scaler": best_scaler},
                training_params=DEFAULT_TRAINING_CONFIG,
                data=data, feature_cols=feature_cols, target_col=target_col,
                device=device, stage="stage2_seq_len"
            )
            results.append(result)
        
        best = self._find_best_result(results)
        if not best:
            print("\n❌ Stage 2 failed - using default")
            return DEFAULT_DATASET_CONFIG["seq_len"], results
        
        best_seq_len = best["dataset_params"]["seq_len"]
        print(f"\n✅ Best Seq_len: {best_seq_len} (Val Loss: {best['best_val_loss']:.6f})")
        return best_seq_len, results
    
    def stage3_find_best_model_params(
        self, model_type: str, best_dataset_config: Dict,
        data: pd.DataFrame, feature_cols: List[str], 
        target_col: str, device: torch.device
    ) -> Tuple[Dict, List[Dict]]:
        """Stage 3: Find best model parameters"""
        print(f"\n{'='*80}")
        print(f"Tuning {model_type}")
        print(f"{'='*80}")
        
        results = []
        param_grid = MODEL_PARAMS[model_type]
        default_config = get_default_model_config(model_type)
        
        # Test each parameter one at a time
        for param_name, param_values in param_grid.items():
            print(f"\nTesting parameter: {param_name}")
            for value in param_values:
                test_config = default_config.copy()
                test_config[param_name] = value
                print(f"  {param_name}={value}")
                
                result = self.tuner.train_with_config(
                    model_type=model_type,
                    model_params=test_config,
                    dataset_params=best_dataset_config,
                    training_params=DEFAULT_TRAINING_CONFIG,
                    data=data, feature_cols=feature_cols, target_col=target_col,
                    device=device, stage=f"stage3_model_{param_name}"
                )
                results.append(result)
        
        best = self._find_best_result(results)
        if not best:
            print(f"\n❌ All {model_type} runs failed - using default")
            return default_config, results
        
        best_config = best["model_params"]
        print(f"\n✅ Best {model_type} config: {best_config}")
        print(f"   Val Loss: {best['best_val_loss']:.6f}")
        return best_config, results
    
    def stage4_find_best_training_params(
        self, best_model_type: str, best_model_params: Dict,
        best_dataset_config: Dict, data: pd.DataFrame,
        feature_cols: List[str], target_col: str, device: torch.device
    ) -> Tuple[Dict, List[Dict]]:
        """Stage 4: Fine-tune training parameters"""
        print("\n" + "="*80)
        print("STAGE 4: Fine-tuning Training Parameters")
        print("="*80)
        print(f"\nUsing best model: {best_model_type}")
        print(f"Model params: {best_model_params}")
        
        results = []
        for param_name, param_values in TRAINING_PARAMS.items():
            print(f"\nTesting parameter: {param_name}")
            for value in param_values:
                test_config = DEFAULT_TRAINING_CONFIG.copy()
                test_config[param_name] = value
                print(f"  {param_name}={value}")
                
                result = self.tuner.train_with_config(
                    model_type=best_model_type,
                    model_params=best_model_params,
                    dataset_params=best_dataset_config,
                    training_params=test_config,
                    data=data, feature_cols=feature_cols, target_col=target_col,
                    device=device, stage=f"stage4_training_{param_name}"
                )
                results.append(result)
        
        best = self._find_best_result(results)
        if not best:
            print(f"\n❌ All training runs failed - using default")
            return DEFAULT_TRAINING_CONFIG, results
        
        best_config = best["training_params"]
        print(f"\n✅ Best training config: {best_config}")
        print(f"   Val Loss: {best['best_val_loss']:.6f}")
        return best_config, results
    
    def final_train_all_models(
        self, model_types: List[str], best_model_configs: Dict[str, Dict],
        best_dataset_config: Dict, best_training_config: Dict,
        data: pd.DataFrame, feature_cols: List[str], 
        target_col: str, device: torch.device
    ) -> List[Dict]:
        """Final: Train all models with optimal configurations"""
        print("\n" + "="*80)
        print("FINAL: Training All Models with Optimal Configurations")
        print("="*80)
        
        results = []
        for model_type in model_types:
            print(f"\nFinal training: {model_type}")
            result = self.tuner.train_with_config(
                model_type=model_type,
                model_params=best_model_configs[model_type],
                dataset_params=best_dataset_config,
                training_params=best_training_config,
                data=data, feature_cols=feature_cols, target_col=target_col,
                device=device, stage="final"
            )
            results.append(result)
        return results
    
    def print_summary(
        self, all_results: List[Dict], stage1_results: List[Dict],
        stage2_results: List[Dict], stage4_results: List[Dict],
        best_dataset_config: Dict, best_training_config: Dict,
        best_model_configs: Dict[str, Dict], final_results: List[Dict]
    ):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print("STAGED TUNING SUMMARY")
        print("="*80)
        
        stage3_count = len([r for r in all_results if 'stage3' in r.get('tuning_stage', '')])
        print(f"\nTotal runs: {len(all_results)}")
        print(f"Stage 1 (Scaler): {len(stage1_results)} runs")
        print(f"Stage 2 (Seq_len): {len(stage2_results)} runs")
        print(f"Stage 3 (Model params): {stage3_count} runs")
        print(f"Stage 4 (Training params): {len(stage4_results)} runs")
        print(f"Final: {len(final_results)} runs")
        
        print("\n" + "-"*80)
        print("BEST CONFIGURATIONS")
        print("-"*80)
        print(f"Dataset: {best_dataset_config}")
        print(f"Training: {best_training_config}")
        print("\nModel configs:")
        for model_type, config in best_model_configs.items():
            print(f"  {model_type}: {config}")
        
        print("\n" + "-"*80)
        print("FINAL MODEL PERFORMANCE")
        print("-"*80)
        successful = [r for r in final_results if r["status"] == "success"]
        if successful:
            sorted_final = sorted(successful, key=lambda x: x["best_val_loss"])
            for idx, result in enumerate(sorted_final):
                print(f"{idx+1}. {result['model_type']}: {result['best_val_loss']:.6f}")
        else:
            print("No successful final runs")
    
    def run_staged_search(
        self, model_types: List[str], data: pd.DataFrame,
        feature_cols: List[str], target_col: str, experiment_name: str = None
    ):
        """Execute complete staged grid search"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self._setup_experiment(experiment_name)
        all_results = []
        
        # Stage 1: Scaler
        best_scaler, stage1_results = self.stage1_find_best_scaler(
            data, feature_cols, target_col, device
        )
        all_results.extend(stage1_results)
        self.tuner.save_results(all_results, "results_stage1_scaler.json")
        
        if not any(r["status"] == "success" for r in stage1_results):
            print("\n❌ Stage 1 failed - aborting")
            return all_results
        
        # Stage 2: Seq_len
        best_seq_len, stage2_results = self.stage2_find_best_seq_len(
            best_scaler, data, feature_cols, target_col, device
        )
        all_results.extend(stage2_results)
        best_dataset_config = {"seq_len": best_seq_len, "scaler": best_scaler}
        self.tuner.save_results(all_results, "results_stage2_dataset.json")
        
        # Stage 3: Model params
        print("\n" + "="*80)
        print("STAGE 3: Finding Best Model Parameters")
        print("="*80)
        
        best_model_configs = {}
        for model_type in model_types:
            best_config, stage3_results = self.stage3_find_best_model_params(
                model_type, best_dataset_config, data, feature_cols, target_col, device
            )
            best_model_configs[model_type] = best_config
            all_results.extend(stage3_results)
            self.tuner.save_results(all_results, f"results_stage3_{model_type}.json")
        
        # Stage 4: Training params
        stage3_successful = [r for r in all_results 
                           if r.get("tuning_stage", "").startswith("stage3") 
                           and r["status"] == "success"]
        
        if not stage3_successful:
            print("\n❌ No successful models from stage 3 - using defaults")
            best_model_type = model_types[0]
            best_training_config = DEFAULT_TRAINING_CONFIG
            stage4_results = []
        else:
            best_overall = min(stage3_successful, key=lambda x: x["best_val_loss"])
            best_model_type = best_overall["model_type"]
            best_model_params = best_model_configs[best_model_type]
            
            best_training_config, stage4_results = self.stage4_find_best_training_params(
                best_model_type, best_model_params, best_dataset_config,
                data, feature_cols, target_col, device
            )
            all_results.extend(stage4_results)
        
        self.tuner.save_results(all_results, "results_stage4_training.json")
        
        # Final: Train all models
        final_results = self.final_train_all_models(
            model_types, best_model_configs, best_dataset_config, best_training_config,
            data, feature_cols, target_col, device
        )
        all_results.extend(final_results)
        
        # Save and summarize
        self.tuner.results = all_results
        self.tuner.save_results(all_results, "results_final.json")
        self.print_summary(
            all_results, stage1_results, stage2_results, stage4_results,
            best_dataset_config, best_training_config, best_model_configs, final_results
        )
        
        return all_results
