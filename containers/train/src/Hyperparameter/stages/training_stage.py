"""Stage 4: Find best training parameters"""
from .base_stage import BaseStage
from typing import Tuple, List, Dict
import pandas as pd
import torch
from ..HyperparameterConfig import TRAINING_PARAMS, DEFAULT_TRAINING_CONFIG


class TrainingStage(BaseStage):
    """Stage 4: Fine-tune training parameters"""
    
    def __init__(self, tuner):
        super().__init__(tuner, "stage4_training")
    
    def run(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, device: torch.device, **kwargs) -> Tuple[Dict, List[Dict]]:
        """Find best training parameters"""
        model_type = kwargs.get("model_type")
        model_params = kwargs.get("model_params", {})
        dataset_config = kwargs.get("dataset_config", {})
        
        self._print_header(f"STAGE 4: Fine-tuning Training Parameters for {model_type}")
        print(f"\nUsing model: {model_type}")
        print(f"Model params: {model_params}")
        
        results = []
        for param_name, param_values in TRAINING_PARAMS.items():
            print(f"\nTesting parameter: {param_name}")
            for value in param_values:
                test_config = DEFAULT_TRAINING_CONFIG.copy()
                test_config[param_name] = value
                print(f"  {param_name}={value}")
                
                result = self.tuner.train_with_config(
                    model_type=model_type,
                    model_params=model_params,
                    dataset_params=dataset_config,
                    training_params=test_config,
                    data=data, feature_cols=feature_cols, target_col=target_col,
                    device=device, stage=f"{self.stage_name}_{param_name}"
                )
                results.append(result)
        
        best = self._find_best_result(results)
        best_config = best["training_params"] if best else DEFAULT_TRAINING_CONFIG
        
        if best:
            print(f"\n✅ Best training config: {best_config}")
            print(f"   Val Loss: {best['best_val_loss']:.6f}")
        else:
            print(f"\n❌ All training runs failed - using default")
        
        return best_config, results