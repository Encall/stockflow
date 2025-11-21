"""Stage 3: Find best model parameters"""
from .base_stage import BaseStage
from typing import Tuple, List, Dict
import pandas as pd
import torch
from ..HyperparameterConfig import MODEL_PARAMS, get_default_model_config


class ModelStage(BaseStage):
    """Stage 3: Find best model architecture parameters"""
    
    def __init__(self, tuner):
        super().__init__(tuner, "stage3_model")
    
    def run(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, device: torch.device, **kwargs) -> Tuple[Dict, List[Dict]]:
        """Find best model parameters"""
        model_type = kwargs.get("model_type")
        dataset_config = kwargs.get("dataset_config", {})
        training_config = kwargs.get("training_config", {})
        
        self._print_header(f"STAGE 3: Tuning {model_type}")
        
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
                    dataset_params=dataset_config,
                    training_params=training_config,
                    data=data, feature_cols=feature_cols, target_col=target_col,
                    device=device, stage=f"{self.stage_name}_{param_name}"
                )
                results.append(result)
        
        best = self._find_best_result(results)
        best_config = best["model_params"] if best else default_config
        
        if best:
            print(f"\n✅ Best {model_type} config: {best_config}")
            print(f"   Val Loss: {best['best_val_loss']:.6f}")
        else:
            print(f"\n❌ All {model_type} runs failed - using default")
        
        return best_config, results