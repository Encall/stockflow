"""Stage 1: Find best scaler"""
from .base_stage import BaseStage
from typing import Tuple, List, Dict
import pandas as pd
import torch
from ..HyperparameterConfig import DATASET_PARAMS, DEFAULT_DATASET_CONFIG, get_default_model_config


class ScalerStage(BaseStage):
    """Stage 1: Find best data scaler"""
    
    def __init__(self, tuner):
        super().__init__(tuner, "stage1_scaler")
    
    def run(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, device: torch.device, **kwargs) -> Tuple[str, List[Dict]]:
        """Find best scaler"""
        self._print_header("STAGE 1: Finding Best Scaler")
        
        results = []
        default_model = "LSTM"
        default_config = get_default_model_config(default_model)
        default_training = kwargs.get("training_config", {})
        
        for scaler_name in DATASET_PARAMS["scaler"]:
            print(f"\nTesting scaler: {scaler_name}")
            result = self.tuner.train_with_config(
                model_type=default_model,
                model_params=default_config,
                dataset_params={
                    "seq_len": DEFAULT_DATASET_CONFIG["seq_len"], 
                    "scaler": scaler_name
                },
                training_params=default_training,
                data=data, feature_cols=feature_cols, target_col=target_col,
                device=device, stage=self.stage_name
            )
            results.append(result)
        
        best = self._find_best_result(results)
        best_scaler = best["dataset_params"]["scaler"] if best else DEFAULT_DATASET_CONFIG["scaler"]
        self._print_result(results, best_scaler, "Scaler")
        
        return best_scaler, results