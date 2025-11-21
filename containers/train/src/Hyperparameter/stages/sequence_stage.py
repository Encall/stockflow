"""Stage 2: Find best sequence length"""
from .base_stage import BaseStage
from typing import Tuple, List, Dict
import pandas as pd
import torch
from ..HyperparameterConfig import DATASET_PARAMS, DEFAULT_DATASET_CONFIG, get_default_model_config


class SequenceStage(BaseStage):
    """Stage 2: Find best sequence length"""
    
    def __init__(self, tuner):
        super().__init__(tuner, "stage2_seq_len")
    
    def run(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, device: torch.device, **kwargs) -> Tuple[int, List[Dict]]:
        """Find best sequence length"""
        self._print_header("STAGE 2: Finding Best Sequence Length")
        
        best_scaler = kwargs.get("best_scaler", DEFAULT_DATASET_CONFIG["scaler"])
        results = []
        default_model = "LSTM"
        default_config = get_default_model_config(default_model)
        default_training = kwargs.get("training_config", {})
        
        for seq_len in DATASET_PARAMS["seq_len"]:
            print(f"\nTesting seq_len: {seq_len}")
            result = self.tuner.train_with_config(
                model_type=default_model,
                model_params=default_config,
                dataset_params={"seq_len": seq_len, "scaler": best_scaler},
                training_params=default_training,
                data=data, feature_cols=feature_cols, target_col=target_col,
                device=device, stage=self.stage_name
            )
            results.append(result)
        
        best = self._find_best_result(results)
        best_seq_len = best["dataset_params"]["seq_len"] if best else DEFAULT_DATASET_CONFIG["seq_len"]
        self._print_result(results, best_seq_len, "Seq_len")
        
        return best_seq_len, results