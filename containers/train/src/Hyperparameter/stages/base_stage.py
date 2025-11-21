"""Base class for all tuning stages"""
from typing import Dict, List, Tuple, Any
import pandas as pd
import torch
from abc import ABC, abstractmethod


class BaseStage(ABC):
    """Base class for hyperparameter tuning stages"""
    
    def __init__(self, tuner, stage_name: str):
        self.tuner = tuner
        self.stage_name = stage_name
    
    def _find_best_result(self, results: List[Dict]) -> Dict:
        """Find best result from a list based on validation loss"""
        successful = [r for r in results if r["status"] == "success"]
        if not successful:
            return None
        return min(successful, key=lambda x: x["best_val_loss"])
    
    def _print_header(self, title: str):
        """Print stage header"""
        print("\n" + "="*80)
        print(title)
        print("="*80)
    
    def _print_result(self, results: List[Dict], best_value: Any, metric_name: str):
        """Print stage results"""
        best = self._find_best_result(results)
        if best:
            print(f"\n✅ Best {metric_name}: {best_value} (Val Loss: {best['best_val_loss']:.6f})")
        else:
            print(f"\n❌ Stage failed - using default")
    
    @abstractmethod
    def run(self, data: pd.DataFrame, feature_cols: List[str], 
            target_col: str, device: torch.device, **kwargs) -> Tuple[Any, List[Dict]]:
        """Run the stage - must be implemented by subclasses"""
        pass