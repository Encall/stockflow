"""Core analyzer class for hyperparameter tuning results"""

import json
import os
from typing import List, Dict, Any
from collections import defaultdict


class ResultAnalyzer:
    """Load and organize hyperparameter tuning results"""
    
    def __init__(self, results_file: str = "results_final.json"):
        self.results_file = results_file
        self.results = []
        self.successful_results = []
        
    def load_results(self) -> bool:
        """Load results from JSON file"""
        if not os.path.exists(self.results_file):
            print(f"\n❌ Results file not found: {self.results_file}")
            print(f"   Please run hyperparameter tuning first:")
            print(f"   python run_tuning.py")
            return False
        
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            self.successful_results = [r for r in self.results if r.get("status") == "success"]
            
            print(f"\n📊 Loaded Results")
            print(f"{'─'*80}")
            print(f"   Total runs:      {len(self.results)}")
            print(f"   ✅ Successful:   {len(self.successful_results)}")
            print(f"   ❌ Failed:       {len(self.results) - len(self.successful_results)}")
            
            return True
            
        except Exception as e:
            print(f"\n❌ Error loading results: {e}")
            return False
    
    def get_best_overall(self) -> Dict[str, Any]:
        """Get best result overall"""
        if not self.successful_results:
            return {}
        return min(self.successful_results, key=lambda x: x["best_val_loss"])
    
    def get_results_by_stage(self) -> Dict[str, List[Dict]]:
        """Group results by tuning stage"""
        by_stage = defaultdict(list)
        for r in self.successful_results:
            stage = r.get('tuning_stage', 'unknown')
            by_stage[stage].append(r)
        return by_stage
    
    def get_results_by_model(self) -> Dict[str, List[Dict]]:
        """Group results by model type"""
        by_model = defaultdict(list)
        for r in self.successful_results:
            by_model[r['model_type']].append(r)
        return by_model
    
    def get_param_impact(self, param_path: str) -> Dict[Any, List[float]]:
        """Get parameter impact analysis
        
        Args:
            param_path: Path to parameter like 'dataset_params.scaler' or 'training_params.lr'
        """
        param_results = defaultdict(list)
        
        for r in self.successful_results:
            # Parse path
            parts = param_path.split('.')
            value = r
            for part in parts:
                value = value.get(part, {})
            
            if value and value != {}:
                param_results[value].append(r['best_val_loss'])
        
        return param_results
    
    def get_failed_results(self) -> List[Dict]:
        """Get all failed results"""
        return [r for r in self.results if r.get("status") == "failed"]
    
    def get_best_by_model(self) -> Dict[str, Dict[str, Any]]:
        """Get best result for each model type"""
        by_model = self.get_results_by_model()
        best_by_model = {}
        
        for model_type, results in by_model.items():
            best_by_model[model_type] = min(results, key=lambda x: x['best_val_loss'])
        
        return best_by_model
    
    def get_best_by_stage(self, stage: str) -> Dict[str, Any]:
        """Get best result for a specific stage"""
        stage_results = [r for r in self.successful_results 
                        if r.get('tuning_stage', '').startswith(stage.replace('stage3', 'stage3_'))]
        
        if not stage_results:
            return {}
        
        return min(stage_results, key=lambda x: x['best_val_loss'])
