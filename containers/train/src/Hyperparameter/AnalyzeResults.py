#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results
Provides insights and visualizations of tuning experiments
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict


class ResultAnalyzer:
    def __init__(self, results_file: str = "results_final.json"):
        """Initialize analyzer with results file"""
        self.results_file = results_file
        self.results = []
        self.successful_results = []
        
    def load_results(self) -> bool:
        """Load results from JSON file"""
        if not os.path.exists(self.results_file):
            print(f"❌ Results file not found: {self.results_file}")
            return False
        
        try:
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)
            
            # Filter successful results
            self.successful_results = [r for r in self.results if r.get("status") == "success"]
            
            print(f"✓ Loaded {len(self.results)} results")
            print(f"✓ {len(self.successful_results)} successful runs")
            print(f"✓ {len(self.results) - len(self.successful_results)} failed runs")
            return True
            
        except Exception as e:
            print(f"❌ Error loading results: {e}")
            return False
    
    def print_summary(self):
        """Print overall summary"""
        if not self.successful_results:
            print("\n⚠️  No successful results to analyze")
            return
        
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING SUMMARY")
        print("="*80)
        
        # Group by model type
        by_model = defaultdict(list)
        for result in self.successful_results:
            by_model[result["model_type"]].append(result)
        
        print(f"\nTotal Experiments: {len(self.results)}")
        print(f"Successful: {len(self.successful_results)}")
        print(f"Failed: {len(self.results) - len(self.successful_results)}")
        
        print("\n" + "-"*80)
        print("Results by Model Type:")
        print("-"*80)
        for model_type, results in by_model.items():
            best_loss = min(r["best_val_loss"] for r in results)
            avg_loss = sum(r["best_val_loss"] for r in results) / len(results)
            print(f"\n{model_type}:")
            print(f"  - Experiments: {len(results)}")
            print(f"  - Best Val Loss: {best_loss:.6f}")
            print(f"  - Average Val Loss: {avg_loss:.6f}")
    
    def print_top_n(self, n: int = 10):
        """Print top N best configurations"""
        if not self.successful_results:
            return
        
        # Sort by validation loss
        sorted_results = sorted(self.successful_results, key=lambda x: x["best_val_loss"])
        
        print("\n" + "="*80)
        print(f"TOP {min(n, len(sorted_results))} BEST CONFIGURATIONS")
        print("="*80)
        
        for idx, result in enumerate(sorted_results[:n]):
            print(f"\n{'─'*80}")
            print(f"#{idx+1} - {result['model_type']}")
            print(f"{'─'*80}")
            print(f"Best Val Loss:    {result['best_val_loss']:.6f}")
            print(f"Best Epoch:       {result['best_epoch']}")
            print(f"Run Name:         {result['run_name']}")
            print(f"Run ID:           {result['run_id']}")
            
            print("\nModel Parameters:")
            for key, value in result['model_params'].items():
                print(f"  - {key:20s}: {value}")
            
            print("\nDataset Parameters:")
            for key, value in result['dataset_params'].items():
                print(f"  - {key:20s}: {value}")
            
            print("\nTraining Parameters:")
            for key, value in result['training_params'].items():
                print(f"  - {key:20s}: {value}")
    
    def print_best_by_model(self):
        """Print best configuration for each model type"""
        if not self.successful_results:
            return
        
        # Group by model type
        by_model = defaultdict(list)
        for result in self.successful_results:
            by_model[result["model_type"]].append(result)
        
        print("\n" + "="*80)
        print("BEST CONFIGURATION PER MODEL TYPE")
        print("="*80)
        
        for model_type in sorted(by_model.keys()):
            results = by_model[model_type]
            best_result = min(results, key=lambda x: x["best_val_loss"])
            
            print(f"\n{'─'*80}")
            print(f"{model_type}")
            print(f"{'─'*80}")
            print(f"Best Val Loss:    {best_result['best_val_loss']:.6f}")
            print(f"Best Epoch:       {best_result['best_epoch']}")
            print(f"Run Name:         {best_result['run_name']}")
            print(f"Run ID:           {best_result['run_id']}")
            
            print("\nOptimal Configuration:")
            print("  Model:")
            for key, value in best_result['model_params'].items():
                print(f"    - {key:20s}: {value}")
            
            print("  Dataset:")
            for key, value in best_result['dataset_params'].items():
                print(f"    - {key:20s}: {value}")
            
            print("  Training:")
            for key, value in best_result['training_params'].items():
                print(f"    - {key:20s}: {value}")
    
    def analyze_hyperparameters(self):
        """Analyze impact of individual hyperparameters"""
        if not self.successful_results:
            return
        
        print("\n" + "="*80)
        print("HYPERPARAMETER IMPACT ANALYSIS")
        print("="*80)
        
        # Group by model type
        by_model = defaultdict(list)
        for result in self.successful_results:
            by_model[result["model_type"]].append(result)
        
        for model_type, results in by_model.items():
            print(f"\n{'─'*80}")
            print(f"{model_type} - Parameter Analysis")
            print(f"{'─'*80}")
            
            # Analyze model parameters
            self._analyze_param_group(results, "model_params", "Model Parameters")
            
            # Analyze dataset parameters
            self._analyze_param_group(results, "dataset_params", "Dataset Parameters")
            
            # Analyze training parameters
            self._analyze_param_group(results, "training_params", "Training Parameters")
    
    def _analyze_param_group(self, results: List[Dict], param_group: str, title: str):
        """Analyze impact of parameters in a group"""
        print(f"\n{title}:")
        
        # Collect all parameter values
        param_values = defaultdict(lambda: defaultdict(list))
        
        for result in results:
            params = result.get(param_group, {})
            for param_name, param_value in params.items():
                param_values[param_name][param_value].append(result["best_val_loss"])
        
        # Calculate statistics for each parameter
        for param_name, value_dict in sorted(param_values.items()):
            print(f"\n  {param_name}:")
            
            # Sort by average loss
            value_stats = []
            for value, losses in value_dict.items():
                avg_loss = sum(losses) / len(losses)
                value_stats.append((value, avg_loss, len(losses)))
            
            value_stats.sort(key=lambda x: x[1])
            
            for value, avg_loss, count in value_stats:
                print(f"    {str(value):20s}: avg_loss={avg_loss:.6f} (n={count})")
    
    def print_failed_runs(self):
        """Print information about failed runs"""
        failed_results = [r for r in self.results if r.get("status") == "failed"]
        
        if not failed_results:
            print("\n✓ No failed runs")
            return
        
        print("\n" + "="*80)
        print(f"FAILED RUNS ({len(failed_results)})")
        print("="*80)
        
        for idx, result in enumerate(failed_results):
            print(f"\n{idx+1}. {result['model_type']} - {result['run_name']}")
            if "error" in result:
                error_lines = result["error"].split("\n")
                print(f"   Error: {error_lines[0]}")
    
    def save_best_configs(self, output_file: str = "best_configs.json"):
        """Save best configuration for each model type"""
        if not self.successful_results:
            print("\n⚠️  No successful results to save")
            return
        
        # Group by model type and get best
        by_model = defaultdict(list)
        for result in self.successful_results:
            by_model[result["model_type"]].append(result)
        
        best_configs = {}
        for model_type, results in by_model.items():
            best_result = min(results, key=lambda x: x["best_val_loss"])
            best_configs[model_type] = {
                "best_val_loss": best_result["best_val_loss"],
                "best_epoch": best_result["best_epoch"],
                "run_id": best_result["run_id"],
                "run_name": best_result["run_name"],
                "model_params": best_result["model_params"],
                "dataset_params": best_result["dataset_params"],
                "training_params": best_result["training_params"]
            }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(best_configs, f, indent=2)
            print(f"\n✓ Best configurations saved to {output_file}")
        except Exception as e:
            print(f"\n❌ Error saving best configs: {e}")


def main():
    """Main analysis function"""
    
    # Initialize analyzer
    analyzer = ResultAnalyzer("results_final.json")
    
    # Load results
    if not analyzer.load_results():
        print("\n⚠️  Cannot proceed without results file")
        print("Run hyperparameter tuning first: python run_tuning.py")
        return
    
    # Perform analyses
    analyzer.print_summary()
    analyzer.print_top_n(10)
    analyzer.print_best_by_model()
    analyzer.analyze_hyperparameters()
    analyzer.print_failed_runs()
    analyzer.save_best_configs()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("  1. Review best_configs.json for optimal configurations")
    print("  2. Check MLflow UI for detailed metrics: http://127.0.0.1:8080")
    print("  3. Use best model for deployment")
    print("="*80)


if __name__ == "__main__":
    main()
