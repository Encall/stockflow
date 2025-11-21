"""Staged tuning orchestrator - simplified"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from typing import Dict, List
import pandas as pd

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from Hyperparameter.HyperparameterConfig import DEFAULT_TRAINING_CONFIG
from Hyperparameter.HyperparameterTuner import HyperparameterTuner
from Hyperparameter.ModelRegistry import register_model
from Hyperparameter.stages.scaler_stage import ScalerStage
from Hyperparameter.stages.sequence_stage import SequenceStage
from Hyperparameter.stages.model_stage import ModelStage
from Hyperparameter.stages.training_stage import TrainingStage


class StagedTuning:
    """Orchestrates staged hyperparameter tuning"""
    
    def __init__(self, tuner: HyperparameterTuner):
        self.tuner = tuner
        self.scaler_stage = ScalerStage(tuner)
        self.sequence_stage = SequenceStage(tuner)
        self.model_stage = ModelStage(tuner)
        self.training_stage = TrainingStage(tuner)
    
    def _setup_experiment(self, stock_symbol: str):
        """Setup MLflow experiment"""
        experiment_name = f"stock_{stock_symbol}"
        if self.tuner.use_mlflow:
            try:
                mlflow.set_experiment(experiment_name)
                print(f"\nMLflow Experiment: {experiment_name}")
            except Exception as e:
                print(f"Warning: MLflow setup failed: {e}")
                self.tuner.use_mlflow = False
    
    def run_staged_search(
        self, model_types: List[str], data: pd.DataFrame,
        feature_cols: List[str], target_col: str, stock_symbol: str
    ):
        """Execute complete staged grid search"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        self._setup_experiment(stock_symbol)
        all_results = []
        
        # Stage 1: Scaler
        best_scaler, stage1_results = self.scaler_stage.run(
            data, feature_cols, target_col, device,
            training_config=DEFAULT_TRAINING_CONFIG
        )
        all_results.extend(stage1_results)
        self.tuner.save_results(all_results, f"results_{stock_symbol}_stage1.json")
        
        if not any(r["status"] == "success" for r in stage1_results):
            print("\n❌ Stage 1 failed - aborting")
            return all_results
        
        # Stage 2: Sequence Length
        best_seq_len, stage2_results = self.sequence_stage.run(
            data, feature_cols, target_col, device,
            best_scaler=best_scaler,
            training_config=DEFAULT_TRAINING_CONFIG
        )
        all_results.extend(stage2_results)
        best_dataset_config = {"seq_len": best_seq_len, "scaler": best_scaler}
        self.tuner.save_results(all_results, f"results_{stock_symbol}_stage2.json")
        
        # Stage 3: Model Parameters (for each model)
        print("\n" + "="*80)
        print("STAGE 3: Finding Best Model Parameters")
        print("="*80)
        
        best_model_configs = {}
        for model_type in model_types:
            best_config, stage3_results = self.model_stage.run(
                data, feature_cols, target_col, device,
                model_type=model_type,
                dataset_config=best_dataset_config,
                training_config=DEFAULT_TRAINING_CONFIG
            )
            best_model_configs[model_type] = best_config
            all_results.extend(stage3_results)
        
        self.tuner.save_results(all_results, f"results_{stock_symbol}_stage3.json")
        
        # Stage 4: Training Parameters (for each model)
        print("\n" + "="*80)
        print("STAGE 4: Fine-tuning Training Parameters")
        print("="*80)
        
        best_training_configs = {}
        stage4_all_results = []
        
        for model_type in model_types:
            if model_type not in best_model_configs:
                continue
            
            print(f"\n{'─'*80}")
            best_training, stage4_results = self.training_stage.run(
                data, feature_cols, target_col, device,
                model_type=model_type,
                model_params=best_model_configs[model_type],
                dataset_config=best_dataset_config
            )
            best_training_configs[model_type] = best_training
            stage4_all_results.extend(stage4_results)
            all_results.extend(stage4_results)
        
        self.tuner.save_results(all_results, f"results_{stock_symbol}_stage4.json")
        
        # Final: Register best models
        final_results = self._finalize_and_register(
            model_types, stage4_all_results, stock_symbol
        )
        all_results.extend(final_results)
        
        self.tuner.save_results(all_results, f"results_{stock_symbol}_final.json")
        self._print_summary(
            all_results, stage1_results, stage2_results, stage4_all_results,
            best_dataset_config, best_training_configs, best_model_configs,
            final_results, stock_symbol
        )
        
        return all_results
    
    def _finalize_and_register(self, model_types: List[str], 
                              stage4_results: List[Dict], 
                              stock_symbol: str) -> List[Dict]:
        """Select best models and register to MLflow"""
        print("\n" + "="*80)
        print("FINAL: Selecting Best Models")
        print("="*80)
        
        final_results = []
        registered_models = []
        
        for model_type in model_types:
            model_results = [r for r in stage4_results 
                           if r.get("model_type") == model_type 
                           and r["status"] == "success"]
            
            if not model_results:
                continue
            
            best_result = min(model_results, key=lambda x: x["best_val_loss"])
            best_result["tuning_stage"] = "final"
            best_result["stock_symbol"] = stock_symbol
            final_results.append(best_result)
            
            print(f"\n🏆 {model_type}")
            print(f"   Val Loss: {best_result['best_val_loss']:.6f}")
        
        # Find overall best and register
        if final_results and self.tuner.use_mlflow:
            best_overall = min(final_results, key=lambda x: x["best_val_loss"])
            
            print("\n" + "="*80)
            print(f"🏆 BEST OVERALL: {best_overall['model_type']}")
            print(f"   Val Loss: {best_overall['best_val_loss']:.6f}")
            print("="*80)
            
            print("\n📦 Registering models...")
            for result in final_results:
                is_best = (result["model_type"] == best_overall["model_type"])
                info = register_model(
                    run_id=result.get("run_id"),
                    model_name=f"{stock_symbol}_{result['model_type']}",
                    stock_symbol=stock_symbol,
                    model_type=result["model_type"],
                    val_loss=result["best_val_loss"],
                    dataset_params=result["dataset_params"],
                    model_params=result["model_params"],
                    training_params=result["training_params"],
                    is_best=is_best
                )
                if info:
                    registered_models.append(info)
        
        return final_results
    
    def _print_summary(self, all_results, stage1_results, stage2_results, 
                      stage4_results, dataset_config, training_configs, 
                      model_configs, final_results, stock_symbol):
        """Print comprehensive summary"""
        print("\n" + "="*80)
        print(f"SUMMARY - {stock_symbol}")
        print("="*80)
        
        print(f"\nTotal runs: {len(all_results)}")
        print(f"  Stage 1: {len(stage1_results)}")
        print(f"  Stage 2: {len(stage2_results)}")
        print(f"  Stage 3: {len([r for r in all_results if 'stage3' in r.get('tuning_stage','')])}")
        print(f"  Stage 4: {len(stage4_results)}")
        print(f"  Final: {len(final_results)}")
        
        print(f"\nBest Dataset Config: {dataset_config}")
        
        print("\nFinal Models:")
        successful = [r for r in final_results if r["status"] == "success"]
        if successful:
            sorted_final = sorted(successful, key=lambda x: x["best_val_loss"])
            for idx, r in enumerate(sorted_final, 1):
                status = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
                print(f"  {status} {r['model_type']:12s}  {r['best_val_loss']:.6f}")
        
        print("\n🎉 Tuning complete!")