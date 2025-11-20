#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results with beautiful formatting
"""

import json
import os
from typing import Dict, List, Any
from collections import defaultdict
from datetime import datetime

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class ResultAnalyzer:
    """Analyze and visualize hyperparameter tuning results"""
    
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
    
    def print_executive_summary(self):
        """Print executive summary with key findings"""
        if not self.successful_results:
            print("\n⚠️  No successful results to analyze")
            return
        
        print("\n" + "="*80)
        print("📈 EXECUTIVE SUMMARY")
        print("="*80)
        
        # Overall best model
        best = min(self.successful_results, key=lambda x: x["best_val_loss"])
        
        print(f"\n🏆 BEST OVERALL MODEL")
        print(f"{'─'*80}")
        print(f"   Model:           {best['model_type']}")
        print(f"   Val Loss:        {best['best_val_loss']:.6f}")
        print(f"   Run ID:          {best.get('run_id', 'N/A')}")
        
        # Best by stage
        by_stage = defaultdict(list)
        for r in self.successful_results:
            by_stage[r.get('tuning_stage', 'unknown')].append(r)
        
        if 'final' in by_stage:
            print(f"\n🎯 FINAL MODELS COMPARISON")
            print(f"{'─'*80}")
            final_sorted = sorted(by_stage['final'], key=lambda x: x['best_val_loss'])
            for idx, result in enumerate(final_sorted, 1):
                status = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
                print(f"   {status} {result['model_type']:12s}  Val Loss: {result['best_val_loss']:.6f}")
        
        # Best configurations
        if 'final' in by_stage:
            best_final = min(by_stage['final'], key=lambda x: x['best_val_loss'])
            
            print(f"\n⚙️  OPTIMAL CONFIGURATION")
            print(f"{'─'*80}")
            
            print(f"\n   Dataset:")
            for key, val in best_final['dataset_params'].items():
                print(f"      • {key:12s}: {val}")
            
            print(f"\n   Model ({best_final['model_type']}):")
            for key, val in best_final['model_params'].items():
                print(f"      • {key:12s}: {val}")
            
            print(f"\n   Training:")
            for key, val in best_final['training_params'].items():
                print(f"      • {key:12s}: {val}")
    
    def print_stage_breakdown(self):
        """Print breakdown by tuning stage"""
        if not self.successful_results:
            return
        
        print("\n" + "="*80)
        print("🔍 STAGE-BY-STAGE BREAKDOWN")
        print("="*80)
        
        by_stage = defaultdict(list)
        for r in self.successful_results:
            stage = r.get('tuning_stage', 'unknown')
            by_stage[stage].append(r)
        
        stage_order = ['stage1_scaler', 'stage2_seq_len', 'stage3', 'stage4', 'final']
        stage_names = {
            'stage1_scaler': '1️⃣  Stage 1: Scaler Selection',
            'stage2_seq_len': '2️⃣  Stage 2: Sequence Length',
            'stage3': '3️⃣  Stage 3: Model Architecture',
            'stage4': '4️⃣  Stage 4: Training Optimization',
            'final': '🎯 Final: Production Models'
        }
        
        for stage_key in stage_order:
            results = [r for r in self.successful_results 
                      if r.get('tuning_stage', '').startswith(stage_key.replace('stage3', 'stage3_'))]
            
            if results:
                best = min(results, key=lambda x: x['best_val_loss'])
                
                print(f"\n{stage_names.get(stage_key, stage_key)}")
                print(f"{'─'*80}")
                print(f"   Experiments:     {len(results)}")
                print(f"   Best Val Loss:   {best['best_val_loss']:.6f}")
                
                if stage_key == 'stage1_scaler':
                    print(f"   ✅ Best Scaler:  {best['dataset_params']['scaler']}")
                elif stage_key == 'stage2_seq_len':
                    print(f"   ✅ Best Seq Len: {best['dataset_params']['seq_len']}")
                elif stage_key == 'stage3':
                    print(f"   ✅ Best Model:   {best['model_type']}")
    
    def print_model_comparison(self):
        """Print detailed model comparison"""
        if not self.successful_results:
            return
        
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        by_model = defaultdict(list)
        for r in self.successful_results:
            by_model[r['model_type']].append(r)
        
        # Create comparison table
        print(f"\n{'Model':<15} {'Runs':>6} {'Best Loss':>12} {'Avg Loss':>12} {'Worst Loss':>12}")
        print(f"{'─'*15} {'─'*6:>6} {'─'*12:>12} {'─'*12:>12} {'─'*12:>12}")
        
        for model_type in sorted(by_model.keys()):
            results = by_model[model_type]
            losses = [r['best_val_loss'] for r in results]
            
            best_loss = min(losses)
            avg_loss = sum(losses) / len(losses)
            worst_loss = max(losses)
            
            print(f"{model_type:<15} {len(results):>6} {best_loss:>12.6f} {avg_loss:>12.6f} {worst_loss:>12.6f}")
    
    def print_hyperparameter_insights(self):
        """Print insights about hyperparameter impact"""
        if not self.successful_results:
            return
        
        print("\n" + "="*80)
        print("💡 HYPERPARAMETER INSIGHTS")
        print("="*80)
        
        # Analyze scaler impact
        scaler_results = defaultdict(list)
        for r in self.successful_results:
            scaler = r.get('dataset_params', {}).get('scaler')
            if scaler:
                scaler_results[scaler].append(r['best_val_loss'])
        
        if scaler_results:
            print(f"\n📏 Scaler Impact:")
            print(f"{'─'*80}")
            for scaler in sorted(scaler_results.keys(), 
                               key=lambda x: sum(scaler_results[x])/len(scaler_results[x])):
                losses = scaler_results[scaler]
                avg = sum(losses) / len(losses)
                print(f"   {scaler:12s}  Avg Loss: {avg:.6f}  (n={len(losses)})")
        
        # Analyze sequence length impact
        seq_results = defaultdict(list)
        for r in self.successful_results:
            seq_len = r.get('dataset_params', {}).get('seq_len')
            if seq_len:
                seq_results[seq_len].append(r['best_val_loss'])
        
        if seq_results:
            print(f"\n📊 Sequence Length Impact:")
            print(f"{'─'*80}")
            for seq_len in sorted(seq_results.keys(), 
                                key=lambda x: sum(seq_results[x])/len(seq_results[x])):
                losses = seq_results[seq_len]
                avg = sum(losses) / len(losses)
                print(f"   seq_len={seq_len:3d}   Avg Loss: {avg:.6f}  (n={len(losses)})")
        
        # Analyze learning rate impact
        lr_results = defaultdict(list)
        for r in self.successful_results:
            lr = r.get('training_params', {}).get('lr')
            if lr:
                lr_results[lr].append(r['best_val_loss'])
        
        if lr_results:
            print(f"\n🎓 Learning Rate Impact:")
            print(f"{'─'*80}")
            for lr in sorted(lr_results.keys(), 
                           key=lambda x: sum(lr_results[x])/len(lr_results[x])):
                losses = lr_results[lr]
                avg = sum(losses) / len(losses)
                print(f"   lr={lr:7.5f}     Avg Loss: {avg:.6f}  (n={len(losses)})")
    
    def print_failed_analysis(self):
        """Print analysis of failed runs"""
        failed_results = [r for r in self.results if r.get("status") == "failed"]
        
        if not failed_results:
            print("\n" + "="*80)
            print("✅ NO FAILED RUNS")
            print("="*80)
            print("\n   All experiments completed successfully! 🎉")
            return
        
        print("\n" + "="*80)
        print(f"⚠️  FAILED RUNS ANALYSIS ({len(failed_results)})")
        print("="*80)
        
        # Group by error type
        by_error = defaultdict(list)
        for r in failed_results:
            error = r.get('error', 'Unknown error')
            error_type = error.split('\n')[0][:50]  # First line, first 50 chars
            by_error[error_type].append(r)
        
        print(f"\n   Failed by error type:")
        for error_type, results in by_error.items():
            print(f"   • {error_type:50s} ({len(results)} runs)")
        
        print(f"\n   Recent failures:")
        for idx, r in enumerate(failed_results[-3:], 1):
            print(f"   {idx}. {r['model_type']} - {r['run_name']}")
            if 'error' in r:
                error_lines = r['error'].split('\n')
                print(f"      Error: {error_lines[0][:60]}")
    
    def save_summary_report(self, output_file: str = "analysis_summary.txt"):
        """Save analysis summary to text file"""
        if not self.successful_results:
            return
        
        with open(output_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPERPARAMETER TUNING ANALYSIS REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")
            
            # Best model
            best = min(self.successful_results, key=lambda x: x["best_val_loss"])
            f.write("BEST MODEL\n")
            f.write("-"*80 + "\n")
            f.write(f"Model Type:      {best['model_type']}\n")
            f.write(f"Val Loss:        {best['best_val_loss']:.6f}\n")
            f.write(f"Run ID:          {best.get('run_id', 'N/A')}\n")
            f.write(f"Run Name:        {best['run_name']}\n\n")
            
            # Configuration
            f.write("OPTIMAL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write("\nDataset Parameters:\n")
            for key, val in best['dataset_params'].items():
                f.write(f"  {key}: {val}\n")
            
            f.write("\nModel Parameters:\n")
            for key, val in best['model_params'].items():
                f.write(f"  {key}: {val}\n")
            
            f.write("\nTraining Parameters:\n")
            for key, val in best['training_params'].items():
                f.write(f"  {key}: {val}\n")
        
        print(f"\n💾 Summary report saved to: {output_file}")
    
    def save_best_configs(self, output_file: str = "best_configs.json"):
        """Save best configuration for each model type"""
        if not self.successful_results:
            return
        
        by_model = defaultdict(list)
        for r in self.successful_results:
            by_model[r['model_type']].append(r)
        
        best_configs = {}
        for model_type, results in by_model.items():
            best = min(results, key=lambda x: x['best_val_loss'])
            best_configs[model_type] = {
                "best_val_loss": best['best_val_loss'],
                "best_epoch": best['best_epoch'],
                "run_id": best.get('run_id'),
                "run_name": best['run_name'],
                "model_params": best['model_params'],
                "dataset_params": best['dataset_params'],
                "training_params": best['training_params']
            }
        
        with open(output_file, 'w') as f:
            json.dump(best_configs, f, indent=2)
        
        print(f"💾 Best configs saved to: {output_file}")
    
    def log_summary_to_mlflow(self, experiment_name: str = None):
        """Log analysis summary to MLflow"""
        if not MLFLOW_AVAILABLE or not self.successful_results:
            return
        
        try:
            # Get experiment name from results
            if experiment_name is None:
                experiment_name = "tuning_analysis"
            
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log overall metrics
                best = min(self.successful_results, key=lambda x: x["best_val_loss"])
                mlflow.log_metric("best_val_loss_overall", best['best_val_loss'])
                mlflow.log_metric("total_runs", len(self.results))
                mlflow.log_metric("successful_runs", len(self.successful_results))
                mlflow.log_metric("failed_runs", len(self.results) - len(self.successful_results))
                
                # Log best model info
                mlflow.log_param("best_model_type", best['model_type'])
                mlflow.log_param("best_run_id", best.get('run_id', 'N/A'))
                
                # Log best configs as params
                for key, val in best['dataset_params'].items():
                    mlflow.log_param(f"best_dataset_{key}", val)
                
                for key, val in best['model_params'].items():
                    mlflow.log_param(f"best_model_{key}", val)
                
                for key, val in best['training_params'].items():
                    mlflow.log_param(f"best_training_{key}", val)
                
                # Log model comparison
                by_model = defaultdict(list)
                for r in self.successful_results:
                    by_model[r['model_type']].append(r['best_val_loss'])
                
                for model_type, losses in by_model.items():
                    mlflow.log_metric(f"{model_type}_best_loss", min(losses))
                    mlflow.log_metric(f"{model_type}_avg_loss", sum(losses)/len(losses))
                    mlflow.log_metric(f"{model_type}_worst_loss", max(losses))
                    mlflow.log_metric(f"{model_type}_num_runs", len(losses))
                
                # Save and log artifacts
                self.save_summary_report("analysis_summary.txt")
                self.save_best_configs("best_configs.json")
                mlflow.log_artifact("analysis_summary.txt")
                mlflow.log_artifact("best_configs.json")
                mlflow.log_artifact(self.results_file)
                
                # Set tags
                mlflow.set_tag("analysis_type", "hyperparameter_tuning")
                mlflow.set_tag("best_model", best['model_type'])
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                print(f"\n✅ Analysis logged to MLflow experiment: {experiment_name}")
                
        except Exception as e:
            print(f"\n⚠️  Could not log to MLflow: {e}")


def main():
    """Main analysis function"""
    
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER TUNING ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ResultAnalyzer("results_final.json")
    
    # Load results
    if not analyzer.load_results():
        return
    
    # Perform analyses
    analyzer.print_executive_summary()
    analyzer.print_stage_breakdown()
    analyzer.print_model_comparison()
    analyzer.print_hyperparameter_insights()
    analyzer.print_failed_analysis()
    
    # Save outputs
    print("\n" + "="*80)
    print("💾 SAVING OUTPUTS")
    print("="*80)
    analyzer.save_summary_report()
    analyzer.save_best_configs()
    analyzer.log_summary_to_mlflow()
    
    # Next steps
    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    print("\n   1. 📄 Review analysis_summary.txt for detailed report")
    print("   2. ⚙️  Check best_configs.json for optimal configurations")
    print("   3. 🔬 View MLflow UI for interactive exploration:")
    print("      http://127.0.0.1:8080")
    print("   4. 🚀 Deploy best model to production")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()