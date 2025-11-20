"""Save analysis results to files and MLflow"""

import json
import mlflow
from datetime import datetime


class ResultSavers:
    """Save analysis results to various outputs"""
    
    @staticmethod
    def save_summary_report(analyzer, output_file='analysis_summary.txt'):
        """Save text summary report"""
        from io import StringIO
        import sys
        from .ResultPrinters import ResultPrinters
        
        # Capture print output
        old_stdout = sys.stdout
        sys.stdout = buffer = StringIO()
        
        # Print all sections
        ResultPrinters.print_executive_summary(analyzer)
        ResultPrinters.print_stage_breakdown(analyzer)
        ResultPrinters.print_model_comparison(analyzer)
        ResultPrinters.print_hyperparameter_insights(analyzer)
        ResultPrinters.print_failed_analysis(analyzer)
        
        # Get output
        sys.stdout = old_stdout
        summary_text = buffer.getvalue()
        
        # Save to file with UTF-8 encoding
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)
        
        print(f"\n📄 Summary report saved to: {output_file}")
        return output_file
    
    @staticmethod
    def save_best_configs(analyzer, output_file='best_configs.json'):
        """Save best configurations in structured JSON"""
        if not analyzer.successful_results:
            return None
        
        best_overall = analyzer.get_best_overall()
        best_by_model = analyzer.get_best_by_model()
        
        configs = {
            'timestamp': datetime.now().isoformat(),
            'best_overall': {
                'model_type': best_overall['model_type'],
                'val_loss': best_overall['best_val_loss'],
                'run_id': best_overall.get('run_id', 'N/A'),
                'config': {
                    'dataset_params': best_overall['dataset_params'],
                    'model_params': best_overall['model_params'],
                    'training_params': best_overall['training_params']
                }
            },
            'best_by_model': {}
        }
        
        for model_type, result in best_by_model.items():
            configs['best_by_model'][model_type] = {
                'val_loss': result['best_val_loss'],
                'run_id': result.get('run_id', 'N/A'),
                'config': {
                    'dataset_params': result['dataset_params'],
                    'model_params': result['model_params'],
                    'training_params': result['training_params']
                }
            }
        
        with open(output_file, 'w') as f:
            json.dump(configs, f, indent=2)
        
        print(f"💾 Best configurations saved to: {output_file}")
        return output_file
    
    @staticmethod
    def save_stage_configs(analyzer, output_file='stage_configs.json'):
        """Save best config per stage for staged tuning analysis"""
        if not analyzer.successful_results:
            return None
        
        stage_configs = {
            'timestamp': datetime.now().isoformat(),
            'stages': {}
        }
        
        stage_order = ['stage1_scaler', 'stage2_seq_len', 'stage3', 'stage4', 'final']
        stage_names = {
            'stage1_scaler': 'Stage 1: Scaler Selection',
            'stage2_seq_len': 'Stage 2: Sequence Length',
            'stage3': 'Stage 3: Model Architecture',
            'stage4': 'Stage 4: Training Optimization',
            'final': 'Final: Production Models'
        }
        
        for stage_key in stage_order:
            results = [r for r in analyzer.successful_results 
                      if r.get('tuning_stage', '').startswith(stage_key.replace('stage3', 'stage3_'))]
            
            if results:
                best = min(results, key=lambda x: x['best_val_loss'])
                
                stage_configs['stages'][stage_key] = {
                    'name': stage_names.get(stage_key, stage_key),
                    'num_experiments': len(results),
                    'best_result': {
                        'model_type': best['model_type'],
                        'val_loss': best['best_val_loss'],
                        'run_id': best.get('run_id', 'N/A'),
                        'config': {
                            'dataset_params': best['dataset_params'],
                            'model_params': best['model_params'],
                            'training_params': best['training_params']
                        }
                    }
                }
                
                # Add stage-specific findings
                if stage_key == 'stage1_scaler':
                    stage_configs['stages'][stage_key]['finding'] = {
                        'best_scaler': best['dataset_params']['scaler']
                    }
                elif stage_key == 'stage2_seq_len':
                    stage_configs['stages'][stage_key]['finding'] = {
                        'best_seq_len': best['dataset_params']['seq_len']
                    }
                elif stage_key == 'stage3':
                    stage_configs['stages'][stage_key]['finding'] = {
                        'best_model': best['model_type']
                    }
        
        with open(output_file, 'w') as f:
            json.dump(stage_configs, f, indent=2)
        
        print(f"📊 Stage configurations saved to: {output_file}")
        return output_file
    
    @staticmethod
    def log_summary_to_mlflow(analyzer, experiment_name='hyperparameter_tuning'):
        """Log summary metrics to MLflow"""
        if not analyzer.successful_results:
            print("⚠️  No results to log to MLflow")
            return None
        
        try:
            mlflow.set_experiment(experiment_name)
            
            with mlflow.start_run(run_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Overall best
                best = analyzer.get_best_overall()
                mlflow.log_metric('best_overall_val_loss', best['best_val_loss'])
                mlflow.log_param('best_model_type', best['model_type'])
                
                # Best per model
                best_by_model = analyzer.get_best_by_model()
                for model_type, result in best_by_model.items():
                    mlflow.log_metric(f'best_{model_type}_val_loss', result['best_val_loss'])
                
                # Stage metrics
                by_stage = analyzer.get_results_by_stage()
                for stage, results in by_stage.items():
                    best_stage = min(results, key=lambda x: x['best_val_loss'])
                    mlflow.log_metric(f'{stage}_best_loss', best_stage['best_val_loss'])
                    mlflow.log_metric(f'{stage}_num_experiments', len(results))
                
                # Overall statistics
                all_losses = [r['best_val_loss'] for r in analyzer.successful_results]
                mlflow.log_metric('avg_val_loss', sum(all_losses) / len(all_losses))
                mlflow.log_metric('total_successful_runs', len(analyzer.successful_results))
                mlflow.log_metric('total_failed_runs', len(analyzer.get_failed_results()))
                
                # Log best config as artifact
                best_config = {
                    'model_type': best['model_type'],
                    'dataset_params': best['dataset_params'],
                    'model_params': best['model_params'],
                    'training_params': best['training_params']
                }
                
                with open('temp_best_config.json', 'w') as f:
                    json.dump(best_config, f, indent=2)
                
                mlflow.log_artifact('temp_best_config.json')
                
                print(f"\n✅ Summary logged to MLflow experiment: {experiment_name}")
                return mlflow.active_run().info.run_id
                
        except Exception as e:
            print(f"\n⚠️  Error logging to MLflow: {e}")
            return None
