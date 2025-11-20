"""Print methods for result analysis"""

from collections import defaultdict


class ResultPrinters:
    """Print analysis results in formatted way"""
    
    @staticmethod
    def print_executive_summary(analyzer):
        """Print executive summary with key findings"""
        if not analyzer.successful_results:
            print("\n⚠️  No successful results to analyze")
            return
        
        print("\n" + "="*80)
        print("📈 EXECUTIVE SUMMARY")
        print("="*80)
        
        best = analyzer.get_best_overall()
        
        print(f"\n🏆 BEST OVERALL MODEL")
        print(f"{'─'*80}")
        print(f"   Model:           {best['model_type']}")
        print(f"   Val Loss:        {best['best_val_loss']:.6f}")
        print(f"   Run ID:          {best.get('run_id', 'N/A')}")
        
        # Final models comparison
        by_stage = analyzer.get_results_by_stage()
        if 'final' in by_stage:
            print(f"\n🎯 FINAL MODELS COMPARISON")
            print(f"{'─'*80}")
            final_sorted = sorted(by_stage['final'], key=lambda x: x['best_val_loss'])
            for idx, result in enumerate(final_sorted, 1):
                status = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}."
                print(f"   {status} {result['model_type']:12s}  Val Loss: {result['best_val_loss']:.6f}")
            
            # Best configuration
            best_final = final_sorted[0]
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
    
    @staticmethod
    def print_stage_breakdown(analyzer):
        """Print breakdown by tuning stage"""
        if not analyzer.successful_results:
            return
        
        print("\n" + "="*80)
        print("🔍 STAGE-BY-STAGE BREAKDOWN")
        print("="*80)
        
        stage_order = ['stage1_scaler', 'stage2_seq_len', 'stage3', 'stage4', 'final']
        stage_names = {
            'stage1_scaler': '1️⃣  Stage 1: Scaler Selection',
            'stage2_seq_len': '2️⃣  Stage 2: Sequence Length',
            'stage3': '3️⃣  Stage 3: Model Architecture',
            'stage4': '4️⃣  Stage 4: Training Optimization',
            'final': '🎯 Final: Production Models'
        }
        
        for stage_key in stage_order:
            results = [r for r in analyzer.successful_results 
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
    
    @staticmethod
    def print_model_comparison(analyzer):
        """Print detailed model comparison"""
        if not analyzer.successful_results:
            return
        
        print("\n" + "="*80)
        print("📊 MODEL PERFORMANCE COMPARISON")
        print("="*80)
        
        by_model = analyzer.get_results_by_model()
        
        print(f"\n{'Model':<15} {'Runs':>6} {'Best Loss':>12} {'Avg Loss':>12} {'Worst Loss':>12}")
        print(f"{'─'*15} {'─'*6:>6} {'─'*12:>12} {'─'*12:>12} {'─'*12:>12}")
        
        for model_type in sorted(by_model.keys()):
            results = by_model[model_type]
            losses = [r['best_val_loss'] for r in results]
            
            best_loss = min(losses)
            avg_loss = sum(losses) / len(losses)
            worst_loss = max(losses)
            
            print(f"{model_type:<15} {len(results):>6} {best_loss:>12.6f} {avg_loss:>12.6f} {worst_loss:>12.6f}")
    
    @staticmethod
    def print_hyperparameter_insights(analyzer):
        """Print insights about hyperparameter impact"""
        if not analyzer.successful_results:
            return
        
        print("\n" + "="*80)
        print("💡 HYPERPARAMETER INSIGHTS")
        print("="*80)
        
        # Scaler impact
        scaler_results = analyzer.get_param_impact('dataset_params.scaler')
        if scaler_results:
            print(f"\n📏 Scaler Impact:")
            print(f"{'─'*80}")
            for scaler in sorted(scaler_results.keys(), 
                               key=lambda x: sum(scaler_results[x])/len(scaler_results[x])):
                losses = scaler_results[scaler]
                avg = sum(losses) / len(losses)
                print(f"   {scaler:12s}  Avg Loss: {avg:.6f}  (n={len(losses)})")
        
        # Sequence length impact
        seq_results = analyzer.get_param_impact('dataset_params.seq_len')
        if seq_results:
            print(f"\n📊 Sequence Length Impact:")
            print(f"{'─'*80}")
            for seq_len in sorted(seq_results.keys(), 
                                key=lambda x: sum(seq_results[x])/len(seq_results[x])):
                losses = seq_results[seq_len]
                avg = sum(losses) / len(losses)
                print(f"   seq_len={seq_len:3d}   Avg Loss: {avg:.6f}  (n={len(losses)})")
        
        # Learning rate impact
        lr_results = analyzer.get_param_impact('training_params.lr')
        if lr_results:
            print(f"\n🎓 Learning Rate Impact:")
            print(f"{'─'*80}")
            for lr in sorted(lr_results.keys(), 
                           key=lambda x: sum(lr_results[x])/len(lr_results[x])):
                losses = lr_results[lr]
                avg = sum(losses) / len(losses)
                print(f"   lr={lr:7.5f}     Avg Loss: {avg:.6f}  (n={len(losses)})")
    
    @staticmethod
    def print_failed_analysis(analyzer):
        """Print analysis of failed runs"""
        failed_results = analyzer.get_failed_results()
        
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
            error_type = error.split('\n')[0][:50]
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
