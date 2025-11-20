#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results with beautiful formatting
Refactored to use modular components
"""

from .ResultAnalyzer import ResultAnalyzer
from .ResultPrinters import ResultPrinters
from .ResultSavers import ResultSavers


def main(results_file: str = "results_final.json"):
    """Main analysis function - orchestrates analysis, printing, and saving"""
    
    print("\n" + "="*80)
    print("🔬 HYPERPARAMETER TUNING ANALYSIS")
    print("="*80)
    
    # Initialize analyzer
    analyzer = ResultAnalyzer(results_file)
    
    # Load results
    if not analyzer.load_results():
        return
    
    # Print analyses
    ResultPrinters.print_executive_summary(analyzer)
    ResultPrinters.print_stage_breakdown(analyzer)
    ResultPrinters.print_model_comparison(analyzer)
    ResultPrinters.print_hyperparameter_insights(analyzer)
    ResultPrinters.print_failed_analysis(analyzer)
    
    # Save outputs
    print("\n" + "="*80)
    print("💾 SAVING OUTPUTS")
    print("="*80)
    
    ResultSavers.save_summary_report(analyzer)
    ResultSavers.save_best_configs(analyzer)
    ResultSavers.save_stage_configs(analyzer)
    ResultSavers.log_summary_to_mlflow(analyzer)
    
    # Next steps
    print("\n" + "="*80)
    print("📋 NEXT STEPS")
    print("="*80)
    print("\n   1. 📄 Review analysis_summary.txt for detailed report")
    print("   2. ⚙️  Check best_configs.json for optimal configurations")
    print("   3. 📊 See stage_configs.json for stage-by-stage analysis")
    print("   4. 🔬 View MLflow UI for interactive exploration:")
    print("      http://127.0.0.1:8080")
    print("   5. 🚀 Deploy best model to production")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()
