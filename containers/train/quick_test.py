#!/usr/bin/env python3
"""
Quick test script for hyperparameter tuning system
Tests basic functionality before running full tuning
"""

import sys
import json

import torch
from src.Hyperparameter.HyperparameterTuner import HyperparameterTuner
from src.Hyperparameter.StagedTuning import StagedTuning
from src.Hyperparameter.HyperparameterConfig import DATASET_PARAMS
import src.GetDummies


def quick_test():
    """Test single model configuration (fastest)"""
    
    print("="*80)
    print("QUICK TEST - Single Configuration")
    print("="*80)
    
    # Generate test data
    print("\n1. Generating test data (500 rows)...")
    data = src.GetDummies.get_dummy(
        spec={"Open": "float", "High": "float", "Low": "float", "Close": "float", "Volume": "int"},
        n_rows=500
    )
    print(f"   ✓ Data shape: {data.shape}")
    
    # Initialize tuner
    print("\n2. Initializing tuner (no MLflow)...")
    tuner = HyperparameterTuner(use_mlflow=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Device: {device}")
    
    # Simple config
    config = {
        "model_params": {"hidden_size": 64, "num_layers": 2, "dropout": 0.1, "output_size": 1},
        "dataset_params": {"seq_len": 30, "scaler": "minmax"},
        "training_params": {"lr": 0.001, "epochs": 5, "batch_size": 32, "loss_fn": "MSE"}
    }
    
    print("\n3. Training LSTM (5 epochs)...")
    result = tuner.train_with_config(
        model_type="LSTM",
        model_params=config["model_params"],
        dataset_params=config["dataset_params"],
        training_params=config["training_params"],
        data=data,
        feature_cols=["Open", "High", "Low", "Volume"],
        target_col="Close",
        device=device,
        stage="test"
    )
    
    # Check result
    print("\n4. Results:")
    if result["status"] == "success":
        print("   ✅ Test PASSED!")
        print(f"   - Val Loss: {result['best_val_loss']:.6f}")
        print(f"   - Best Epoch: {result['best_epoch']}")
        
        with open("test_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        print("\n   💾 Saved to: test_result.json")
        return True
    else:
        print("   ❌ Test FAILED!")
        print(f"   - Error: {result.get('error', 'Unknown')}")
        return False


def test_stage1():
    """Test Stage 1 with 2 scalers only"""
    
    print("="*80)
    print("STAGE 1 TEST - Scaler Selection (2 scalers)")
    print("="*80)
    
    # Generate test data
    print("\n1. Generating test data (500 rows)...")
    data = src.GetDummies.get_dummy(
        spec={"Open": "float", "High": "float", "Low": "float", "Close": "float", "Volume": "int"},
        n_rows=500
    )
    print(f"   ✓ Data shape: {data.shape}")
    
    # Initialize
    print("\n2. Initializing staged tuner (no MLflow)...")
    tuner = HyperparameterTuner(use_mlflow=False)
    staged = StagedTuning(tuner)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Device: {device}")
    
    # Limit scalers for quick test
    print("\n3. Testing 2 scalers (minmax, standard)...")
    original_scalers = DATASET_PARAMS["scaler"]
    DATASET_PARAMS["scaler"] = ["minmax", "standard"]
    
    try:
        best_scaler, results = staged.stage1_find_best_scaler(
            data, 
            ["Open", "High", "Low", "Volume"], 
            "Close", 
            device
        )
        
        # Restore
        DATASET_PARAMS["scaler"] = original_scalers
        
        # Show results
        print("\n4. Results:")
        successful = [r for r in results if r["status"] == "success"]
        print(f"   Total runs: {len(results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   ✅ Best: {best_scaler}")
        
        for r in results:
            status = "✅" if r["status"] == "success" else "❌"
            scaler_name = r["dataset_params"]["scaler"]
            if r["status"] == "success":
                print(f"   {status} {scaler_name:10s} → Val Loss: {r['best_val_loss']:.6f}")
            else:
                print(f"   {status} {scaler_name:10s} → FAILED")
        
        # Save
        with open("test_stage1.json", 'w') as f:
            json.dump(results, f, indent=2)
        print("\n   💾 Saved to: test_stage1.json")
        
        return len(successful) > 0
        
    except Exception as e:
        DATASET_PARAMS["scaler"] = original_scalers
        print(f"\n   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mini_staged():
    """Test mini staged pipeline (Stage 1 + Stage 2 only)"""
    
    print("="*80)
    print("MINI STAGED TEST - Stage 1 & 2 (Limited)")
    print("="*80)
    
    # Generate test data
    print("\n1. Generating test data (500 rows)...")
    data = src.GetDummies.get_dummy(
        spec={"Open": "float", "High": "float", "Low": "float", "Close": "float", "Volume": "int"},
        n_rows=500
    )
    print(f"   ✓ Data shape: {data.shape}")
    
    # Initialize
    print("\n2. Initializing staged tuner (no MLflow)...")
    tuner = HyperparameterTuner(use_mlflow=False)
    staged = StagedTuning(tuner)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = "Close"
    print(f"   ✓ Device: {device}")
    
    # Limit options for quick test
    original_scalers = DATASET_PARAMS["scaler"]
    original_seq_lens = DATASET_PARAMS["seq_len"]
    DATASET_PARAMS["scaler"] = ["minmax", "standard"]
    DATASET_PARAMS["seq_len"] = [30, 50, 60]
    
    all_results = []
    
    try:
        # Stage 1
        print("\n3. Stage 1: Finding best scaler (2 options)...")
        best_scaler, stage1_results = staged.stage1_find_best_scaler(
            data, feature_cols, target_col, device
        )
        all_results.extend(stage1_results)
        print(f"   ✅ Best scaler: {best_scaler}")
        
        # Stage 2
        print("\n4. Stage 2: Finding best seq_len (3 options)...")
        best_seq_len, stage2_results = staged.stage2_find_best_seq_len(
            best_scaler, data, feature_cols, target_col, device
        )
        all_results.extend(stage2_results)
        print(f"   ✅ Best seq_len: {best_seq_len}")
        
        # Restore
        DATASET_PARAMS["scaler"] = original_scalers
        DATASET_PARAMS["seq_len"] = original_seq_lens
        
        # Summary
        print("\n5. Summary:")
        successful = [r for r in all_results if r["status"] == "success"]
        print(f"   Total runs: {len(all_results)}")
        print(f"   Successful: {len(successful)}")
        print(f"   Best config: scaler={best_scaler}, seq_len={best_seq_len}")
        
        # Save
        with open("test_mini_staged.json", 'w') as f:
            json.dump(all_results, f, indent=2)
        print("\n   💾 Saved to: test_mini_staged.json")
        
        return len(successful) > 0
        
    except Exception as e:
        DATASET_PARAMS["scaler"] = original_scalers
        DATASET_PARAMS["seq_len"] = original_seq_lens
        print(f"\n   ❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_menu():
    """Print test menu"""
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING - QUICK TEST SUITE")
    print("="*80)
    print("\nTest Options:")
    print("  1. Quick Test      - Single LSTM config (fastest, ~1-2 min)")
    print("  2. Stage 1 Test    - Test scaler selection (2 scalers, ~3-5 min)")
    print("  3. Mini Staged     - Stage 1 & 2 limited (5 runs, ~8-10 min)")
    print("  4. All Tests       - Run all above tests")
    print("="*80)


def main():
    """Main test runner"""
    
    print_menu()
    
    # Get choice
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\nSelect test (1/2/3/4) [default: 1]: ").strip() or "1"
    
    try:
        results = []
        
        # Run selected tests
        if choice in ["1", "4"]:
            print("\n" + "="*80)
            print("RUNNING: Quick Test")
            print("="*80)
            results.append(("Quick Test", quick_test()))
        
        if choice in ["2", "4"]:
            print("\n" + "="*80)
            print("RUNNING: Stage 1 Test")
            print("="*80)
            results.append(("Stage 1 Test", test_stage1()))
        
        if choice in ["3", "4"]:
            print("\n" + "="*80)
            print("RUNNING: Mini Staged Test")
            print("="*80)
            results.append(("Mini Staged", test_mini_staged()))
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        all_passed = all(result[1] for result in results)
        
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {status:12s} - {test_name}")
        
        if all_passed:
            print("\n🎉 All tests completed successfully!")
            print("\n📁 Generated files:")
            if choice in ["1", "4"]:
                print("  • test_result.json")
            if choice in ["2", "4"]:
                print("  • test_stage1.json")
            if choice in ["3", "4"]:
                print("  • test_mini_staged.json")
            
            print("\n🚀 Next steps:")
            print("  1. Review test result files")
            print("  2. Run full staged tuning:")
            print("     python run_tuning.py")
            print("  3. Analyze results:")
            print("     python analyze.py")
            
            print("\n📊 Full tuning will perform:")
            print("  • Stage 1: Best scaler (5 runs)")
            print("  • Stage 2: Best seq_len (5 runs)")
            print("  • Stage 3: Best model params (~20-30 runs/model)")
            print("  • Stage 4: Best training params (~15 runs)")
            print("  • Final: All models with best configs (4 runs)")
            print("  • Total: ~50-60 runs (vs 384,000 with full grid!)")
        else:
            print("\n⚠️  Some tests failed")
            print("\nTroubleshooting:")
            print("  1. Check dependencies are installed")
            print("  2. Verify model files exist in src/model/")
            print("  3. Ensure PyTorch and sklearn are working")
        
        print("="*80)
        
        return 0 if all_passed else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())