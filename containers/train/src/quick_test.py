#!/usr/bin/env python3
"""
Quick test script for hyperparameter tuning system
Tests basic functionality before running full tuning
"""

import sys
import json

import torch
from Hyperparameter.HyperparameterTuner import HyperparameterTuner
from Hyperparameter.StagedTuning import StagedTuning
from Hyperparameter.HyperparameterConfig import DATASET_PARAMS
import DataLoader


def quick_test():
    """Test single model configuration (fastest)"""
    
    print("="*80)
    print("QUICK TEST - Single Configuration")
    print("="*80)
    
    # Load real data
    print("\n1. Loading data from MinIO...")
    data_loader = DataLoader.DataLoader("DIG")
    data = data_loader.get_data()
    
    if data is None:
        print("   ❌ Failed to load data")
        return False
    
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
        feature_cols=["open", "high", "low", "volume"],
        target_col="close",
        device=device,
        stage="test"
    )
    
    # Check result
    print("\n4. Results:")
    if result["status"] == "success":
        print("   ✅ Test PASSED!")
        print(f"   - Val Loss: {result['best_val_loss']:.6f}")
        print(f"   - Best Epoch: {result['best_epoch']}")
        
        # Save result
        with open("test_quick.json", 'w') as f:
            json.dump([result], f, indent=2)
        print("\n   💾 Saved to: test_quick.json")
        
        return True
    else:
        print("   ❌ Test FAILED!")
        print(f"   - Error: {result.get('error', 'Unknown error')}")
        return False


def test_stage1():
    """Test Stage 1 with 2 scalers only"""
    
    print("="*80)
    print("STAGE 1 TEST - Scaler Selection (2 scalers)")
    print("="*80)
    
    # Load real data
    print("\n1. Loading data from MinIO...")
    data_loader = DataLoader.DataLoader("DIG")
    data = data_loader.get_data()
    
    if data is None:
        print("   ❌ Failed to load data")
        return False
    
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
            ["open", "high", "low", "volume"], 
            "close", 
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
    
    # Load real data
    print("\n1. Loading data from MinIO...")
    data_loader = DataLoader.DataLoader("DIG")
    data = data_loader.get_data()
    
    if data is None:
        print("   ❌ Failed to load data")
        return False
    
    print(f"   ✓ Data shape: {data.shape}")
    
    # Initialize
    print("\n2. Initializing staged tuner (no MLflow)...")
    tuner = HyperparameterTuner(use_mlflow=False)
    staged = StagedTuning(tuner)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_cols = ["open", "high", "low", "volume"]
    target_col = "close"
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
    
    # Check if real data is available
    print("\n🔍 Checking data availability...")
    data_loader = DataLoader.DataLoader("DIG")
    data = data_loader.get_data()
    
    if data is None:
        print("❌ Cannot load data from MinIO.")
        print("\nPlease ensure:")
        print("  1. MinIO is running and accessible")
        print("  2. Gold layer data exists in MinIO")
        print("  3. .env file has correct MinIO credentials")
        return 1
    
    print(f"✅ Data available: {data.shape}")
    print(f"   Columns: {list(data.columns)}")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        print_menu()
        choice = input("\nEnter test number (1-4): ").strip()
    
    results = []
    
    if choice == "1":
        print("\n▶️  Running Quick Test...")
        results.append(("Quick Test", quick_test()))
        
    elif choice == "2":
        print("\n▶️  Running Stage 1 Test...")
        results.append(("Stage 1", test_stage1()))
        
    elif choice == "3":
        print("\n▶️  Running Mini Staged Test...")
        results.append(("Mini Staged", test_mini_staged()))
        
    elif choice == "4":
        print("\n▶️  Running All Tests...")
        results.append(("Quick Test", quick_test()))
        results.append(("Stage 1", test_stage1()))
        results.append(("Mini Staged", test_mini_staged()))
        
    else:
        print(f"\n❌ Invalid choice: {choice}")
        return 1
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n✅ All tests passed!")
        print("\nYou can now run the full hyperparameter tuning:")
        print("  1. Set MLflow URI in environment:")
        print("     export MLFLOW_TRACKING_URI=http://localhost:5000")
        print("  2. Run full tuning:")
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
        print("  4. Verify MinIO connection and data availability")
    
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())