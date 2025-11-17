#!/usr/bin/env python3
"""
Quick test script for hyperparameter tuning system
Tests a single LSTM configuration to verify everything works
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from containers.train.src.Hyperparameter.HyperparameterTuning import HyperparameterTuner
from src.GetDummies import get_dummy

def quick_test():
    """Run a quick test with minimal configuration"""
    
    print("="*80)
    print("QUICK TEST - Hyperparameter Tuning System")
    print("="*80)
    
    # Generate small test data
    print("\n1. Generating test data...")
    data = get_dummy(
        spec={
            "Open": "float",
            "High": "float",
            "Low": "float",
            "Close": "float",
            "Volume": "int"
        },
        n_rows=500  # Small dataset for quick test
    )
    print(f"   ✓ Data shape: {data.shape}")
    
    feature_cols = ["Open", "High", "Low", "Volume"]
    target_col = "Close"
    
    # Initialize tuner (without MLflow)
    print("\n2. Initializing tuner...")
    tuner = HyperparameterTuner(use_mlflow=False)
    print("   ✓ Tuner initialized")
    
    # Test with single configuration
    print("\n3. Testing single configuration...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   ✓ Using device: {device}")
    

    
    test_config = {
        "model_params": {
            "hidden_size": 64,
            "num_layers": 2,
            "dropout": 0.1,
            "output_size": 1
        },
        "dataset_params": {
            "seq_len": 30,
            "scaler": "minmax"
        },
        "training_params": {
            "lr": 0.001,
            "epochs": 5,  # Just 5 epochs for quick test
            "batch_size": 32,
            "loss_fn": "MSE"
        }
    }
    
    print("\n   Configuration:")
    for key, params in test_config.items():
        print(f"   - {key}: {params}")
    
    print("\n4. Training model...")
    result = tuner.train_with_config(
        model_type="LSTM",
        model_params=test_config["model_params"],
        dataset_params=test_config["dataset_params"],
        training_params=test_config["training_params"],
        data=data,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device
    )
    
    # Check result
    print("\n5. Results:")
    if result["status"] == "success":
        print("   ✅ Test PASSED!")
        print(f"   - Model Type: {result['model_type']}")
        print(f"   - Best Val Loss: {result['best_val_loss']:.6f}")
        print(f"   - Best Epoch: {result['best_epoch']}")
        print(f"   - Run Name: {result['run_name']}")
        
        # Save test result
        import json
        with open("test_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        print("\n   ✓ Result saved to test_result.json")
        
        return True
    else:
        print("   ❌ Test FAILED!")
        print(f"   - Error: {result.get('error', 'Unknown error')}")
        return False

def main():
    try:
        success = quick_test()
        
        print("\n" + "="*80)
        if success:
            print("✅ QUICK TEST COMPLETED SUCCESSFULLY")
            print("\nNext steps:")
            print("  1. Review test_result.json")
            print("  2. Run full tuning: python run_tuning.py")
            print("  3. Analyze results: python analyze.py")
        else:
            print("❌ QUICK TEST FAILED")
            print("\nCheck the error messages above and fix the issues")
        print("="*80)
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
