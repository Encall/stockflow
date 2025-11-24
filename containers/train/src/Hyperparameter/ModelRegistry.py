"""
Model Registry utilities for MLflow
"""
from typing import Dict, Optional
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def register_model(
    run_id: str,
    model_name: str,
    stock_symbol: str,
    model_type: str,
    val_loss: float,
    dataset_params: Dict,
    model_params: Dict,
    training_params: Dict,
    is_best: bool = False
) -> Optional[Dict]:
    """
    Register model to MLflow Model Registry with tags and aliases
    
    Args:
        run_id: MLflow run ID
        model_name: Name for the registered model
        stock_symbol: Stock symbol
        model_type: Model type (LSTM, GRU, etc)
        val_loss: Validation loss
        dataset_params: Dataset configuration
        model_params: Model configuration
        training_params: Training configuration
        is_best: Whether this is the best model
        
    Returns:
        Dict with registration info or None if failed
    """
    if not MLFLOW_AVAILABLE:
        print("⚠️  MLflow not available, skipping registration")
        return None
    
    try:
        # Register model
        print(f"   📝 Registering model: {model_name}")
        model_uri = f"runs:/{run_id}/model"
        
        # Tags for the model version
        version_tags = {
            "stock_symbol": stock_symbol,
            "model_type": model_type,
            "val_loss": f"{val_loss:.6f}",
            "registered_date": datetime.now().isoformat(),
            "scaler": dataset_params.get('scaler', 'unknown'),
            "seq_len": str(dataset_params.get('seq_len', 0))
        }
        
        # Add "best" tag if this is the best model
        if is_best:
            version_tags["best"] = "true"
        
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name,
            tags=version_tags
        )
        
        client = MlflowClient()
        
        # Add description
        description = f"{'🏆 BEST ' if is_best else ''}{model_type} model for {stock_symbol} stock. " \
                     f"Validation Loss: {val_loss:.6f}. " \
                     f"Dataset: {dataset_params}"
        
        client.update_model_version(
            name=model_name,
            version=model_version.version,
            description=description
        )
        
        # Set alias based on whether it's best or not
        if is_best:
            # Best model goes to Production with "best" alias
            client.set_registered_model_alias(
                name=model_name,
                alias="best",
                version=model_version.version
            )
            client.set_registered_model_alias(
                name=model_name,
                alias="production",
                version=model_version.version
            )
            print(f"   ✅ Model registered as BEST: {model_name} v{model_version.version}")
            print(f"   📌 Aliases: production, best")
            stage = "Production"
        else:
            # Other models go to Staging with "candidate" alias
            client.set_registered_model_alias(
                name=model_name,
                alias="candidate",
                version=model_version.version
            )
            print(f"   ✅ Model registered: {model_name} v{model_version.version}")
            print(f"   📌 Alias: candidate")
            stage = "Staging"
        
        return {
            "model_name": model_name,
            "version": model_version.version,
            "run_id": run_id,
            "val_loss": val_loss,
            "is_best": is_best,
            "stage": stage,
            "aliases": ["production", "best"] if is_best else ["candidate"]
        }
        
    except Exception as e:
        print(f"   ❌ Error registering model: {str(e)}")
        import traceback
        traceback.print_exc()
        return None