from typing import Any, Dict, List, Optional, Union
import joblib
import pandas as pd
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    model_name: str
    version: str
    feature_columns: List[str]
    target_column: Optional[str] = None


class InferenceEngine:
    """
    A lightweight wrapper for loading ML models and performing batch/real-time predictions.
    Integrates seamlessly into the Spark ETL pipeline for downstream tasks.
    """

    def __init__(self, model_path: str, metadata: ModelMetadata):
        self.model_path = model_path
        self.metadata = metadata
        self.model = self._load_model()

    def _load_model(self) -> Any:
        """Loads a pre-trained model from a serialized file."""
        try:
            model = joblib.load(self.model_path)
            print(f"Model '{self.metadata.model_name}' (v{self.metadata.version}) loaded successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {self.model_path}: {e}")

    def predict_batch(self, input_data: Union[pd.DataFrame, List[Dict[str, Any]]]) -> pd.Series:
        """Performs batch predictions on the provided input data."""
        if isinstance(input_data, list):
            input_data = pd.DataFrame(input_data)
        
        # Ensure all required feature columns are present
        missing_features = set(self.metadata.feature_columns) - set(input_data.columns)
        if missing_features:
            raise ValueError(f"Input data is missing required features: {missing_features}")
        
        X = input_data[self.metadata.feature_columns]
        predictions = self.model.predict(X)
        
        return pd.Series(predictions, name="prediction")

    def predict_real_time(self, single_record: Dict[str, Any]) -> Any:
        """Performs a single prediction for real-time inference scenarios."""
        # Convert record to the expected input format (e.g., list of records for DataFrame)
        input_df = pd.DataFrame([single_record])
        return self.predict_batch(input_df).iloc[0]


if __name__ == "__main__":
    # Example usage (simulated)
    # metadata = ModelMetadata(
    #     model_name="churn_predictor", 
    #     version="1.0.0", 
    #     feature_columns=["age", "tenure", "usage_score"]
    # )
    # engine = InferenceEngine(model_path="models/churn_model.pkl", metadata=metadata)
    # sample_data = pd.DataFrame({"age": [30], "tenure": [12], "usage_score": [0.85]})
    # predictions = engine.predict_batch(sample_data)
    # print(predictions)
    pass
