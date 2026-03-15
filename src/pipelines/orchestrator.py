import argparse
import logging
from typing import Dict, Any
from etl.pyspark_processor import SparkETLProcessor, ETLConfig
from models.inference_engine import InferenceEngine, ModelMetadata


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Manages the end-to-end data pipeline workflow.
    Workflow: Extract -> Transform -> Load -> ML Inference.
    """

    def __init__(self, etl_config: ETLConfig, model_metadata: ModelMetadata, model_path: str):
        self.etl_processor = SparkETLProcessor(etl_config)
        self.inference_engine = InferenceEngine(model_path, model_metadata)

    def run_pipeline(self, input_path: str, output_path: str):
        """Executes the full pipeline workflow."""
        try:
            logger.info("Starting Pipeline Execution...")

            # Step 1: Extract & Transform
            raw_df = self.etl_processor.read_data(input_path)
            transformed_df = self.etl_processor.transform(raw_df)

            # Step 2: Load (Intermediate storage)
            self.etl_processor.write_data(transformed_df, output_path)
            logger.info(f"ETL Step Complete. Transformed data saved to {output_path}")

            # Step 3: ML Inference (Batch)
            # For demonstration, we convert a sample of Spark DF to Pandas
            # In a real Spark environment, you would use a Spark-native inference (e.g., Pandas UDF or Spark ML)
            pandas_subset = transformed_df.limit(1000).toPandas()
            predictions = self.inference_engine.predict_batch(pandas_subset)
            
            logger.info(f"ML Inference Step Complete. Sample predictions: {predictions.head().tolist()}")

            logger.info("Pipeline Execution Finished Successfully.")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        finally:
            self.etl_processor.stop()


def main():
    parser = argparse.ArgumentParser(description="Scalable Data & AI Orchestrator")
    parser.add_argument("--input", required=True, help="Path to input data")
    parser.add_argument("--output", required=True, help="Path to output data")
    parser.add_argument("--model", required=True, help="Path to ML model file")
    
    args = parser.parse_args()

    # Define configurations (typically loaded from YAML/JSON files)
    etl_cfg = ETLConfig(app_name="Production_Pipeline", master="local[*]")
    
    # Mock model metadata for demonstration
    model_meta = ModelMetadata(
        model_name="production_model",
        version="1.2.0",
        feature_columns=["feature1", "feature2", "feature3"]
    )

    orchestrator = PipelineOrchestrator(
        etl_config=etl_cfg,
        model_metadata=model_meta,
        model_path=args.model
    )

    orchestrator.run_pipeline(args.input, args.output)


if __name__ == "__main__":
    main()
