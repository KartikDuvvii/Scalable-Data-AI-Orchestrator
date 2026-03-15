from typing import Any, Dict, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, lit, current_timestamp
from pyspark.sql.types import StructType
from pydantic import BaseModel, Field


class ETLConfig(BaseModel):
    app_name: str = Field(default="SparkETLProcessor")
    master: str = Field(default="local[*]")
    schema_validation: bool = Field(default=True)


class SparkETLProcessor:
    """
    Modular PySpark ETL processor handling data ingestion, schema validation, and transformations.
    Designed for scalability and maintainability.
    """

    def __init__(self, config: ETLConfig):
        self.config = config
        self.spark = (
            SparkSession.builder.appName(self.config.app_name)
            .master(self.config.master)
            .getOrCreate()
        )

    def read_data(self, source_path: str, format: str = "parquet", schema: Optional[StructType] = None) -> DataFrame:
        """Reads data from the specified source with optional schema enforcement."""
        reader = self.spark.read.format(format)
        if schema:
            reader = reader.schema(schema)
        
        df = reader.load(source_path)
        print(f"Data ingested from {source_path}. Rows count: {df.count()}")
        return df

    def validate_schema(self, df: DataFrame, expected_schema: StructType) -> bool:
        """Validates the schema of the provided DataFrame against an expected schema."""
        if not self.config.schema_validation:
            return True
        
        # Simple schema comparison logic
        return df.schema.simpleString() == expected_schema.simpleString()

    def transform(self, df: DataFrame) -> DataFrame:
        """Applies core transformations to the DataFrame."""
        transformed_df = df.withColumn("processed_at", current_timestamp()) \
                           .withColumn("is_processed", lit(True))
        
        # Add custom transformations here
        # e.g., cleaning, feature engineering
        
        return transformed_df

    def write_data(self, df: DataFrame, output_path: str, format: str = "parquet", mode: str = "overwrite") -> None:
        """Writes the processed DataFrame to the destination."""
        df.write.format(format).mode(mode).save(output_path)
        print(f"Data successfully written to {output_path}.")

    def stop(self) -> None:
        """Stops the Spark session."""
        self.spark.stop()


if __name__ == "__main__":
    # Example usage
    config = ETLConfig()
    processor = SparkETLProcessor(config)
    
    # In a real scenario, paths would be passed via arguments or config files
    # raw_df = processor.read_data("path/to/raw/data")
    # transformed_df = processor.transform(raw_df)
    # processor.write_data(transformed_df, "path/to/processed/data")
    
    processor.stop()
