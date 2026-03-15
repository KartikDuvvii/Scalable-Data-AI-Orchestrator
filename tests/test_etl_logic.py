import pytest
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from src.etl.pyspark_processor import SparkETLProcessor, ETLConfig


@pytest.fixture(scope="session")
def spark_session():
    """Provides a shared Spark session for all tests in the session."""
    return SparkSession.builder.master("local[*]").appName("pytest-spark").getOrCreate()


@pytest.fixture
def etl_processor():
    """Provides a SparkETLProcessor instance with default config."""
    config = ETLConfig(app_name="TestProcessor", master="local[*]")
    processor = SparkETLProcessor(config)
    yield processor
    processor.stop()


def test_transform_adds_columns(spark_session, etl_processor):
    """Verifies that the transformation adds the expected audit columns."""
    # Create sample data
    schema = StructType([
        StructField("id", IntegerType(), True),
        StructField("feature1", StringType(), True)
    ])
    data = [(1, "A"), (2, "B")]
    df = spark_session.createDataFrame(data, schema)

    # Apply transformation
    transformed_df = etl_processor.transform(df)

    # Assertions
    expected_columns = {"id", "feature1", "processed_at", "is_processed"}
    actual_columns = set(transformed_df.columns)
    
    assert expected_columns.issubset(actual_columns)
    assert transformed_df.count() == 2


def test_schema_validation_passes(spark_session, etl_processor):
    """Verifies that schema validation correctly identifies matching schemas."""
    schema = StructType([
        StructField("col1", StringType(), False),
        StructField("col2", IntegerType(), True)
    ])
    df = spark_session.createDataFrame([], schema)

    assert etl_processor.validate_schema(df, schema) is True


def test_schema_validation_fails(spark_session, etl_processor):
    """Verifies that schema validation fails when schemas mismatch."""
    schema_actual = StructType([
        StructField("col1", StringType(), False)
    ])
    schema_expected = StructType([
        StructField("col1", IntegerType(), False)  # Different type
    ])
    df = spark_session.createDataFrame([], schema_actual)

    assert etl_processor.validate_schema(df, schema_expected) is False
