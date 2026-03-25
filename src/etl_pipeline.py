"""
House Sale Data ETL Pipeline
============================
Implement the three functions below to complete the ETL pipeline.

Steps:
  1. EXTRACT  – load the CSV into a PySpark DataFrame
  2. TRANSFORM – split the data by neighborhood and save each as a separate CSV
  3. LOAD      – insert each neighborhood DataFrame into its own PostgreSQL table
"""
from __future__ import annotations

import csv  # noqa: F401
import os  # noqa: F401
from pathlib import Path

from dotenv import load_dotenv  # noqa: F401
from pyspark.sql import DataFrame, SparkSession  # noqa: F401
from pyspark.sql import functions as F  # noqa: F401
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ── Predefined constants (do not modify) ──────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

NEIGHBORHOODS = [
    "Downtown", "Green Valley", "Hillcrest", "Lakeside", "Maple Heights",
    "Oakwood", "Old Town", "Riverside", "Suburban Park", "University District",
]

OUTPUT_DIR   = ROOT / "output" / "by_neighborhood"
OUTPUT_FILES = {hood: OUTPUT_DIR / f"{hood.replace(' ', '_').lower()}.csv" for hood in NEIGHBORHOODS}

PG_TABLES = {hood: f"public.{hood.replace(' ', '_').lower()}" for hood in NEIGHBORHOODS}

PG_COLUMN_SCHEMA = (
    "house_id TEXT, neighborhood TEXT, price INTEGER, square_feet INTEGER, "
    "num_bedrooms INTEGER, num_bathrooms INTEGER, house_age INTEGER, "
    "garage_spaces INTEGER, lot_size_acres NUMERIC(6,2), has_pool BOOLEAN, "
    "recently_renovated BOOLEAN, energy_rating TEXT, location_score INTEGER, "
    "school_rating INTEGER, crime_rate INTEGER, "
    "distance_downtown_miles NUMERIC(6,2), sale_date DATE, days_on_market INTEGER"
)

# Column names derived from PG_COLUMN_SCHEMA for the load step
_PG_COLUMNS = [col.split()[0] for col in PG_COLUMN_SCHEMA.split(", ")]

_CSV_SCHEMA = StructType([
    StructField("house_id", StringType(), True),
    StructField("neighborhood", StringType(), True),
    StructField("price", IntegerType(), True),
    StructField("square_feet", IntegerType(), True),
    StructField("num_bedrooms", IntegerType(), True),
    StructField("num_bathrooms", IntegerType(), True),
    StructField("house_age", IntegerType(), True),
    StructField("garage_spaces", IntegerType(), True),
    StructField("lot_size_acres", DoubleType(), True),
    StructField("has_pool", BooleanType(), True),
    StructField("recently_renovated", BooleanType(), True),
    StructField("energy_rating", StringType(), True),
    StructField("location_score", IntegerType(), True),
    StructField("school_rating", IntegerType(), True),
    StructField("crime_rate", IntegerType(), True),
    StructField("distance_downtown_miles", DoubleType(), True),
    StructField("sale_date", DateType(), True),
    StructField("days_on_market", IntegerType(), True),
    StructField("buyer_id", StringType(), True),
    StructField("buyer_budget", IntegerType(), True),
    StructField("buyer_age_group", StringType(), True),
    StructField("buyer_family_size", IntegerType(), True),
    StructField("buyer_income_level", StringType(), True),
    StructField("has_children", BooleanType(), True),
    StructField("employment_type", StringType(), True),
    StructField("buyer_preference", StringType(), True),
    StructField("first_time_buyer", BooleanType(), True),
])


def extract(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load the CSV dataset into a PySpark DataFrame with correct data types."""
    return (
        spark.read.csv(
            csv_path,
            header=True,
            schema=_CSV_SCHEMA,
            dateFormat="M/d/yy",
        )
    )


def transform(df: DataFrame) -> dict[str, DataFrame]:
    """Split the data by neighborhood and save each as a separate CSV file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    partitions: dict[str, DataFrame] = {}
    for hood in NEIGHBORHOODS:
        hood_df = df.filter(F.col("neighborhood") == hood).orderBy("house_id")
        partitions[hood] = hood_df
        # Write via pandas so booleans are True/False and dates are ISO-formatted
        hood_df.toPandas().to_csv(OUTPUT_FILES[hood], index=False, float_format='%g')
    return partitions


def load(partitions: dict[str, DataFrame], jdbc_url: str, pg_props: dict) -> None:
    """Insert each neighborhood dataset into its own PostgreSQL table."""
    for hood, hood_df in partitions.items():
        hood_df.select(_PG_COLUMNS).write.jdbc(
            url=jdbc_url,
            table=PG_TABLES[hood],
            mode="overwrite",
            properties=pg_props,
        )


# ── Main (do not modify) ───────────────────────────────────────────────────────
def main() -> None:
    load_dotenv(ROOT / ".env")

    jdbc_url = (
        f"jdbc:postgresql://{os.getenv('PG_HOST', 'localhost')}:"
        f"{os.getenv('PG_PORT', '5432')}/{os.environ['PG_DATABASE']}"
    )
    pg_props = {
        "user":     os.environ["PG_USER"],
        "password": os.getenv("PG_PASSWORD", ""),
        "driver":   "org.postgresql.Driver",
    }
    csv_path = str(ROOT / os.getenv("DATASET_DIR", "dataset") / os.getenv("DATASET_FILE", "historical_purchases.csv"))

    spark = (
        SparkSession.builder.appName("HouseSaleETL")
        .config("spark.jars.packages", "org.postgresql:postgresql:42.7.3")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    df         = extract(spark, csv_path)
    partitions = transform(df)
    load(partitions, jdbc_url, pg_props)

    spark.stop()


if __name__ == "__main__":
    main()
