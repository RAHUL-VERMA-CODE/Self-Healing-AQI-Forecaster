import os
import sys
import numpy as np
import pandas as pd


"""
defining common constant variable for training pipeline
"""
TARGET_COLUMN="aqi"
ARTIFACT_DIR:str="Artifacts"
# FILE_NAME:str="raw_data.csv"  # (due to file size we avoid this)which is already saved in AQI_Data folder

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

"""
Data validation related constant start with DATA_VALIDATION VAR NAME 
"""
DATA_VALIDATION_DIR_NAME:str="data_validation"
DATA_VALIDATION_VALID_DIR:str="validated"
DATA_VALIDATION_INVALID_DIR:str="invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str="Drift_Report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str="report.yaml"

"""
Data transformation related constant start with DATA_TRANSFORMATION VAR NAME 
"""
DATA_TRANSFORMATION_DIR_NAME:str="data_transformation"
DATA_TRANSFORMATION_DATA_DIR:str="transformed"
DATA_TRANSFORMATION_OBJECT_DIR:str="transformed_object"
PREPROCESSING_OBJECT_FILE_NAME="preprocessing.pkl"
DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npy"

DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npy"