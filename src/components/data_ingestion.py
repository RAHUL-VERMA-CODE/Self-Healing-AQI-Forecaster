from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifact

import sys
import os
import pandas as pd


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise customException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            dataframe = dataframe.sort_values(by="timestamp").reset_index(drop=True)

            test_size = self.data_ingestion_config.train_test_split_ratio

            # split by unique timestamp, not row index — all 5 locations share
            # the same hours, so splitting by index would leak rows across sets
            unique_ts = dataframe["timestamp"].unique()
            cutoff_ts = unique_ts[int(len(unique_ts) * (1 - test_size))]

            train_set = dataframe[dataframe["timestamp"] < cutoff_ts].copy()
            test_set = dataframe[dataframe["timestamp"] >= cutoff_ts].copy()

            logging.info(f"Train: {train_set.shape}, Test: {test_set.shape}")

            os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)

            # stable path (not timestamped) — DriftMonitor always reads this
            # as the "latest" baseline, regardless of which run folder it's in
            os.makedirs("final", exist_ok=True)
            train_set.to_csv(os.path.join("final", "train.csv"), index=False)

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            # keep schema in sync with new_data/incoming.csv — check_and_retrain.py
            # merges the two, so they must have identical columns
            df = pd.read_csv("AQI_Data/AQI_cleaned.csv")
            logging.info(f"Raw dataset shape: {df.shape}")

            self.split_data_as_train_test(df)

            return DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path
            )

        except Exception as e:
            raise customException(e, sys)