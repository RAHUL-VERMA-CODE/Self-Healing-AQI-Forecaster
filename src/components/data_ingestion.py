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
            logging.info("Data ingestion configuration initialized successfully")
        except Exception as e:
            raise customException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting time-based train-test split.")

            dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
            dataframe = dataframe.sort_values(
                by="timestamp"
            ).reset_index(drop=True)

            test_size = self.data_ingestion_config.train_test_split_ratio
            split_index = int(len(dataframe) * (1 - test_size))

            train_set = dataframe.iloc[:split_index].copy()
            test_set = dataframe.iloc[split_index:].copy()

            logging.info(
                f"Train shape: {train_set.shape}, "
                f"Test shape: {test_set.shape}"
            )
            logging.info(
                f"Train period: {train_set['timestamp'].min()} -> "
                f"{train_set['timestamp'].max()}"
            )
            logging.info(
                f"Test period: {test_set['timestamp'].min()} -> "
                f"{test_set['timestamp'].max()}"
            )

            dir_path = os.path.dirname(
                self.data_ingestion_config.training_file_path
            )
            os.makedirs(dir_path, exist_ok=True)

            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False
            )
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False
            )

            logging.info(
                f"Training file saved at: "
                f"{self.data_ingestion_config.training_file_path}"
            )
            logging.info(
                f"Testing file saved at: "
                f"{self.data_ingestion_config.testing_file_path}"
            )

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("========== Data Ingestion Started ==========")

            df = pd.read_csv("AQI_Data/AQI_cleaned.csv")
            logging.info(f"Raw dataset shape: {df.shape}")

            self.split_data_as_train_test(df)

            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=(
                    self.data_ingestion_config.training_file_path
                ),
                test_file_path=(
                    self.data_ingestion_config.testing_file_path
                )
            )

            logging.info("Data Ingestion Artifact created successfully.")
            logging.info("========== Data Ingestion Completed ==========")

            return data_ingestion_artifact

        except Exception as e:
            raise customException(e, sys)