from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifacts_entity import DataIngestionArtifact, DataValidationArtifact
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise customException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise customException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["columns"]
            expected_number = len(expected_columns)
            actual_number = len(dataframe.columns)

            logging.info(f"Required number of columns: {expected_number}")
            logging.info(f"Dataframe has columns: {actual_number}")

            return actual_number == expected_number
        except Exception as e:
            raise customException(e, sys)

    def is_numerical_columns_exits(self, dataframe: pd.DataFrame) -> bool:
        try:
            expected_columns = self.schema_config["numerical_columns"]

            for column in expected_columns:
                if column not in dataframe.columns:
                    logging.error(f"Column {column} is missing in dataset.")
                    return False
            return True
        except Exception as e:
            raise customException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            drift_status = False
            report = {}

            drift_columns = ["co", "no2", "o3", "pm10", "pm25", "so2", "aqi"]

            for column in drift_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    logging.warning(f"Column {column} not found in both datasets.")
                    continue

                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                ks_result = ks_2samp(d1, d2)
                p_value = float(ks_result.pvalue)

                if p_value < threshold:
                    column_drift = True
                    drift_status = True
                    logging.warning(f"Drift detected in {column} | p-value: {p_value}")
                else:
                    column_drift = False
                    logging.info(f"No drift in {column} | p-value: {p_value}")

                report[column] = {
                    "p_value": p_value,
                    "drift_status": column_drift
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)

            write_yaml_file(
                file_path=drift_report_file_path,
                content=report
            )

            if drift_status:
                logging.warning("Dataset drift detected. Pipeline will continue.")
            else:
                logging.info("No significant dataset drift detected.")

            return drift_status

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            error_message = ""

            status = self.validate_number_of_columns(train_df)
            if not status:
                error_message += "Train dataframe does not contain required columns.\n"

            status = self.validate_number_of_columns(test_df)
            if not status:
                error_message += "Test dataframe does not contain required columns.\n"

            numerical_col_exist = self.is_numerical_columns_exits(train_df)
            if not numerical_col_exist:
                error_message += "Train dataframe does not contain all numerical columns.\n"

            numerical_col_exist = self.is_numerical_columns_exits(test_df)
            if not numerical_col_exist:
                error_message += "Test dataframe does not contain all numerical columns.\n"

            if error_message:
                logging.error("Data validation failed due to schema errors.")

                invalid_dir_path = os.path.dirname(
                    self.data_validation_config.invalid_train_file_path
                )
                os.makedirs(invalid_dir_path, exist_ok=True)

                train_df.to_csv(
                    self.data_validation_config.invalid_train_file_path,
                    index=False
                )
                test_df.to_csv(
                    self.data_validation_config.invalid_test_file_path,
                    index=False
                )

                raise customException(error_message, sys)

            drift_detected = self.detect_dataset_drift(
                base_df=train_df,
                current_df=test_df
            )

            valid_dir_path = os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )
            os.makedirs(valid_dir_path, exist_ok=True)

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False
            )
            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False
            )

            data_validation_artifact = DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            logging.info(
                f"Data validation completed. Drift detected: {drift_detected}"
            )

            return data_validation_artifact

        except Exception as e:
            raise customException(e, sys)