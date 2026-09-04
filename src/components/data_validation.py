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
            expected = len(self.schema_config["columns"])
            return len(dataframe.columns) == expected
        except Exception as e:
            raise customException(e, sys)

    def is_numerical_columns_exits(self, dataframe: pd.DataFrame) -> bool:
        try:
            for column in self.schema_config["numerical_columns"]:
                if column not in dataframe.columns:
                    logging.error(f"Missing column: {column}")
                    return False
                # not just presence — dtype has to actually be numeric
                if not pd.api.types.is_numeric_dtype(dataframe[column]):
                    logging.error(f"{column} is not numeric")
                    return False
            return True
        except Exception as e:
            raise customException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            drift_status = False
            report = {}
            # reuse schema's numerical_columns instead of a separate hardcoded list
            drift_columns = self.schema_config["numerical_columns"]

            for column in drift_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    continue

                p_value = float(ks_2samp(base_df[column].dropna(), current_df[column].dropna()).pvalue)
                column_drift = p_value < threshold
                drift_status = drift_status or column_drift

                if column_drift:
                    logging.warning(f"Drift in {column} | p={p_value:.4f}")

                report[column] = {"p_value": p_value, "drift_status": column_drift}

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path, content=report)

            return drift_status

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = DataValidation.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = DataValidation.read_data(self.data_ingestion_artifact.test_file_path)

            error_message = ""
            if not self.validate_number_of_columns(train_df):
                error_message += "Train: column count mismatch.\n"
            if not self.validate_number_of_columns(test_df):
                error_message += "Test: column count mismatch.\n"
            if not self.is_numerical_columns_exits(train_df):
                error_message += "Train: numerical columns check failed.\n"
            if not self.is_numerical_columns_exits(test_df):
                error_message += "Test: numerical columns check failed.\n"

            if error_message:
                logging.error("Validation failed — schema mismatch.")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok=True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path, index=False)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path, index=False)
                raise customException(error_message, sys)

            drift_detected = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info(f"Validation done. Drift: {drift_detected}")

            return DataValidationArtifact(
                validation_status=True,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise customException(e, sys)