from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataValidationConfig
from src.entity.artifacts_entity import DataIngestionArtifact,DataValidationArtifact
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os
import sys
from src.utils.main_utils.utils import read_yaml_file,write_yaml_file

class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self.schema_config=read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise customException(e,sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise customException(e,sys)

    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns=len(self.schema_config["columns"])
            logging.info(f"Required number of columns:{number_of_columns}")
            logging.info(f"Data frame has columns:{len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise customException(e,sys)

    def is_numerical_columns_exits(self,dataframe:pd.DataFrame)->bool:
        try:
            expected_columns=self.schema_config["numerical_columns"]
            for column in expected_columns:
                if column not in dataframe.columns:
                    logging.error(f"Column {column} is missing in dataset.")
                    return False
            return True
        except Exception as e:
            raise customException(e,sys)

    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report={}
            for column in base_df.columns:
                d1=base_df[column]
                d2=current_df[column]
                is_same_dist=ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found=True
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                }})

            drift_report_file_path=self.data_validation_config.drift_report_file_path
            dir_path=os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)

            return status

        except Exception as e:
            raise customException(e,sys)

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path=self.data_ingestion_artifact.trained_file_path
            test_file_path=self.data_ingestion_artifact.test_file_path

            ## read the data from train and test
            train_df=DataValidation.read_data(train_file_path)
            test_df=DataValidation.read_data(test_file_path)

            error_message=""

            ## validate number of columns
            status=self.validate_number_of_columns(train_df)
            if not status:
                error_message+=f"Train dataframe does not contain all columns.\n"

            status=self.validate_number_of_columns(test_df)
            if not status:
                error_message+=f"Test dataframe does not contain all columns.\n"

            ## validate number of numeric columns
            numerical_col_exist=self.is_numerical_columns_exits(dataframe=train_df)
            if not numerical_col_exist:
                error_message+=f"Train dataframe does not contain all numerical columns.\n"

            numerical_col_exist=self.is_numerical_columns_exits(dataframe=test_df)
            if not numerical_col_exist:
                error_message+=f"Test dataframe does not contain all numerical columns.\n"

            ## create directory for invalid data
            invalid_dir_path=os.path.dirname(
                self.data_validation_config.invalid_train_file_path
            )
            os.makedirs(invalid_dir_path,exist_ok=True)

            ## save invalid data and stop pipeline
            if error_message:
                train_df.to_csv(
                    self.data_validation_config.invalid_train_file_path,
                    index=False,
                    header=True
                )

                test_df.to_csv(
                    self.data_validation_config.invalid_test_file_path,
                    index=False,
                    header=True
                )

                raise customException(error_message,sys)

            ### data drift check
            status=self.detect_dataset_drift(
                base_df=train_df,
                current_df=test_df
            )

            ## create directory for valid data
            dir_path=os.path.dirname(
                self.data_validation_config.valid_train_file_path
            )
            os.makedirs(dir_path,exist_ok=True)

            ## save valid train data
            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True
            )

            ## save valid test data
            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True
            )

            data_validation_artifact=DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

            return data_validation_artifact

        except Exception as e:
            raise customException(e,sys)