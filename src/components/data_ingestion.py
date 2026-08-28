from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts_entity import DataIngestionArtifact

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
            logging.info("data ingestion configuration initialized successfully")

        except Exception as e:
            raise customException(e,sys)    

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info("Starting train-test split.")
            train_set,test_set=train_test_split(
                dataframe,
                test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info(
                f"Train shape: {train_set.shape}, "
                f"Test shape: {test_set.shape}"
            ) 
            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)

            # save train data
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,
                index=False,
                header=True
            )

              # Save test data
            test_set.to_csv(
                self.data_ingestion_config.testing_file_path,
                index=False,
                header=True
            )

            logging.info(
                f"Training file saved at: "
                f"{self.data_ingestion_config.training_file_path}"
            )
      
        except Exception as e:
            raise customException(e,sys)    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
            try:
                logging.info("==========Data ingestion started==========")

                df=pd.read_csv("AQI_Data/AQI_cleaned.csv")
                self.split_data_as_train_test(df)

                # create artifacts

                data_ingestion_artifacts=DataIngestionArtifact(
                    trained_file_path=(
                        self.data_ingestion_config.training_file_path
                    ),
                    test_file_path=self.data_ingestion_config.testing_file_path
                )
                
                logging.info(
                "Data Ingestion Artifact created successfully."
                )

                logging.info("========== Data Ingestion Completed ==========")
                return data_ingestion_artifacts
            except Exception as e:
                raise customException(e,sys)