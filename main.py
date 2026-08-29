from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation


from src.exception.exception import customException
from src.logging.logger import logging


from src.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig)

import sys

if __name__=="__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        
        data_ingestion_config=DataIngestionConfig(training_pipeline_config)
        data_ingestion=DataIngestion(data_ingestion_config)
        data_ingestion_artifacts=(
            data_ingestion.initiate_data_ingestion())
        print(data_ingestion_artifacts)

        data_validation_config=DataValidationConfig(training_pipeline_config)
        data_validation=DataValidation(data_ingestion_artifacts,data_validation_config)
        data_validation_artifact=data_validation.initiate_data_validation()
        print(data_validation_artifact)

    except Exception as e:
        raise customException(e,sys)
    