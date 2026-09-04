from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifacts_entity import DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils.utils import save_obj
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
import os
import sys


class DataTransformation:
    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise customException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise customException(e, sys)

    def get_data_transformer_object(self):
        try:
            # "aqi" here is the current-hour value — a valid input, not the target.
            # it's known at prediction time, so no leakage
            numerical_columns = ["co", "no2", "o3", "pm10", "pm25", "so2", "aqi",
                                  "hour", "month", "day_of_week", "is_weekend"]
            categorical_columns = ["location_name"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num", num_pipeline, numerical_columns),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise customException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)

            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

            train_df = train_df.sort_values(["location_name", "timestamp"]).reset_index(drop=True)
            test_df = test_df.sort_values(["location_name", "timestamp"]).reset_index(drop=True)

            # next-hour aqi, per location — this is what we're actually predicting
            train_df["target_aqi"] = train_df.groupby("location_name")["aqi"].shift(-1)
            test_df["target_aqi"] = test_df.groupby("location_name")["aqi"].shift(-1)

            # last row per location has no "next hour" — nothing to predict against
            train_df = train_df.dropna(subset=["target_aqi"]).reset_index(drop=True)
            test_df = test_df.dropna(subset=["target_aqi"]).reset_index(drop=True)

            columns_to_drop = ["timestamp", "target_aqi"]

            X_train = train_df.drop(columns=columns_to_drop)
            y_train = train_df["target_aqi"]

            X_test = test_df.drop(columns=columns_to_drop)
            y_test = test_df["target_aqi"]

            logging.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

            preprocessor = self.get_data_transformer_object()
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[X_train_transformed, y_train.values]
            test_arr = np.c_[X_test_transformed, y_test.values]

            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            np.save(self.data_transformation_config.transformed_train_file_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_file_path, test_arr)
            save_obj(self.data_transformation_config.transformed_object_file_path, preprocessor)

            logging.info("Transformation done")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            raise customException(e, sys)