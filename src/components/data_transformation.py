from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifacts_entity import DataValidationArtifact, DataTransformationArtifact
from src.utils.main_utils.utils import save_obj
from sklearn.compose import ColumnTransformer
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
            numerical_columns = [
                "location_lat", "location_lon", "co", "no2", "o3",
                "pm10", "pm25", "so2", "hour", "day", "month",
                "year", "day_of_week", "is_weekend"
            ]
            categorical_columns = ["location_name"]

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), numerical_columns),
                    ("cat", OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    ), categorical_columns)
                ]
            )
            logging.info("Preprocessing pipeline created")
            return preprocessor
        except Exception as e:
            raise customException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("========== Data Transformation Started ==========")

            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            train_df = DataTransformation.read_data(train_file_path)
            test_df = DataTransformation.read_data(test_file_path)

            train_df["timestamp"] = pd.to_datetime(train_df["timestamp"])
            test_df["timestamp"] = pd.to_datetime(test_df["timestamp"])

            train_df = train_df.sort_values(
                ["location_name", "timestamp"]
            ).reset_index(drop=True)
            test_df = test_df.sort_values(
                ["location_name", "timestamp"]
            ).reset_index(drop=True)

            # Next-hour AQI target
            train_df["target_aqi"] = (
                train_df.groupby("location_name")["aqi"].shift(-1)
            )
            test_df["target_aqi"] = (
                test_df.groupby("location_name")["aqi"].shift(-1)
            )

            train_df = train_df.dropna(
                subset=["target_aqi"]
            ).reset_index(drop=True)
            test_df = test_df.dropna(
                subset=["target_aqi"]
            ).reset_index(drop=True)

            # Timestamp features
            for df in [train_df, test_df]:
                df["hour"] = df["timestamp"].dt.hour
                df["day"] = df["timestamp"].dt.day
                df["month"] = df["timestamp"].dt.month
                df["year"] = df["timestamp"].dt.year
                df["day_of_week"] = df["timestamp"].dt.dayofweek
                df["is_weekend"] = (
                    df["day_of_week"] >= 5
                ).astype(int)

            target_column = "target_aqi"

            # Current AQI must not be used as feature
            columns_to_drop = [
                "timestamp",
                "aqi",
                target_column
            ]

            X_train = train_df.drop(columns=columns_to_drop)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=columns_to_drop)
            y_test = test_df[target_column]

            logging.info(f"X_train shape: {X_train.shape}")
            logging.info(f"X_test shape: {X_test.shape}")

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            train_arr = np.c_[
                X_train_transformed,
                y_train.values
            ]
            test_arr = np.c_[
                X_test_transformed,
                y_test.values
            ]

            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_train_file_path
                ),
                exist_ok=True
            )
            os.makedirs(
                os.path.dirname(
                    self.data_transformation_config.transformed_object_file_path
                ),
                exist_ok=True
            )

            np.save(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )
            np.save(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            save_obj(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor
            )

            logging.info("Data transformation completed")
            logging.info("========== Data Transformation Completed ==========")

            return DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )

        except Exception as e:
            raise customException(e, sys)