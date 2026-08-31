import os
import sys
import numpy as np
import mlflow

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils.ml_utils.model.estimator import AQIModel
from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.artifacts_entity import (
    DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
)
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils.utils import save_obj, load_obj, evaluate_models

import dagshub

dagshub.init(repo_owner="RAHUL-VERMA-CODE",
             repo_name="Self-Healing-AQI-Forecaster",
             mlflow=True)


class ModelTrainer:

    def __init__(self, data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise customException(e, sys)

    def track_mlflow(self, best_model, metric):
        try:
            with mlflow.start_run():
                mlflow.log_metric("r2_score", metric.r2_score)
                mlflow.log_metric("mae", metric.mean_absolute_error)
                mlflow.log_metric("mse", metric.mean_squared_error)
                mlflow.log_metric("rmse", np.sqrt(metric.mean_squared_error))
                mlflow.sklearn.log_model(best_model, name="model")
        except Exception as e:
            raise customException(e, sys)

    def get_regression_score(self, y_true, y_pred) -> RegressionMetricArtifact:
        try:
            metric = RegressionMetricArtifact(
                r2_score=r2_score(y_true, y_pred),
                mean_absolute_error=mean_absolute_error(y_true, y_pred),
                mean_squared_error=mean_squared_error(y_true, y_pred)
            )
            return metric
        except Exception as e:
            raise customException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(
                    random_state=42,
                    n_jobs=-1,
                    objective="reg:squarederror"
                )
            }

            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1]},
                "ElasticNet": {
                    "alpha": [0.01, 0.1],
                    "l1_ratio": [0.5, 0.8]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "absolute_error"],
                    "max_depth": [5, 10, None]
                },
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, None]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.05, 0.1],
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5]
                },
                "XGBoost": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1],
                    "max_depth": [3, 5]
                }
            }

            logging.info("Starting model evaluation and hyperparameter tuning...")

            model_report = evaluate_models(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                models=models,
                params=params
            )

            valid_report = {
                k: v for k, v in model_report.items() if np.isfinite(v)
            }

            if not valid_report:
                raise Exception("No valid model score was produced.")

            best_model_name = max(valid_report, key=valid_report.get)
            best_model_score = valid_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(
                f"Best model: {best_model_name} | R2: {best_model_score}"
            )

            train_metric = self.get_regression_score(
                y_train, best_model.predict(x_train)
            )

            test_metric = self.get_regression_score(
                y_test, best_model.predict(x_test)
            )

            logging.info(f"Train metrics: {train_metric}")
            logging.info(f"Test metrics: {test_metric}")

            self.track_mlflow(best_model, test_metric)

            preprocessor = load_obj(
                self.data_transformation_artifact.transformed_object_file_path
            )

            model_path = self.model_trainer_config.trained_model_file_path

            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            save_obj(
                model_path,
                AQIModel(
                    preprocessor=preprocessor,
                    model=best_model
                )
            )

            logging.info(f"Model saved at: {model_path}")

            return ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            raise customException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = np.load(
                self.data_transformation_artifact.transformed_train_file_path
            )

            test_arr = np.load(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train = train_arr[:, :-1]
            y_train = train_arr[:, -1]
            x_test = test_arr[:, :-1]
            y_test = test_arr[:, -1]

            logging.info(
                f"Train shape: {x_train.shape} | Test shape: {x_test.shape}"
            )

            return self.train_model(
                x_train, y_train, x_test, y_test
            )

        except Exception as e:
            raise customException(e, sys)

