import os
import sys
import numpy as np
import mlflow
import mlflow.xgboost
import dagshub

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils.ml_utils.model.estimator import AQIModel
from src.exception.exception import customException
from src.logging.logger import logging
from src.entity.artifacts_entity import DataTransformationArtifact, ModelTrainerArtifact, RegressionMetricArtifact
from src.entity.config_entity import ModelTrainerConfig
from src.utils.main_utils.utils import save_obj, load_obj, evaluate_models

dagshub.init(repo_owner="RAHUL-VERMA-CODE", repo_name="Self-Healing-AQI-Forecaster", mlflow=True)


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

                # XGBoost needs mlflow.xgboost's logger — mlflow.sklearn's
                # skops-based serializer blocks XGBoost's internal C++ types
                # as "untrusted" by default
                if isinstance(best_model, XGBRegressor):
                    mlflow.xgboost.log_model(best_model, name="model")
                else:
                    mlflow.sklearn.log_model(best_model, name="model")

        except Exception as e:
            raise customException(e, sys)

    def get_regression_score(self, y_true, y_pred) -> RegressionMetricArtifact:
        try:
            return RegressionMetricArtifact(
                r2_score=r2_score(y_true, y_pred),
                mean_absolute_error=mean_absolute_error(y_true, y_pred),
                mean_squared_error=mean_squared_error(y_true, y_pred)
            )
        except Exception as e:
            raise customException(e, sys)

    def train_model(self, x_train, y_train, x_test, y_test):
        try:
            # a spread of linear, tree, and boosting models — let evaluate_models
            # pick whichever generalizes best on the test set
            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42, n_jobs=-1, objective="reg:squarederror")
            }

            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1, 1.0, 10.0]},
                "Lasso": {"alpha": [0.001, 0.01, 0.1]},
                "ElasticNet": {"alpha": [0.01, 0.1], "l1_ratio": [0.5, 0.8]},
                "Decision Tree": {"criterion": ["squared_error", "absolute_error"], "max_depth": [5, 10, None]},
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, None]},
                "Gradient Boosting": {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200], "max_depth": [3, 5]},
                "XGBoost": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}
            }

            model_report = evaluate_models(
                x_train=x_train, y_train=y_train,
                x_test=x_test, y_test=y_test,
                models=models, params=params
            )

            valid_report = {k: v for k, v in model_report.items() if np.isfinite(v)}
            if not valid_report:
                raise Exception("No valid model score produced.")

            best_model_name = max(valid_report, key=valid_report.get)
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name} | R2: {valid_report[best_model_name]:.4f}")

            train_metric = self.get_regression_score(y_train, best_model.predict(x_train))
            test_metric = self.get_regression_score(y_test, best_model.predict(x_test))

            self.track_mlflow(best_model, test_metric)

            preprocessor = load_obj(self.data_transformation_artifact.transformed_object_file_path)
            model_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            # bundle preprocessor + model together — inference just needs one object
            trained_model = AQIModel(preprocessor=preprocessor, model=best_model)
            save_obj(model_path, trained_model)
            logging.info(f"Model saved: {model_path}")

            # stable path (not timestamped) — FastAPI always loads from here,
            # so it doesn't need to track which run's Artifacts folder is latest.
            # Same idea as final/train.csv in data_ingestion.py
            os.makedirs("final", exist_ok=True)
            save_obj(os.path.join("final", "model.pkl"), trained_model)
            save_obj(os.path.join("final", "preprocessor.pkl"), preprocessor)
            logging.info("Latest model + preprocessor copied to final/")

            return ModelTrainerArtifact(
                trained_model_file_path=model_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )

        except Exception as e:
            raise customException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = np.load(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = np.load(self.data_transformation_artifact.transformed_test_file_path)

            x_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            x_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise customException(e, sys)
