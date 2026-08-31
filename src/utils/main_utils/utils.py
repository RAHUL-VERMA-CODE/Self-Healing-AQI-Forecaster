import yaml
from src.exception.exception import customException
from src.logging.logger import logging
import os
import sys
import pickle
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import r2_score

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise customException(e, sys)

def write_yaml_file(file_path: str, content: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise customException(e, sys)

def save_obj(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object method")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object method")
    except Exception as e:
        raise customException(e, sys)

def load_obj(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise customException(e, sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, params):
    try:
        report = {}
        time_split = TimeSeriesSplit(n_splits=3)

        for model_name, model in models.items():
            logging.info(f"Training model: {model_name}")

            gs = GridSearchCV(
                estimator=model,
                param_grid=params[model_name],
                cv=time_split,
                n_jobs=-1,
                scoring="r2"
            )
            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(
                f"{model_name} | Train R2: {train_score:.6f} | "
                f"Test R2: {test_score:.6f}"
            )

            report[model_name] = test_score

        return report

    except Exception as e:
        raise customException(e, sys)