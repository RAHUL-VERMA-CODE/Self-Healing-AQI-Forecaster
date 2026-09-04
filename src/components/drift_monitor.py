import sys
import pandas as pd
from scipy.stats import ks_2samp

from src.exception.exception import customException
from src.logging.logger import logging
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DriftMonitor:
    def __init__(self, baseline_path: str, new_data_path: str, threshold: float = 0.05):
        self.baseline_path = baseline_path
        self.new_data_path = new_data_path
        self.threshold = threshold
        # reuse the same schema used in data_validation, so drift columns
        # stay consistent across training and production checks
        self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)

    @staticmethod
    def read_data(path) -> pd.DataFrame:
        try:
            return pd.read_csv(path)
        except Exception as e:
            raise customException(e, sys)

    def check_drift(self, report_path: str) -> bool:
        try:
            baseline_df = self.read_data(self.baseline_path)
            new_df = self.read_data(self.new_data_path)

            drift_columns = self.schema_config["numerical_columns"]
            report = {}
            drift_status = False

            for column in drift_columns:
                if column not in baseline_df.columns or column not in new_df.columns:
                    logging.warning(f"Skipping '{column}' — missing in baseline or new data.")
                    continue

                d1 = baseline_df[column].dropna()
                d2 = new_df[column].dropna()

                p_value = float(ks_2samp(d1, d2).pvalue)
                column_drift = p_value < self.threshold
                drift_status = drift_status or column_drift

                if column_drift:
                    logging.warning(f"Production drift detected in '{column}': p_value={p_value:.6f}")

                report[column] = {"p_value": p_value, "drift_status": column_drift}

            write_yaml_file(report_path, report)
            return drift_status

        except Exception as e:
            raise customException(e, sys)