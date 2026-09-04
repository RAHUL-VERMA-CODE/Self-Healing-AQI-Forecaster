import os 
import sys 
import pandas as pd 
 
from src.exception.exception import customException 
from src.logging.logger import logging 
from src.pipeline.training_pipeline import TrainingPipeline 
from src.components.drift_monitor import DriftMonitor 
 
BASELINE_PATH = os.path.join("final", "train.csv") 
NEW_DATA_PATH = os.path.join("new_data", "incoming.csv") 
MAIN_DATA_PATH = os.path.join("AQI_Data", "AQI_cleaned.csv") 
PRODUCTION_DRIFT_REPORT = os.path.join("final", "production_drift_report.yaml") 
 
 
def merge_new_data_into_main_dataset(): 
    try: 
        main_df = pd.read_csv(MAIN_DATA_PATH) 
        new_df = pd.read_csv(NEW_DATA_PATH) 
 
        # guard against silent schema drift — concat won't error on 
        # mismatched columns, it just fills gaps with NaN 
        if set(new_df.columns) != set(main_df.columns): 
            raise ValueError( 
                f"Column mismatch. main: {list(main_df.columns)}, " 
                f"new: {list(new_df.columns)}" 
            ) 
 
        combined = pd.concat([main_df, new_df], ignore_index=True) 
        combined = combined.drop_duplicates(subset=["timestamp", "location_name"]) 
        combined.to_csv(MAIN_DATA_PATH, index=False) 
 
        logging.info(f"Merged {len(new_df)} new rows. Total rows: {len(combined)}") 
 
        # clear incoming.csv so the same rows aren't merged twice 
        new_df.iloc[0:0].to_csv(NEW_DATA_PATH, index=False) 
 
    except Exception as e: 
        raise customException(e, sys) 
 
 
def check_drift_and_retrain(): 
    try: 
        if not os.path.exists(BASELINE_PATH): 
            logging.warning(f"Baseline file not found: {BASELINE_PATH}. Run main.py first.") 
            return 
 
        if not os.path.exists(NEW_DATA_PATH): 
            logging.warning(f"New data file not found: {NEW_DATA_PATH}") 
            return 
 
        monitor = DriftMonitor( 
            baseline_path=BASELINE_PATH, 
            new_data_path=NEW_DATA_PATH, 
            threshold=0.05 
        ) 
 
        drift_detected = monitor.check_drift(report_path=PRODUCTION_DRIFT_REPORT) 
 
        if not drift_detected: 
            print("Drift NOT detected") 
            logging.info("No drift detected, retraining skipped.") 
            return 
 
        print("Drift DETECTED") 
        logging.info("Drift detected. Merging new data and retraining.") 
        merge_new_data_into_main_dataset() 
 
        pipeline = TrainingPipeline() 
        artifact = pipeline.run_pipeline() 
 
        logging.info(f"Retraining completed. New model artifact: {artifact}") 
 
    except Exception as e: 
        raise customException(e, sys) 
 
 
if __name__ == "__main__": 
    check_drift_and_retrain() 