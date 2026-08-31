import sys
from src.pipeline.training_pipeline import TrainingPipeline
from src.exception.exception import customException

if __name__=="__main__":
    try:
        training_pipeline=TrainingPipeline()
        model_trainer_artifact=training_pipeline.run_pipeline()
        print(model_trainer_artifact)
    except Exception as e:
        raise customException(e,sys)
    