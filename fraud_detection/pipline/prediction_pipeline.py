import sys
from fraud_detection.exception import ExceptionHandle
from fraud_detection.logger import logging
from fraud_detection.components.prediction import Prediction
from pandas import DataFrame

class PredictionPipeline:
    def __init__(self):
        self.prediction = Prediction()

    def initiate_prediction_pipeline(self, df: DataFrame) -> DataFrame:
        try:
            isvalidation = self.prediction.PredictDataValidation(df)
            if isvalidation:
                result = self.prediction.ModelPrediction(df)
                return result
            else:
                logging.error("Data validation failed")
                return None

        except Exception as e:
            raise ExceptionHandle(e, sys)