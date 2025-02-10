from fraud_detection.pipline.training_pipeline import TrainPipeline

obj = TrainPipeline()
obj.run_pipeline()


# from fraud_detection.pipline.prediction_pipeline import PredictionPipeline

# obj = PredictionPipeline()
# res = obj.initiate_prediction_pipeline()
# if res is not None:
#     print(res.columns)
# else:
#     print("Prediction failed")