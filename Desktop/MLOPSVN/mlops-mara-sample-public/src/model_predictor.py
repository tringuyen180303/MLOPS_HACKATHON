import argparse
import logging
import os
import random
import time

import mlflow
import pandas as pd
import uvicorn
import yaml
from fastapi import FastAPI, Request
from pandas.util import hash_pandas_object
from pydantic import BaseModel

from problem_config import ProblemConst, create_prob_config, get_prob_config
from raw_data_processor import RawDataProcessor
from utils import AppConfig, AppPath

from evidently.report import Report
from evidently import ColumnMapping


from evidently import dashboard
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.dashboard.tabs import DataDriftTab, CatTargetDriftTab
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric

from evidently.dashboard import Dashboard

PREDICTOR_API_PORT = 8000


class Data(BaseModel):
    id: str
    rows: list
    columns: list


class ModelPredictor:
    def __init__(self, config_file_path):
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")
        #logging.info("URI: ", AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        
        #logging.info("phase_id", self.config['phase_id'])
        #logging.info("prob_id", self.config['prob_id'])

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load model
        model_uri = os.path.join(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        )
        self.model = mlflow.pyfunc.load_model(model_uri)

    def detect_drift(self,feature_df) -> int:
        # watch drift between coming requests and training data
        drift_report = Report(metrics=[DatasetDriftMetric()])
        
        # get reference_df from train_data
        train_x, train_y = RawDataProcessor.load_train_data(self.prob_config)
        
        # No need to transfer train_x and train_y to numpy ndarray
        #train_x = train_x.to_numpy()
        #train_y = train_y.to_numpy()
        #reference_df = pd.concat([train_x, train_y], axis=1)

        # reference_df is only data not including label
        reference_df = train_x
        print(reference_df)
        print(feature_df)
        logging.info(feature_df.shape)
        logging.info(reference_df.shape)
        print(reference_df[:10])

        column_mapping = ColumnMapping()
        # problem_1
        cols_cat = ["feature1", "feature2"]
        cols_num = ["feature3", "feature4", "feature5", 
        "feature6", "feature7", "feature8", "feature8", 
        "feature9", "feature10", "feature11", "feature12", 
        "feature13", "feature14", "feature15", "feature16"]

        # problem_2
        num_cols = ["feature2", "feature5", "feature13", "feature18"]
        cat_cols = [
    "feature1", "feature3", "feature4", "feature6", "feature7",
    "feature8", "feature9", "feature10", "feature11", "feature12",
    "feature14", "feature15", "feature16", "feature17", "feature19", "feature20"]

        column_mapping.categorical_features = cat_cols
        column_mapping.numerical_features = num_cols
        drift_report.run(current_data=feature_df, reference_data=reference_df) #column_mapping=column_mapping)

        
        #logging.info("drift results", drift_results)
        drift_results = drift_report.as_dict()
        print("Drift results", drift_results)
        drift_score = drift_results["metrics"][0]["result"]["dataset_drift"]
        # Set a threshold and return 1 if drift_score exceeds the threshold, otherwise 0
        #drift_threshold = 0.8
        #if drift_score > drift_threshold:
        #    return 1
        #return 0

        if drift_score == True:
            return 1
        return 0 


    def predict(self, data: Data):
        start_time = time.time()

        # preprocess
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )
        # save request data for improving models
        ModelPredictor.save_request_data(
            feature_df, self.prob_config.captured_data_dir, data.id
        )

        prediction = self.model.predict(feature_df)
        logging.info("prediction", prediction)
        is_drifted = self.detect_drift(feature_df)

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction takes {run_time} ms")
        return {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

    @staticmethod
    def save_request_data(feature_df: pd.DataFrame, captured_data_dir, data_id: str):
        if data_id.strip():
            filename = data_id
        else:
            filename = hash_pandas_object(feature_df).sum()
        output_file_path = os.path.join(captured_data_dir, f"{filename}.parquet")
        feature_df.to_parquet(output_file_path, index=False)
        return output_file_path


class PredictorApi:
    def __init__(self, predictor: ModelPredictor, predictor_2: ModelPredictor):
        self.predictor = predictor
        self.predictor_2 = predictor_2
        self.app = FastAPI()

        @self.app.get("/")
        async def root():
            return {"message": "hello"}

        @self.app.post("/phase-2/prob-1/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor.predict(data)
            logging.warning("Response for prob1", response)
            print(response)
            self._log_response(response)
            return response

        @self.app.post("/phase-2/prob-2/predict")
        async def predict(data: Data, request: Request):
            self._log_request(request)
            response = self.predictor_2.predict(data)
            logging.warning("prob2", response) 
            print(response)
            self._log_response(response)
            return response
    @staticmethod
    def _log_request(request: Request):
        logging.info(request)

    @staticmethod
    def _log_response(response: dict):
        logging.info(response)

    def run(self, port):
        uvicorn.run(self.app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    default_config_path = (
        AppPath.MODEL_CONFIG_DIR
        / ProblemConst.PHASE2
        / ProblemConst.PROB1
        / "model-1.yaml"
    ).as_posix()

    default_config_path_2 = (
        AppPath.MODEL_CONFIG_DIR
        # ProbblemConst.PHASE1 = "phase-1"
        / ProblemConst.PHASE2
        # ProbblemConst.PROB1 = "prob-1"
        / "prob-2"
        / "model-1.yaml"
    ).as_posix()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default=default_config_path)
    parser.add_argument("--config-path-2", type=str, default = default_config_path_2)
   # parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
   # parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
   # parser.add_argument("--phase-id-2", type=str, default=ProblemConst.PHASE1)
   # parser.add_argument("--prob-id-2", type=str, default=ProblemConst.PROB1)

    parser.add_argument("--port", type=int, default=PREDICTOR_API_PORT)
    args = parser.parse_args()

    #problem_config = get_prob_config(args.phase_id, args.prob_id)
    #problem_config_2 = get_prob_config(args.phase_id_2, args.prob_id_2)

    predictor = ModelPredictor(config_file_path=args.config_path) #problem_config=problem_config)
    predictor_2 = ModelPredictor(config_file_path=args.config_path_2) # problem_config=problem_config_2)

    api = PredictorApi(predictor, predictor_2)
    #api.pred
    api.run(port=args.port)
