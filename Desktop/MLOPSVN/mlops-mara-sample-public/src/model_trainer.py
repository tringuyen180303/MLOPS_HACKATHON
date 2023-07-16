import argparse
import logging
import pickle
import mlflow
import numpy as np
import xgboost as xgb
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, precision_score, recall_score

from problem_config import (
    ProblemConfig,
    ProblemConst,
    get_prob_config,
)
from raw_data_processor import RawDataProcessor
from utils import AppConfig


class ModelTrainer:
    EXPERIMENT_NAME = "xgb-1"

    @staticmethod
    def train_model(prob_config: ProblemConfig, model_params, add_captured_data=False):
        logging.info("start train_model")
        # init mlflow
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(
            f"{prob_config.phase_id}_{prob_config.prob_id}_{ModelTrainer.EXPERIMENT_NAME}"
        )
        logging.info(f"load at port: {AppConfig.MLFLOW_TRACKING_URI}")
        # load train data
        train_x, train_y = RawDataProcessor.load_train_data(prob_config)
        train_x = train_x.to_numpy()
        if not (isinstance(train_y, np.ndarray)):
            train_y = train_y.to_numpy()
        logging.info(f"loaded {len(train_x)} samples")

        if add_captured_data:
            captured_x, captured_y = RawDataProcessor.load_capture_data(prob_config)
            captured_x = captured_x.to_numpy()
            captured_y = captured_y.to_numpy()
            train_x = np.concatenate((train_x, captured_x))
            train_y = np.concatenate((train_y, captured_y))
            logging.info(f"added {len(captured_x)} captured samples")

        # train model
        if len(np.unique(train_y)) == 2:
            objective = "binary:logistic"
        else:
            objective = "multi:softprob"
        
        model = xgb.XGBClassifier(objective=objective, **model_params)
        model.fit(train_x, train_y)

        #Load model in from batch
        #with open("/Users/tringuyen/mlops-mara-sample-public/src/model_1.sav", "rb") as f:
         #   model = pickle.load(f)
        
        # evaluate
        test_x, test_y = RawDataProcessor.load_test_data(prob_config)
        if not (isinstance(test_y, np.ndarray)):
            test_y = test_y.to_numpy()
        predictions = model.predict(test_x)
        #auc_score = roc_auc_score(test_y, predictions, multi_class="ovo", average="micro")
        #auc_score_2 = roc_auc_score(test_y, predictions, multi_class="ovr" )

        averageOfPrediction = ""
        if len(np.unique(train_y)) == 2:
            averageOfPrediction = "binary"
        else:
            averageOfPrediction = "micro"
            #f1 = f1_score(test_y, predictions, average="macro")
        f1 = f1_score(test_y, predictions, average=averageOfPrediction)
        recall = recall_score(test_y, predictions, average=averageOfPrediction)
        precision = precision_score(test_y, predictions, average=averageOfPrediction)
        acc_score = accuracy_score(test_y, predictions)
        metrics = {#"test_auc": auc_score,
                    #"test_auc_2": auc_score_2,
                    "acc_score": acc_score,
                    "f1_score": f1,
                    "recall_score": recall,
                    "precision_score": precision}

        logging.info(f"metrics: {metrics}")

        # mlflow log
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        signature = infer_signature(test_x, predictions)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
        )
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name="phase-1_prob-1_model-1",
            version = 3,
            stage="Production"
        )
        mlflow.end_run()
        logging.info("finish train_model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB1)
    parser.add_argument("--phase-id-2", type=str, default=ProblemConst.PHASE1)
    parser.add_argument("--prob-id-2", type=str, default=ProblemConst.PROB1)
    
    parser.add_argument(
        "--add-captured-data", type=lambda x: (str(x).lower() == "true"), default=False
    )
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    model_config = {"random_state": prob_config.random_state}
    ModelTrainer.train_model(
        prob_config, model_config, add_captured_data=args.add_captured_data
    )

    prob_config_2 = get_prob_config(args.phase_id_2, args.prob_id_2)
    ModelTrainer.train_model(
        prob_config_2, model_config, add_captured_data=args.add_captured_data
    )



