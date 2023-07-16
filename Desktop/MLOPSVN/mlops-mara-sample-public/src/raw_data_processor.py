import argparse
import logging
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from problem_config import ProblemConfig, ProblemConst, get_prob_config


class RawDataProcessor:
    @staticmethod
    def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        for col in categorical_cols:
            apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df

    @staticmethod
    def process_raw_data(prob_config: ProblemConfig):
        logging.info("start process_raw_data")
        training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data, category_index = RawDataProcessor.build_category_features(
            training_data, prob_config.categorical_cols
        )

        
        
        train, dev = train_test_split(
            training_data,
            test_size=prob_config.test_size,
            random_state=prob_config.random_state,
        )

        with open(prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)

        target_col = prob_config.target_col
        train_x = train.drop([target_col], axis=1)
        train_y = train[[target_col]]
        #train_y = le.fit_transform(train_y)
        test_x = dev.drop([target_col], axis=1)
        test_y = dev[[target_col]]
        #test_y = le.fit_transform(test_y)


        train_x.to_parquet(prob_config.train_x_path, index=False)
        train_y.to_parquet(prob_config.train_y_path, index=False)
        test_x.to_parquet(prob_config.test_x_path, index=False)
        test_y.to_parquet(prob_config.test_y_path, index=False)
        logging.info("finish process_raw_data")

    @staticmethod
    def load_train_data(prob_config: ProblemConfig):
        le = preprocessing.LabelEncoder()
        train_x_path = prob_config.train_x_path
        train_y_path = prob_config.train_y_path
        train_x = pd.read_parquet(train_x_path)
        train_y = pd.read_parquet(train_y_path)
        logging.info("Shape of Train Y", train_y.shape)
        if (train_y[prob_config.target_col].nunique()) > 2:
            train_y = le.fit_transform(train_y)
            return train_x, train_y
        logging.info("Train Y", train_y)
        # new_dict = {'Denial of Service': 0,
        #              'Exploits': 1,
        #               'Information Gathering': 2,
        #                'Malware' : 3,
        #                'Normal': 4,
        #                 'Other':5}
        # if (train_y.nunique()) > 2:
        #     train_y.replace(new_dict)

        
        return train_x, train_y[prob_config.target_col]

    @staticmethod
    def load_test_data(prob_config: ProblemConfig):
        le = preprocessing.LabelEncoder()
        dev_x_path = prob_config.test_x_path
        dev_y_path = prob_config.test_y_path
        dev_x = pd.read_parquet(dev_x_path)
        dev_y = pd.read_parquet(dev_y_path)
        if (dev_y[prob_config.target_col].nunique()) > 2:
            dev_y = le.fit_transform(dev_y)
            return dev_x, dev_y
        # new_dict = {'Denial of Service': 0,
        #              'Exploits': 1,
        #               'Information Gathering': 2,
        #                'Malware' : 3,
        #                'Normal': 4,
        #                 'Other':5}
        # if (dev_y.nunique()) > 2:
        #     dev_y.replace(new_dict)

        return dev_x, dev_y[prob_config.target_col]

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def load_capture_data(prob_config: ProblemConfig):
        captured_x_path = prob_config.captured_x_path
        captured_y_path = prob_config.uncertain_y_path
        captured_x = pd.read_parquet(captured_x_path)
        captured_y = pd.read_parquet(captured_y_path)
        return captured_x, captured_y[prob_config.target_col]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-id", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id", type=str, default=ProblemConst.PROB2)
    parser.add_argument("--phase-id-2", type=str, default=ProblemConst.PHASE2)
    parser.add_argument("--prob-id-2", type=str, default=ProblemConst.PROB2)
    args = parser.parse_args()

    prob_config = get_prob_config(args.phase_id, args.prob_id)
    RawDataProcessor.process_raw_data(prob_config)

    prob_config_2 = get_prob_config(args.phase_id_2, args.prob_id_2)
    RawDataProcessor.process_raw_data(prob_config_2)
