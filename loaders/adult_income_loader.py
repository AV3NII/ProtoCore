import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  OneHotEncoder, MinMaxScaler

from .base_data_loader import BaseDataLoader

class AdultIncomeDataLoader(BaseDataLoader):
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state

    def ensure_root(self, project_root_indicator = "/ProtoCore"):
        """Navigate to the project root directory if not already there.

            Args:
            - project_root_indicator (str): The file or directory that indicates the project root. Defaults to '.project_root'.

            Returns:
            - str: The path to the project root directory, or None if not found.
            """
        # Start from the current directory
        current_dir = '.'
        while not os.path.exists(os.path.join(current_dir, project_root_indicator)):
            # Move to the parent directory
            parent_dir = os.path.dirname(os.path.abspath(current_dir))
            if parent_dir == current_dir:
                print("Not in the project directory.")
                return None
            current_dir = parent_dir
        os.chdir(current_dir)
        return os.path.abspath(current_dir)

    def load_data(self):
        # Use the local path instead of the URL
        print("loading while in", self.ensure_root())
        local_path = os.path.join(os.getcwd(), "data", "adult.data.csv")

        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]

        data = pd.read_csv(local_path, names=column_names, skipinitialspace=True)
        data = data.replace('?', np.nan).dropna()

        X = data.drop('income', axis=1)
        y = data['income'].apply(lambda x: 1 if x == '>50K' else 0)

        numerical_cols = X.select_dtypes(include=['int64']).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

        min_max_scaler = MinMaxScaler()
        X[numerical_cols] = min_max_scaler.fit_transform(X[numerical_cols])

        onehotencoder = OneHotEncoder(handle_unknown='ignore')
        res = onehotencoder.fit_transform(X[categorical_cols])
        new_cols = onehotencoder.get_feature_names_out(categorical_cols)
        one_h = pd.DataFrame(res.toarray(), columns=new_cols, index=X.index)

        X_preprocessed = pd.concat([X, one_h], axis=1)
        X_preprocessed.drop(categorical_cols, axis=1, inplace=True)

        feature_names = X_preprocessed.columns.tolist()

        input_dim = X_preprocessed.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X_preprocessed, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        # Convert np.arrays to tensors
        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

        self.additional_info = {
            "onehot_encoder": onehotencoder,
            "scaler": min_max_scaler,
            'numerical_cols': numerical_cols,
            'categorical_cols': categorical_cols,
            'processed_column_names': feature_names, # This is the list of all columns after one-hot encoding
            'original_column_names': column_names,
            'n_classes': 2,
            'problem_type': 'binary'
        }

        return X_train, X_test, y_train, y_test, input_dim, self.additional_info

    def process_x(self, x, num_cols, cat_cols, **encoders):
        if num_cols is None:
            num_cols = self.additional_info["numerical_cols"]
        if cat_cols is None:
            cat_cols = self.additional_info["categorical_cols"]
        if encoders is None or len(encoders) == 0:
            encoders = {
                "onehot_encoder": self.additional_info["onehot_encoder"],
                "scaler": self.additional_info["scaler"]
            }

        for encoder_name, encoder in encoders.items():
            if encoder_name == "onehot_encoder":
                res = encoder.transform(x[cat_cols])
                new_cols = encoder.get_feature_names_out(cat_cols)
                one_h = pd.DataFrame(res.toarray(), columns=new_cols, index=x.index)
            elif encoder_name == "scaler":
                x[num_cols] = encoder.transform(x[num_cols])
        x = x.drop(cat_cols, axis=1, inplace=True)
        x = pd.concat([x, one_h], axis=1)
        return x

