import pandas as pd
import urllib.request
from scipy.io import arff                                
import io
from pathlib import Path
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

def load_datasets(data_name):
    if data_name == "hepatitis":
        path_hepatitis = Path("./datasets/hepatitis_data.csv.gz")
        if path_hepatitis.is_file():
            data = pd.read_csv(path_hepatitis, compression='gzip')
        else:
            data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/hepatitis/hepatitis.data", header=None)
            data.to_csv(path_hepatitis, compression='gzip', index=False)

    elif data_name == "diabetes":
        path_diabetes = Path("./datasets/diabetes_data.csv.gz")
        if path_diabetes.is_file():
            data = pd.read_csv(path_diabetes, compression='gzip')
        else:
            url='https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff'
            ddd=urllib.request.urlopen(url)
            data, meta = arff.loadarff(io.StringIO(ddd.read().decode('utf-8')))
            data = pd.DataFrame(data)
            data["Class"] = data["Class"].astype(int)
            data.to_csv(path_diabetes, compression='gzip', index=False)
    return data


class data_preprocessing():
    def __init__(self, data):
        self.data = data
        
    def replace_missing_values(self):
        self.data.replace('?', np.NaN, inplace=True)
        # for i in data.columns:
        #     column_i = pd.to_numeric(data[i], errors="coerce")
        #     column_mean = column_i.mean()
        #     data[i].fillna(value=column_mean, inplace=True)
        missing_props = self.data.isna().mean(axis=0)
        over_threshold = missing_props[missing_props >= 0.4]
        self.data.drop(over_threshold.index, 
                axis=1, 
                inplace=True)
        return self

    def split_to_X_and_y(self, label_col):
        self.X = self.data.drop(label_col, axis=1)
        self.y = self.data[label_col].astype(int)
        a = self.y.unique()
        a = [int(a1) for a1 in a]
        if max(a) != 1:
            self.y = self.y - (max(a) - 1)
        return self


    def transform_X_and_y(self):
        numeric_pipeline = Pipeline(
            steps=[("impute", SimpleImputer(strategy="mean")), 
                ("scale", StandardScaler())]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("oh-encode", OneHotEncoder(handle_unknown="ignore", sparse=False)),
            ]
        )

        cat_cols = self.X.select_dtypes(exclude="number").columns
        num_cols = self.X.select_dtypes(include="number").columns

        full_processor = ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, num_cols),
                ("categorical", categorical_pipeline, cat_cols),
            ]
        )

        self.X_processed = full_processor.fit_transform(self.X)
        self.y_processed = SimpleImputer(strategy="most_frequent").fit_transform(
            self.y.values.reshape(-1, 1)
        )
        return self

    def split_train_test(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_processed, self.y_processed, stratify=self.y_processed, 
            random_state=1121218
        )
        return X_train, X_test, y_train, y_test