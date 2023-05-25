import pandas as pd
import urllib.request
from scipy.io import arff                                
import io
import xgboost as xgb
from data_preprocessing import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV

class xgb_classifier():
    def __init__(self, model):
        self.model = model

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def test(self, X_test):
        preds = self.model.predict(X_test)
        return preds
    
    def accuracy(self, y_true, y_pred):
        return accuracy_score(y_true, y_pred)


if __name__ == "__main__":

    data_names = ["hepatitis", "diabetes"]
    label_cols = ["0", "Class"]

    param_grid = {
    "max_depth": [9, 10, 12],
    "learning_rate": [0.15, 0.1, 0.01],
    "gamma": [0, 0.25, 0.4, 0.5],
    "reg_lambda": [2, 2.5, 3, 3.5],
    "scale_pos_weight": [2, 3, 4],
    "subsample": [0.4, 0.8, 1],
    "colsample_bytree": [0.7, 0.8, 0.9, 1],
    }


    for i in range(1,2):
        raw_data = load_datasets(data_names[i])
        processing = data_preprocessing(raw_data)
        X_train, X_test, y_train, y_test = processing.replace_missing_values().split_to_X_and_y(label_cols[i]).transform_X_and_y().split_train_test()
        # print(X_test)
        xgb_cl = xgb.XGBClassifier()
        grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="accuracy")
        grid_cv.fit(X_train, y_train)
        print(grid_cv.best_score_)
        print(grid_cv.best_params_)


        xgb_cl = xgb.XGBClassifier(**grid_cv.best_params_)
        classifier = xgb_classifier(xgb_cl)
        classifier.train(X_train, y_train)
        preds_test = classifier.test(X_test)
        preds_train = classifier.test(X_train)

        print("train acc for ", data_names[i], ":",classifier.accuracy(y_train, preds_train))
        print("test acc for ", data_names[i], ":", classifier.accuracy(y_test, preds_test))