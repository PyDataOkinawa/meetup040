import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.datasets import load_digits
import numpy as np
import pandas as pd

import pickle
import os
import sys


def load_data(s3_file_path=''):
    d = load_digits()
    data = d['data']

    X = pd.DataFrame(data)
    y = d['target']

    return X, y


def export_mlflow(model, y_pred):
    try:
        model_save_path = os.path.join(os.sep, 'tmp', 'tmp.obj')
        with open(model_save_path, mode='wb') as f:
            pickle.dumps(model)

        for k, v in model.get_params().items():
            mlflow.log_param(k, v)
            
        mlflow.log_metric('accuracy', accuracy_score(y_pred=y_pred, y_true=y_test))
        mlflow.log_metric('precision', precision_score(y_pred=y_pred, y_true=y_test, average='micro'))
        mlflow.log_metric('recall', recall_score(y_pred=y_pred, y_true=y_test, average='micro'))
        mlflow.log_metric('f1_score', f1_score(y_pred=y_pred, y_true=y_test, average='micro'))

        mlflow.set_tag('example', 10)
        mlflow.set_tag('example2', 'test')

        mlflow.log_artifact(model_save_path)
    except Exception as e:
        return False
    
    return True

    
def learn_model(s3_file_path=''):
    # s3のpathが入っているとみなして処理する
    X, y = load_data(s3_file_path)
    val_data = train_test_split(X, y, test_size=0.3, random_state=71)
    X_train, X_test, y_train, y_test = val_data

    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=71)
    rf_clf.fit(X_train, y_train)

    return rf_clf, val_data


if __name__ == '__main__':
    s3_file_path = ''
    model, val_data = learn_model(s3_file_path)
    X_train, X_test, y_train, y_test = val_data

    y_pred = model.predict(X_test)
    export_mlflow(model, y_pred)