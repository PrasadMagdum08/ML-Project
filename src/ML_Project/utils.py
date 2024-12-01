import os
import sys
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd
import pymysql
from mysql import connector
import traceback

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from dotenv import load_dotenv

import pickle

load_dotenv()

host =os.getenv('host')
user = os.getenv('user')
password = os.getenv('password')
db = os.getenv('db')


def read_sql_data():

    logging.info('Reading SQL database started.')

    try:
        mydb = connector.connect(
            host = host,
            user = user,
            password = password,
            db = db
        )

        logging.info(f'SQl database connectivity successful: {mydb}')

        df = pd.read_sql_query('Select * from students', mydb)
        print(df.head())

        return df
    
    except Exception as e:
        raise CustomException(e, sys)
    

def save_object(file_path, obj):

    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        # print(traceback.format_exc())
        raise CustomException(e, sys)
    


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            params = param[list(models.keys())[i]]

            gs = GridSearchCV(model, params, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(** gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

            return report


    except Exception as e:
        raise CustomException(e, sys)