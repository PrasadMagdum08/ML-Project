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