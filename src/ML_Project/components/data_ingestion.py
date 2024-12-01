import os
import sys
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
import pandas as pd

from dataclasses import dataclass
from src.ML_Project.utils import read_sql_data

from sklearn.model_selection import train_test_split

@dataclass
class DataIngestionConfig:

    """Datalass use for creating a class instance without using init method."""

    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    data_data_path = os.path.join('artifacts', 'data.csv')
    logging.info('Create artificats folder.')


class DataIngestion:

    """Initialize data ingestion config class into a variable."""

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        
        try:
            # Reading code
            df = read_sql_data()
            logging.info('Reading from mysql database.')

            os.makedirs(os.path.dirname(self.data_ingestion_config.data_data_path), exist_ok=True)

            df.to_csv(self.data_ingestion_config.data_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.20, random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info('Data Ingestion is completed.')

            """Returing train and test data for the data transformation."""
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path,

            
            )

        except Exception as e:
            raise CustomException(e, sys)

