import sys
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging

from src.ML_Project.components.data_ingestion import DataIngestion
from src.ML_Project.components.data_transformation import DataTransformation
from src.ML_Project.components.model_trainer import ModelTrainer


if __name__ == "__main__":

    try:
        dataIngestion = DataIngestion()
        train_data, test_data = dataIngestion.initiate_data_ingestion()

        dataTransformation = DataTransformation()
        train_arr, test_arr,_ = dataTransformation.initiate_data_transformation(train_data, test_data)

        modelTrainer = ModelTrainer()
        model_done = modelTrainer.initiate_model_trainer(train_arr, test_arr)

        print(model_done)

    except Exception as e:
        raise CustomException(e, sys)