import sys
import os
import numpy as np
import pandas as pd
from src.ML_Project.exception import CustomException
from src.ML_Project.logger import logging
from src.ML_Project.utils import save_object

from dataclasses import dataclass

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation(self):

        try:
            numerical_features = ['writing_score', 'reading_score']

            categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )


            cat_pipeline = Pipeline(
                steps=[
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categoriacal and Numerical transformation completed.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipelines', num_pipeline, numerical_features),
                    ('cat_pipelines', cat_pipeline, categorical_features)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            preprocessor_obj = self.get_data_transformation()

            target_feature = "math_score"

            """Divide the train dataset to independent and dependent feature"""

            input_feature_train_df = train_df.drop(columns=[target_feature], axis=1)
            target_feature_train_df = train_df[target_feature]

            """Divide the test dataset to independent and dependent feature"""

            input_feature_test_df = test_df.drop(columns=[target_feature], axis=1)
            target_feature_test_df = test_df[target_feature]

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )


        except Exception as e:
            raise CustomException(e, sys)