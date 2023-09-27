import sys
from src.MLProject.logger import logging
from src.MLProject.exception import CustomException
from src.MLProject.components.data_ingestion import DataIngestion
from src.MLProject.components.data_ingestion import DataIngestionConfig
from src.MLProject.components.data_transformation import DataTransformationConfig,DataTransformation
from src.MLProject.components.model_trainer import ModelTrainerConfig,ModelTrainer





if __name__ == "__main__":
    logging.info("The execution has started")


    try:
        #data_ingestion_config = DataIngestionConfig()
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        #data_tranformation_config = DataTransformationConfig()
        data_tranformation = DataTransformation()
        train_arr,test_arr,_= data_tranformation.initiate_data_transformation(train_data_path,test_data_path)

        ## Model trainer code
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))


    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)