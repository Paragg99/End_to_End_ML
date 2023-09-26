import os
import sys
from src.MLProject.exception import CustomException
from src.MLProject.logger import logging
import pandas as pd
import pymysql
import dotenv 

dotenv.load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")



def read_sql_data():
    logging.info("Reading SQL database Started")
    try:
         mydb=pymysql.connect(
           host=host,
           user=user,
           password=password,
           db=db  
         )
         logging.info("Connection established",mydb)
         df= pd.read_sql_query("select * from students", mydb)
         print(df.head())

         return df



    except Exception as e:
        raise CustomException(e,sys)