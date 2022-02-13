import findspark
findspark.init()
from pyspark.sql.types import StringType
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
from pyspark.sql.functions import array, lit, col
import logging
import datetime
import traceback
import pandas as pd
import src.global_var as g_var


class DataLakeFetch:
    """
        A class containing all the functions required to fetch data from datalake
    """

    logger = logging.getLogger(__name__)

    def create_spark_session(self):
        """
            Function to create spark session
        params: none
        :return: spark session object
        """
        self.logger.info("Creating Spark Session from the yarn configuration")
        SparkContext.setSystemProperty('spark.sql.crossJoin.enabled', 'true')

        try:
            spark_session = SparkSession.builder.getOrCreate()
        except Exception as e:
            self.logger.critical("Unable to create spark session and the exception is " + str(e) +
                                 " Traceback is " + traceback.format_exc())
            spark_session = None

        return spark_session

    def get_data_from_spark(self, spark_session, parquet_file_path):
        """
            Generic Function to get data from parquet file
        :param spark_session:
        :param parquet_file_path:
        :param pplcd_list:
        :param table_name:
        :return:
        """
        self.logger.info("Reading Parquet file for " + str(parquet_file_path))
        try:
            spark_df = spark_session.read.parquet(parquet_file_path)
        except Exception as e:
            self.logger.error(
                "Unable to read data from parquet file " + str(parquet_file_path) + " exception is " + str(e) +
                " Traceback is " + traceback.format_exc())
            spark_df = None

        return spark_df

    def filter_pplcd_data(self, spark_session, spark_df, pplcd_list, table_name, ppl_cd_colname):
        """
            Function to filter data with respect to people code list
        :param spark_df:
        :param table_name:
        :return:
        """
        self.logger.info("Filtering DataFrame for " + str(table_name))
        if spark_df is None:
            self.logger.warning("Empty Spark Dataframe ")
            return None
        try:
            #filtered_data = spark_df.where(col(ppl_cd_colname).isin(pplcd_list))
            pplcd_spark_df = spark_session.createDataFrame(pplcd_list, schema=StringType()).withColumnRenamed("value",
                                                                                                              "PPLCD")
            filtered_data = spark_df.join(pplcd_spark_df, spark_df.PPL_CD == pplcd_spark_df.PPLCD)

        except Exception as e:
            self.logger.error("Unable to filter data from spark dataframe for " + str(table_name)
                              + " exception is " + str(e) + " Traceback is " + traceback.format_exc())
            filtered_data = None

        return filtered_data

    def convert_to_pandas_df(self, spark_df):

        self.logger.info("Converting spark dataframe to pandas")
        if spark_df is None:
            self.logger.warning("Empty Spark Dataframe")
            df = pd.DataFrame()
        try:
            df = spark_df.toPandas()
        except Exception as e:
            self.logger.error("Unable to convert spark dataframe to pandas dataframe")
            df = pd.DataFrame()

        return df

    def datalake_fetch_main(self, pplcd_list):

        datalake_data = {}

        self.logger.info("Calling function to create spark session")
        spark_session = self.create_spark_session()

        if spark_session is None:
            self.logger.warning("Unable to create spark session")
            return datalake_data

        for file in g_var.DATALAKE_FILE_LIST:

            path = g_var.DATALAKE_PATH + file

            self.logger.info("Calling function to read user data")
            spark_df = self.get_data_from_spark(spark_session, path)

            self.logger.info("Calling Function to filter user data")
            spark_filtered_df = self.filter_pplcd_data(spark_session, spark_df, pplcd_list, path, g_var.DATALAKE_PPLCD_COLNAME)

            self.logger.info("Converting the filtered users data to pandas dataframe")
            pandas_df = self.convert_to_pandas_df(spark_filtered_df)

            datalake_data[file] = pandas_df.to_dict('records')

        return datalake_data
