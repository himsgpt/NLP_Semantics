# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 12:58:23 2019

@author: higupta
"""

from pymongo import MongoClient
import global_var as g_var
import json
import traceback
import logging
import pandas as pd


class Utility:
    '''
        A class containing all the utility functions required in the project
    '''

    logger = logging.getLogger(__name__)

    def init_global_var(self, config_path):
        """
            Initialization method for global variables
            Params:
                - Input: The config file path
        """

        self.logger.info("Reading the json configuration file")
        # Read the jsong config file
        self.read_json_config(config_path)
        self.logger.info("Successfully read the json config file variables")

        self.logger.info("Reading the excel file containing the ontology of the URL to match")
        # Read the Excel file for ontology
        g_var.ACTIONWORDS_DF = self.read_excel_file(g_var.ACTIONWORDS_EXCEL_PATH,
                                                    g_var.ACTIONWORDS_EXCEL_SHEET_NAME,
                                                    g_var.ACTIONWORDS_EXCEL_COLUMNS_FILTER)

    def read_json_config(self, path):
        """
            Function to read json config file

            Params:
                path - Absolute path of the config file to read
        """

        try:
            self.logger.info("Trying to open the json config file")
            with open(path) as file:
                self.logger.info("Loading the json file")
                settings = json.loads(file.read())

                self.logger.info("Reading the json config file variables")

                g_var.MONGODB_SERVER = settings['Mongo_Settings']['MONGODB_SERVER']
                g_var.MONGODB_PORT = settings['Mongo_Settings']['MONGODB_PORT']
                g_var.MONGODB_DB = settings['Mongo_Settings']['MONGODB_DB']
                g_var.MONGODB_INPUT_COLLECTION = settings['Mongo_Settings']['MONGODB_INPUT_COLLECTION']
                g_var.MONGODB_OUTPUT_COLLECTION = settings['Mongo_Settings']['MONGODB_OUTPUT_COLLECTION']
                g_var.MONGODB_INPUT_SCRAPER_FLAG = settings['Mongo_Settings']['MONGODB_INPUT_SCRAPER_FLAG']
                g_var.MONGODB_INPUT_WEB_ADDRESS = settings['Mongo_Settings']['MONGODB_INPUT_WEB_ADDRESS']
                g_var.MONGODB_INPUT_WEB_DOMAIN = settings['Mongo_Settings']['MONGODB_INPUT_WEB_DOMAIN']

        except Exception as e:
            self.logger.error("Could Not Read config File with path " + path
                              + " exception is " + str(e) +
                              " Traceback is " + traceback.format_exc())
        else:
            self.logger.info(" Successfully read the config file with path " + path)

    def read_excel_file(self, path, sheet_name, columns_filter):
        """
            A generic function to read excel file and return the pandas dataframe

            Params:
                path - Absolute path of the excel file to read
                sheet_name - Name of the sheet to read from the excel file
                columns_filter - A list of columns to be filtered from the data

            Output:
                A pandas dataframe
        """

        # Read the data from the excel file
        self.logger.info("Reading the excel file with path " + path
                         + " and Sheet name " + sheet_name)
        df = pd.DataFrame()
        try:
            df = pd.read_excel(path, sheet_name=sheet_name)
        except Exception as e:
            self.logger.error("Could Not Read Excel File Data with path " + path
                              + " and Sheet name " + sheet_name + " exception is " + str(e) +
                              " Traceback is " + traceback.format_exc())
        else:
            self.logger.info(" Successfully read the excel file with path " + path
                             + " and Sheet name " + sheet_name)

        self.logger.info("Filtering the columns from the data")
        try:
            df = df.loc[:, columns_filter]
        except Exception as e:
            self.logger.error("Could not filter the columns from the file exception is" + " exception is " + str(e) +
                              " Traceback is " + traceback.format_exc())
        else:
            self.logger.info(" Successfully filtered the columns from the data ")

        return df

    def connect_to_mogodb(self, server, port):

        self.logger.info("Connection to the mongo database")
        try:
            connection = MongoClient(server, port)
        except Exception as e:
            self.logger.critical(" Unable to Connect to the Mongo Database:" + " exception is " + str(e) +
                                 " Traceback is " + traceback.format_exc())
            connection = None

        return connection

    def close_connection_to_mongodb(self, connection):

        self.logger.info("Closing the connection to MongoDB")
        try:
            connection.close()
        except Exception as e:
            self.logger.warning(" Unable to close connection to the Mongo Database:" + " exception is " + str(e) +
                                " Traceback is " + traceback.format_exc())

    def df_to_json(self, df, path, json_orientation):

        if df.empty:
            self.logger.warning("Empty Data Frame Passed, Writing empty json in path " + str(path)
                                + " with orientation as " + str(json_orientation))
            data = {}
            with open(path, 'w') as outfile:
                json.dump(data, outfile)

            return

        self.logger.info("Writing the dataframe to json in path " + str(path)
                         + " with orientation as " + str(json_orientation))
        try:
            df.to_json(path, orient=json_orientation)
        except Exception as e:
            self.logger.error(" Unable to write database to json file" + " exception is " + str(e) +
                              " Traceback is " + traceback.format_exc())

    def update_json_flag_db(self, connection, db_name, collection_name, unique_id, json_output_path, email_url):

        db = connection[db_name]
        collection = db[collection_name]

        if collection is not None:
            query = {"_id": unique_id}
            new_val = {"$set": {g_var.MONGODB_INPUT_JSON_FLAG: 1,
                                g_var.MONGODB_INPUT_JSON_PATH: json_output_path,
                                g_var.MONGODB_INPUT_EMAIL_URL: email_url}}
            try:
                collection.update_one(query, new_val)
            except Exception as e:
                self.logger.critical("Unable to update to the database" + " exception is " + str(e) +
                                     " Traceback is " + traceback.format_exc())
            else:
                self.logger.info("Successfully updated the record to database ")
        else:
            self.logger.critical("Collection object is None")