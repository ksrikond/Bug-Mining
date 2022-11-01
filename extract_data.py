"""
Data Extraction for HSD Mining Analysis Application

This file contains Data Extraction script. Use config.yaml file to pass query payload, datafile name and format.
"""
import bz2
import csv
import os
import json
import pprint

import _pickle as cPickle
import src.utils.hsd_util as _util

import requests
import urllib3

from requests_kerberos import HTTPKerberosAuth
from src.data.hsd_data import HsdDataObject
from src.data.static_attributes import QUERY_DICT
from src.utils.logger_config import api_logger

cfg = _util.load_configuration('config.yaml')

root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')


class Extraction:
    """ HSD Mining Analysis Data Extraction """

    @staticmethod
    def extract_data(filename, encode=True):
        """Extract data and send it into a format desired

        Parameters
        ----------
        filename: str
            raw data filename to save json into

        encode: bool, default=True
            True if desired format is compressed pickle, False if csv

        Returns
        ----------
        hsd_obj_list : List of HsdDataObject
        """
        api_logger.info("*****HSD data extraction process started*****")

        # this is to ignore the ssl insecure warning as we are passing in 'verify=false'
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        try:
            api_logger.info("Requesting {0} from {1}".format(filename, cfg['extraction']['url']))
            response = requests.post(cfg['extraction']['url'], verify=False, auth=HTTPKerberosAuth(), headers=cfg['extraction']['headers'],
                                     data=QUERY_DICT[cfg['extraction']['queryPayload']], params=cfg['extraction']['params'])
            response.raise_for_status()

            api_logger.info("Successfully fetched data")

            data_rows = response.json()['data']

            filename = os.path.join(datafiles_path, 'raw_data', filename)
            if encode:
                Extraction.json_to_encoded(data_rows, filename)
            else:
                Extraction.json_to_csv(data_rows, filename)

            hsd_obj_list = []
            for row in data_rows:
                # pprint.pprint(row)
                hsd_obj = HsdDataObject(**row)
                hsd_obj_list.append(hsd_obj)
                api_logger.info("Logging HSD object details, ID: {0} Title: {1}".format(hsd_obj.id, hsd_obj.title))

            return hsd_obj_list
        except requests.exceptions.HTTPError as error:
            api_logger.error("HTTP error {0} occurred while request processing: {1}".format(error.response.status_code, error.response.text))
            raise
        except requests.exceptions.RequestException as error:
            api_logger.error("Exception occurred while request processing: {0}".format(error))
            raise

    @staticmethod
    def json_to_csv(data, filename):
        """Write json into a csv file

        Parameters
        ----------
        data: json data

        filename: str
            raw data filename to save json to
        """
        api_logger.info("Saving raw data to csv at {0}".format(filename))

        data_file = open(filename + '.csv', 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(data_file)

        count = 0
        for row in data:
            if count == 0:
                header = row.keys()
                # print(row.keys())
                csv_writer.writerow(header)
                count += 1
            csv_writer.writerow(row.values())
        data_file.close()

    @staticmethod
    def json_to_encoded(data, filename):
        """Write json into a compressed pickle file

        Parameters
        ----------
        data: json data

        filename: str
            raw data filename to save json to
        """
        api_logger.info("Saving raw data to pickle file at {0}".format(filename))

        data_file = bz2.BZ2File(filename + '.pickle', 'w')
        cPickle.dump(data, data_file)
        data_file.close()

    @staticmethod
    def read_from_json(filename):
        """Read json from a file (.json file)

        Parameters
        ----------
        filename: str
            raw data filename to read json from

        Returns
        ----------
        data : json data object
        """
        api_logger.info("Reading json data from {0}".format(filename))

        with open(filename) as file:
            data = json.load(file)
        return data


class Main:
    if __name__ == '__main__':
        results_data = Extraction.extract_data(filename=cfg['extraction']['output']['filename'],
                                               encode=cfg['extraction']['output']['encode'])
        if not results_data:
            api_logger.info("No records found!!")
        else:
            api_logger.info(
                "Data extraction process complete. Numbers of records fetched: {0}".format(len(results_data)))

