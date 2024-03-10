import pandas as pd
import os
import sys

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv

if __name__ == '__main__':
    datasetsPath = Config().getDatasetsPath()
    datasets = Config().getDatasets()

    print("datasetsPath : ", datasetsPath)

    # all dirs in datasetsPath
    datasetsList = os.listdir(datasetsPath)
    print("datasetsList : ", datasetsList)