import os
from jproperties import Properties

class Config:
    def __init__(self, file_path="../../config.properties"):
        file_path = os.path.join(os.path.dirname(__file__), file_path)
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError("Error: Config file does not exist. Please check the README.md file at the root of the project for more information.")
        
        self.properties = Properties()
        with open(file_path, 'rb') as config_file:
            self.properties.load(config_file)

    def getDatasetsPath(self):
        datasets_path = self.properties.get("datasets_path")
        if datasets_path is None:
            raise KeyError("Error: 'datasets_path' property does not exist in the config file. Please check the README.md file at the root of the project for more information.")
        
        datasets_path = datasets_path.data.strip()
        if not os.path.exists(datasets_path):
            raise FileNotFoundError("Error: Datasets directory does not exist. Make sure the path in the 'datasets_path' property is correct. Please check the README.md file at the root of the project for more information.")
        
        return datasets_path

    def getDatasets(self):
        datasets_string = self.properties.get("datasets_list")
        if datasets_string is None:
            raise KeyError("Error: 'datasets_list' property does not exist in the config file.")
        
        datasets_string = datasets_string.data.strip()
        datasets_list = datasets_string.split(',')

        # Keep in the list only the datasets that correspond to a directory found in the datasets directory
        datasets_path = self.getDatasetsPath()
        datasets_list = [dataset for dataset in datasets_list if os.path.exists(os.path.join(datasets_path, dataset))]

        if len(datasets_list) != len(datasets_string.split(',')):
            print("Warning: Some datasets listed in the 'datasets_list' property do not exist in the datasets directory.")

        return [dataset.strip() for dataset in datasets_list]
    
    # Add getFormattedDatasetsPath method here, if it is DEFAULT then path is <directory containing this config file>/twibench/formatted_datasets
    def getFormattedDatasetsPath(self):
        formatted_datasets_path = self.properties.get("formatted_datasets_path")
        if formatted_datasets_path is None:
            raise KeyError("Error: 'formatted_datasets_path' property does not exist in the config file.")
        
        formatted_datasets_path = formatted_datasets_path.data.strip()
        if formatted_datasets_path == "DEFAULT":
            formatted_datasets_path = os.path.join(self.getProjectRootPath(), "twibench/formatted_datasets")

        if not os.path.exists(formatted_datasets_path):
            raise FileNotFoundError("Error: Formatted datasets directory does not exist. Make sure the path in the 'formatted_datasets_path' property is correct. Please check the README.md file at the root of the project for more information.")
        
        return formatted_datasets_path
    
    # Retourner les trois configurations (datasets_path, datasets_list, formatted_datasets_path) dans trois variables différentes
    def getDatasetsConfig(self):
        return self.getDatasetsPath(), self.getDatasets(), self.getFormattedDatasetsPath()
    
    def getProjectRootPath(self):
        return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
if __name__ == "__main__":
    config = Config()
    config.getDatasetsPath()
    config.getDatasets()
    config.getFormattedDatasetsPath()