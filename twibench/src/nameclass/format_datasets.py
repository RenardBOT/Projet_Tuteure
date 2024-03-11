import pandas as pd
import os
import sys
from tqdm import tqdm

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config


def get_subdatasets_list(dataset_path, bot_list, human_list):
    subdatasets = os.listdir(dataset_path)
    return [subdataset for subdataset in subdatasets if subdataset in bot_list or subdataset in human_list]

def process_dataset(dataset, dataset_path, nameclass_path, bot_list, human_list):
    output_path = os.path.join(nameclass_path, f"{dataset}-full.csv")
    subdatasets = get_subdatasets_list(dataset_path, bot_list, human_list)
    
    for subdataset in tqdm(subdatasets, desc=f"Processing {dataset}"):
        output_path = os.path.join(nameclass_path, f"{dataset}_{subdataset}.csv")
        subdataset_path = os.path.join(dataset_path, subdataset)
        users_path = os.path.join(subdataset_path, "users.csv")
        users_df = pd.read_csv(users_path)
        users_df["label"] = "HUMAN" if subdataset in human_list else "BOT"
        users_df = users_df[["screen_name", "label"]]
        users_df.to_csv(output_path, index=False, header=True)

if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    nameclass_path = os.path.join(formatted_datasets_path, "nameclass")

    for dataset in datasets_list:
        dataset_path = os.path.join(datasets_path, dataset)
        print("CHARGEMENT DU DATASET :", dataset)
        
        if dataset == "cresci-2015":
            bot_list = ["INT", "FSF", "TWT"]
            human_list = ["TFP", "E13"]
            process_dataset(dataset, dataset_path, nameclass_path, bot_list, human_list)

        elif dataset == "cresci-2017":
            bot_list = ["social_spambots_1", "social_spambots_2", "social_spambots_3"]
            human_list = ["genuine_accounts"]
            process_dataset(dataset, dataset_path, nameclass_path, bot_list, human_list)
