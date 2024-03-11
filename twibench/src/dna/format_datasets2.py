import pandas as pd
import os
import sys
from tqdm import tqdm

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def process_dataset(dataset_path, subdatasets_list, dna_path, dataset):
    for subdataset in tqdm(subdatasets_list, desc=f"Processing {os.path.basename(dataset_path)}"):
        subdataset_path = os.path.join(dataset_path, subdataset)
        tweets_path = os.path.join(subdataset_path, "tweets.csv")

        dataframe = pd.read_csv(tweets_path, encoding="ISO-8859-1", low_memory=False)

        dataframe = dataframe[["text", "user_id", "in_reply_to_status_id", "retweeted_status_id", "timestamp"]]
        dataframe = dataframe.dropna(subset=["user_id"])
        dataframe["user_id"] = dataframe["user_id"].astype(float)
        dataframe["user_id"] = dataframe["user_id"].astype(int)

        dataframe["DNA"] = "A"
        dataframe.loc[dataframe["in_reply_to_status_id"] != 0, "DNA"] = "C"
        dataframe.loc[dataframe["retweeted_status_id"] != 0, "DNA"] = "T"

        max_dna_length = 999
        dataframe = dataframe.groupby("user_id").agg({"DNA": lambda x: "".join(x)[:max_dna_length]})
        dataframe = dataframe.reset_index()

        # Si genuine_accounts ajouter colonne label "HUMAN" sinon "BOT"
        if dataset == "cresci-2017" and subdataset == "genuine_accounts":
            dataframe["label"] = "HUMAN"
        if dataset == "cresci-2015" and subdataset in ["TFP", "E13"]:
            dataframe["label"] = "HUMAN"
        else:
            dataframe["label"] = "BOT"

        dna_file_path = os.path.join(dna_path, f"{os.path.basename(dataset_path)}_{subdataset}.csv")
        dataframe.to_csv(dna_file_path, index=False)

if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    dna_path = os.path.join(formatted_datasets_path, "dna")

    if not os.path.exists(dna_path):
        os.makedirs(dna_path)

    for dataset in tqdm(datasets_list, desc="Processing datasets"):
        dataset_path = os.path.join(datasets_path, dataset)
        print("dataset:", dataset)

        if dataset == "cresci-2017":
            subdatasets_list = ["social_spambots_1", "social_spambots_2", "social_spambots_3", "genuine_accounts"]
        elif dataset == "cresci-2015":
            subdatasets_list = ["TFP", "E13", "INT", "FSF", "TWT"]

        process_dataset(dataset_path, subdatasets_list, dna_path, dataset)
