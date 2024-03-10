import pandas as pd
import os
import sys

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv


if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    nameclass_path = os.path.join(formatted_datasets_path, "nameclass")

    for dataset in datasets_list:
        dataset_path = os.path.join(datasets_path, dataset)


        print("dataset : ", dataset)
        

        # -------------------------------------------------------------
        # --------------------- CRESCI-2015-full ----------------------
        # -------------------------------------------------------------

        if dataset == "cresci-2015":

            human_list = ["TFP", "E13"]
            bot_list = ["INT", "FSF", "TWT"]
            cresci_2015_subdatasets = os.listdir(dataset_path)

            output_path = os.path.join(nameclass_path, dataset+"-full.csv")

            for subdataset in cresci_2015_subdatasets:
                subdataset_path = os.path.join(dataset_path, subdataset)
                users_path = os.path.join(subdataset_path, "users.csv")
                users_df = pd.read_csv(users_path)
                screen_names = users_df["screen_name"]
                if subdataset in human_list:
                    label = "HUMAN"
                else:
                    label = "BOT"

                with open(output_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    for screen_name in screen_names:
                        writer.writerow([screen_name, label])

        # -------------------------------------------------------------
        # --------------------- CRESCI-2017-full ----------------------
        # -------------------------------------------------------------

        if dataset == "cresci-2017":
            bot_list = ["social_spambots_1", "social_spambots_2", "social_spambots_3"]
            human_list = ["genuine_accounts"]
            cresci_2017_subdatasets = os.listdir(dataset_path)

            cresci_2017_subdatasets = [subdataset for subdataset in cresci_2017_subdatasets if subdataset in bot_list or subdataset in human_list]

            output_path = os.path.join(nameclass_path, dataset+"-full.csv")

            for subdataset in cresci_2017_subdatasets:
                subdataset_path = os.path.join(dataset_path, subdataset)
                users_path = os.path.join(subdataset_path, "users.csv")
                users_df = pd.read_csv(users_path)
                screen_names = users_df["screen_name"]
                if subdataset in human_list:
                    label = "HUMAN"
                else:
                    label = "BOT"

                with open(output_path, "a", newline='') as f:
                    writer = csv.writer(f)
                    for screen_name in screen_names:
                        writer.writerow([screen_name, label])

        # -------------------------------------------------------------
        # --------------------- CRESCI-2017-small----------------------
        # -------------------------------------------------------------
                        
            if dataset == "cresci-2017":
                bot_list = ["social_spambots_1"]
                human_list = ["genuine_accounts"]
                cresci_2017_subdatasets = os.listdir(dataset_path)
                cresci_2017_subdatasets = [subdataset for subdataset in cresci_2017_subdatasets if subdataset in bot_list or subdataset in human_list]

                output_path = os.path.join(nameclass_path, dataset+"-small.csv")

                for subdataset in cresci_2017_subdatasets:
                    subdataset_path = os.path.join(dataset_path, subdataset)
                    users_path = os.path.join(subdataset_path, "users.csv")
                    users_df = pd.read_csv(users_path)
                    screen_names = users_df["screen_name"]
                    if subdataset in human_list:
                        label = "HUMAN"
                    else:
                        label = "BOT"

                    with open(output_path, "a", newline='') as f:
                        writer = csv.writer(f)
                        for screen_name in screen_names:
                            writer.writerow([screen_name, label])
            

                    
            
            

            
