import pandas as pd
import os
import sys

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv


if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    dna_path = os.path.join(formatted_datasets_path, "dna")

    if not os.path.exists(dna_path):
        os.makedirs(dna_path)

    # mix1 : Mixer les adn des cresci-2017_social_spambots_1 and cresci-2017_genuine_accounts en prenant le même nombre de comptes
   
    genuine_dna_path = os.path.join(dna_path, "cresci-2017_genuine_accounts.csv")
    social_spambots1_dna_path = os.path.join(dna_path, "cresci-2017_social_spambots_1.csv")
    output_path = os.path.join(dna_path, "mixed1.csv")

    genuine_dataframe = pd.read_csv(genuine_dna_path, encoding="ISO-8859-1")
    bots_dataframe = pd.read_csv(social_spambots1_dna_path, encoding="ISO-8859-1")

    nb_accs = min(genuine_dataframe.shape[0], bots_dataframe.shape[0])

    print("nb_accs : ", nb_accs)
    print("shapes : ", genuine_dataframe.shape, bots_dataframe.shape)

    # output dataframe contains 3 columns : user_id, dna, label. prepare it.
    output_dataframe = pd.DataFrame(columns=["user_id", "DNA", "label"])


    genuine_dataframe = genuine_dataframe.sample(n=nb_accs)
    bots_dataframe = bots_dataframe.sample(n=nb_accs)

    # print shapes
    print("genuine_dataframe shape : ", genuine_dataframe.shape)
    print("bots_dataframe shape : ", bots_dataframe.shape)

    # put both dataframes in one in output_dataframe, with label HUMAN for genuine accounts and label BOT for social spambots
    genuine_dataframe["label"] = "HUMAN"
    bots_dataframe["label"] = "BOT"

    output_dataframe = pd.concat([genuine_dataframe, bots_dataframe])

    # print head of the output dataframe
    print(output_dataframe.head())

    # count each type
    print(output_dataframe["label"].value_counts())
    exit()

    # write it to csv file
    output_dataframe.to_csv(output_path, index=False)