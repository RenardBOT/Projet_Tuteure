import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import preprocess


# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv
datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
mixed_dna_path = os.path.join(formatted_datasets_path, "mixed_dna")
lcs_path = os.path.join(formatted_datasets_path, "lcs")



# ARGUMENT PARSING. 
def parse_args():
    parser = argparse.ArgumentParser(description='Utilise un modèle d\'apprentissage automatique pour classifier les bots et les humains en fonction de leurs noms')
    parser.add_argument('--show', action='store_true', help='Affiche les ensembles de données disponibles.')
    parser.add_argument('--train',type=str, help='Nom de l\'ensemble de données d\'entrainement')
    parser.add_argument('--test',type=str, help='Nom de l\'ensemble de données de test')
    args = parser.parse_args()

    if args.show:
        print_datasets()
        sys.exit(0)
    else:
        if args.train is None or args.test is None:
            print("ERREUR: Les arguments --train et --test sont obligatoires si --show n'est pas spécifié.")
            print_datasets()
            sys.exit(1)

        train_path = os.path.join(lcs_path, args.train + ".csv")
        test_path = os.path.join(lcs_path, args.test + ".csv")

        

        if not os.path.exists(train_path):
            print("ERREUR: Le dataset", args.dataset, "n'existe pas dans le dossier lcs.")
            sys.exit(1)
        if not os.path.exists(test_path):
            print("ERREUR: Le dataset", args.dataset, "n'existe pas dans le dossier lcs.")
            sys.exit(1)

        if args.train not in datasets_in_common():
            print("ERREUR: Le dataset", args.train,"doit être dans les deux dossiers lcs et mixed_dna avec le même nom.")
            sys.exit(1)

        if args.test not in datasets_in_common():
            print("ERREUR: Le dataset", args.test,"doit être dans les deux dossiers lcs et mixed_dna avec le même nom.")
            sys.exit(1)

        return args
    
def datasets_in_common():
    lst_lcs = []
    for dataset in Path(lcs_path).iterdir():
        lst_lcs.append(dataset.name.split(".")[0])

    lst_dna = []
    for dataset in Path(mixed_dna_path).iterdir():
        lst_dna.append(dataset.name.split(".")[0])

    return list(set(lst_lcs) & set(lst_dna))

def print_datasets():
    for dataset in datasets_in_common():
        print("  -", dataset)

def load_dna(train_path, test_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

def longest_lcs(dna_dataframe,lcs_dataframe,type):
    longest_lcs = []
    print("Calcul de la longueur du plus long LCS pour le jeu de ",type,"...")
    for index, row in tqdm(dna_dataframe.iterrows(), total=dna_dataframe.shape[0]):
        user_dna = row["DNA"]
        for index,row in lcs_dataframe.iterrows():
            if str(row['path']) in str(user_dna):
                longest_lcs.append(row['length'])
                break
    

if __name__ == "__main__":
    args = parse_args()

    train_dna_path = os.path.join(mixed_dna_path, args.train + ".csv")
    test_dna_path = os.path.join(mixed_dna_path, args.test + ".csv")

    train_lcs_path = os.path.join(lcs_path, args.train + ".csv")
    test_lcs_path = os.path.join(lcs_path, args.test + ".csv")

    train_dna_dataframe = pd.read_csv(train_dna_path)
    test_dna_dataframe = pd.read_csv(test_dna_path)

    train_lcs_dataframe = pd.read_csv(train_lcs_path)
    test_lcs_dataframe = pd.read_csv(test_lcs_path)

    longest_lcs(train_dna_dataframe,train_lcs_dataframe,"train")
    longest_lcs(test_dna_dataframe,test_lcs_dataframe,"test")


