from suffix_tree import Tree
import os
import time
import sys
import argparse
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv

sys.setrecursionlimit(1000000)


def from_df(dfdna):
    if "user_id" not in dfdna.columns or "DNA" not in dfdna.columns:
        raise Exception("Le dataframe doit contenir les colonnes user_id et DNA")
    
    dna_dic = {}
    for index, row in dfdna.iterrows():
        dna_dic[row["user_id"]] = row["DNA"]

    output_dataframe = pd.DataFrame(columns=["k", "length", "path"])

    # Computing the k-LCS using suffix trees
    start_time = time.time()
    try:
        tree = Tree(dna_dic)
        
        for k, length, path in tree.common_substrings():
            # retirer tous les espaces de la chaine de caractères
            path = str(path).replace(" ", "")
            output_dataframe = pd.concat([output_dataframe, pd.DataFrame({"k": [k], "length": [length], "path": [path]})], ignore_index=True)
    except Exception as e:
        print("Erreur pendant le calcul du suffix tree (LCS):" + str(e))
    end_time = time.time()

    return output_dataframe, end_time - start_time


# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
import csv
datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
output_lcs_path = os.path.join(formatted_datasets_path, "lcs")
formatted_datasets_path = os.path.join(formatted_datasets_path, "mixed_dna")



# ARGUMENT PARSING. 
def parse_args():
    parser = argparse.ArgumentParser(description='Calcule le tableau des k-LCS pour un ensemble de données donné, au format CSV.')
    show_or_train = parser.add_mutually_exclusive_group(required=True)
    show_or_train.add_argument('--show', action='store_true', help='Affiche les ensembles de données disponibles.')
    show_or_train.add_argument('--dataset',type=str, help='Nom de l\'ensemble de données à convertir en k-LCS sans l\'extension.')
    parser.add_argument('--test', type=int, help='Sépare les données en train et test. La valeur est le pourcentage de données à mettre dans le test, en int.')
    parser.add_argument('--output', type=str, help='Nom du fichier csv de sortie dans le dossier lcs, sans l\'extension.')
    args = parser.parse_args()

    if args.show:
        print_datasets()
        sys.exit(0)
    else:
        dataset_path = os.path.join(formatted_datasets_path, args.dataset + ".csv")
        if not os.path.exists(dataset_path):
            print("Error: The dataset", args.dataset, "does not exist.")
            print_datasets()
            sys.exit(1)
        return args

def print_datasets():
    print("Available datasets:")
    for dataset in Path(formatted_datasets_path).iterdir():
        print("  -", dataset.name.split(".")[0])

def load_dataset(dataset_path):
    csv_path = os.path.join(formatted_datasets_path, dataset_path + ".csv")
    dataframe = pd.read_csv(csv_path)
    return dataframe

def split_dataset(dataframe, test_size): 
    train, test = train_test_split(dataframe, test_size=test_size, stratify=dataframe["label"])
    return train, test

def write_split_lcs(train, test, output_file):

    # keep 10 lines for testing
    train = train.head(10)
    test = test.head(10)

    train_lcs, time_exec_train = from_df(train)
    print("--- Temps d'exécution  intermédiaire pour le jeu de train : ", time_exec_train)
    test_lcs, time_exec_test = from_df(test)
    print("--- Temps d'exécution intermédiaire pour le jeu de test : ", time_exec_test)

    print(f"Temps d'exécution total pour le jeu de données {output_file} : ", time_exec_train + time_exec_test)

    
    train_lcs.to_csv(os.path.join(output_lcs_path, "train_" + output_file), index=False)
    test_lcs.to_csv(os.path.join(output_lcs_path, "test_" + output_file), index=False)

    train.to_csv(os.path.join(formatted_datasets_path, "train_" + output_file), index=False)
    test.to_csv(os.path.join(formatted_datasets_path, "test_" + output_file), index=False)
    

def write_lcs(dataframe, output_file):

    # keep 10 lines for testing
    dataframe = dataframe.head(10)

    lcs, time_exec = from_df(dataframe)

    print(f"Temps d'exécution pour le jeu de données {output_file} : ", time_exec)

    lcs.to_csv(os.path.join(output_lcs_path, output_file), index=False)

    dataframe.to_csv(os.path.join(formatted_datasets_path, output_file), index=False)

def write_dna(dataframe, output_file):
    dataframe.to_csv(os.path.join(formatted_datasets_path, output_file), index=False)





if __name__ == "__main__":
    args = parse_args()
    dataframe = load_dataset(args.dataset)

    output_file = args.dataset + ".csv"
    if args.output:
        output_file = args.output + ".csv"

    test_size = args.test/float(100) if args.test else 0.2

    print("Début de la résolution du problème du plus long sous-chaine commune (LCS).")
    print("ATTENTION : Le calcul du LCS est très long pour les grands ensembles de données.")

    if args.test:
        train, test = split_dataset(dataframe, test_size)
        write_split_lcs(train, test, output_file)
    else:
        write_lcs(dataframe, output_file)

    




