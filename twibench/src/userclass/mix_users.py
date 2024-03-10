import pandas as pd
import os
import sys

# Récupération du fichier de configuration
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def file_dictionnary(path):
    # Créer un dictionnaire associant un numéro d'id commençant à 0 à un nom de fichier
    file_dictionnary = {}
    for i, file in enumerate(os.listdir(path)):
        file_dictionnary[i] = file
    return file_dictionnary

def name_output_file():
    print("Entrez le nom du fichier de sortie : ", end="")
    user_input = input()
    while not user_input.isalnum() and "-" not in user_input and "_" not in user_input:
        print("ERREUR: Le nom du fichier de sortie ne doit contenir que des lettres, des chiffres, des tirets ou des underscores.")
        print("Entrez le nom du fichier de sortie : ", end="")
        user_input = input()
    return user_input

def pick_remaining_dataset(dna_dict):
    print("--- DATASETS RESTANT ---")
    for i, file_name in dna_dict.items():
        print(f"[{i}] - {file_name}")
    print(" ")
    print("Entrez le numéro du fichier à ajouter au mélange : ", end="")
    user_input = int(input())
    while user_input not in dna_dict:
        print("ERREUR: Le numéro entré n'est pas valide.")
        print("Entrez le numéro du fichier à ajouter au mélange : ", end="")
        user_input = int(input())
    return user_input

def pick_sample_size(max_size):
    print(f"Entrez la taille de l'échantillon (entre 1 et {max_size}) : ", end="")
    user_input = int(input())
    while user_input < 1 or user_input > max_size:
        print(f"ERREUR: La taille de l'échantillon doit être supérieure à 0 et inférieure à {max_size}.")
        print(f"Entrez la taille de l'échantillon (entre 1 et {max_size}) : ", end="")
        user_input = int(input())
    return user_input

def exit_choice():
    print("Voulez-vous ajouter un autre fichier au mélange ? (o/n) ", end="")
    user_input = input().lower()
    while user_input not in ["o", "n"]:
        print("ERREUR: La réponse entrée n'est pas valide.")
        print("Voulez-vous ajouter un autre fichier au mélange ? (o/n) ", end="")
        user_input = input().lower()
    if user_input == "n":
        return True
    return False

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_statistics(output_dataframe):
    print("--- STATISTIQUES ---")
    print("Taille du fichier final :", len(output_dataframe), "lignes")
    nb_bots = output_dataframe[output_dataframe["label"] == "BOT"].shape[0]
    nb_humans = output_dataframe[output_dataframe["label"] == "HUMAN"].shape[0]
    print("Nombre de bots :", nb_bots)
    print("Nombre d'humains :", nb_humans)

if __name__ == '__main__':
    userclass_path = os.path.join(Config().getFormattedDatasetsPath(), "userclass")
    file_dictionnary = file_dictionnary(userclass_path)

    exit_app = False

    output_dataframe = pd.DataFrame()

    while not exit_app:
        clear_screen()
        
        user_input = pick_remaining_dataset(file_dictionnary)
        file_path = os.path.join(userclass_path, file_dictionnary[user_input])
        # Vérifier si le fichier existe
        if not os.path.exists(file_path):
            print("ERREUR: Le fichier", file_dictionnary[user_input], "n'existe pas.")
            continue

        print(f"Ajout du fichier {file_path} au mélange.")

        # Ajouter le fichier au dataframe (colonnes user_id et DNA)
        user_dataframe = pd.read_csv(file_path)

        # Taille échantillon
        sample_size = pick_sample_size(user_dataframe.shape[0])
        user_dataframe = user_dataframe.sample(sample_size)


        output_dataframe = pd.concat([output_dataframe, user_dataframe], ignore_index=True)
        
        del file_dictionnary[user_input]

        exit_app = exit_choice()
    
    # Enregistrement
    output_file = name_output_file()
    output_path = os.path.join(Config().getFormattedDatasetsPath(), "mixed_users", output_file + ".csv")
    print(f"Enregistrement du mélange d'ADN dans le fichier {output_path}")
    output_dataframe.to_csv(output_path, index=False, header=True)
    print("Enregistrement terminé.")

    # Afficher les statistiques
    clear_screen()
    print_statistics(output_dataframe)
