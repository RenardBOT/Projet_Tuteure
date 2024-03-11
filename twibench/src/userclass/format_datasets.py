import pandas as pd
import os
import sys
from tqdm import tqdm

# Importation de Config depuis le chemin absolu
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def process_subdataset(dataset, subdataset, dataset_path, user_columns_to_keep, tweet_columns_to_keep, userclassraw_path):
    subdataset_path = os.path.join(dataset_path, subdataset)
    users_path = os.path.join(subdataset_path, "users.csv")
    tweets_path = os.path.join(subdataset_path, "tweets.csv")

    # Lecture des fichiers CSV
    users_dataframe = pd.read_csv(users_path, encoding="ISO-8859-1", low_memory=False)
    tweets_dataframe = pd.read_csv(tweets_path, encoding="ISO-8859-1", low_memory=False)

    # Sélection des colonnes pertinentes
    users_dataframe = users_dataframe[user_columns_to_keep]
    tweets_dataframe = tweets_dataframe[tweet_columns_to_keep]

    # Renommage de la colonne 'id' en 'user_id'
    users_dataframe.rename(columns={'id': 'user_id'}, inplace=True)

    # Conversion de la colonne 'created_at' en datetime
    tweets_dataframe['created'] = pd.to_datetime(tweets_dataframe['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')

    # Trouver l'indice du dernier tweet pour chaque utilisateur
    last_tweet_indices = tweets_dataframe.groupby('user_id')['created_at'].idxmax()

    # Sélectionner les lignes correspondantes dans le dataframe
    last_tweets_df = tweets_dataframe.loc[last_tweet_indices]
    # Supprimer la colonne 'created_at' du dataframe des tweets
    tweets_dataframe = tweets_dataframe.drop(columns=['created_at'])

    # Fusionner les deux dataframes sur la colonne 'user_id'
    merged_df = pd.merge(users_dataframe, last_tweets_df, on='user_id')

    # Ajout de la colonne 'label' en fonction du dataset
    if dataset == "cresci-2017" and subdataset == "genuine_accounts":
        merged_df['label'] = "HUMAN"
    if dataset == "cresci-2015" and subdataset in ["TFP", "E13"]:
        merged_df['label'] = "HUMAN"
    else:
        merged_df['label'] = "BOT"
    merged_df = merged_df.drop(columns=['created_at_x', 'created_at_y'])

    # Sauvegarde du dataframe fusionné
    output_file = os.path.join(userclassraw_path, f"{dataset}_{subdataset}.csv")
    merged_df.to_csv(output_file, index=False)

if __name__ == '__main__':
    # Récupération des chemins et des listes depuis Config
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    userclassraw_path = os.path.join(formatted_datasets_path, "userclass_raw")

    # Création du répertoire userclassraw s'il n'existe pas
    if not os.path.exists(userclassraw_path):
        os.makedirs(userclassraw_path)

    # Colonnes à garder pour les utilisateurs et les tweets
    user_columns_to_keep = [
        'id',
        'screen_name',
        'statuses_count',
        'followers_count',
        'friends_count',
        'favourites_count',
        'location',
        'default_profile_image',
        'created_at',
    ]

    tweet_columns_to_keep = [
        'user_id',
        'retweeted_status_id',
        'num_hashtags',
        'num_mentions',
        'created_at'
    ]

    # Boucle sur les datasets
    for dataset in tqdm(datasets_list, desc="Processing datasets"):
        dataset_path = os.path.join(datasets_path, dataset)
        print("dataset:", dataset)

        # Liste des sous-datasets selon le dataset
        subdatasets_list = []
        if dataset == "cresci-2017":
            subdatasets_list = ["social_spambots_1", "social_spambots_2", "social_spambots_3", "genuine_accounts"]
        elif dataset == "cresci-2015":
            subdatasets_list = ["TFP", "E13", "INT", "FSF", "TWT"]

        # Boucle sur les sous-datasets
        for subdataset in tqdm(subdatasets_list, desc=f"Processing {dataset}"):
            process_subdataset(dataset, subdataset, dataset_path, user_columns_to_keep, tweet_columns_to_keep, userclassraw_path)
