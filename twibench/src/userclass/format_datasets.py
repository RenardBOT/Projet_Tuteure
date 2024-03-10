import pandas as pd
import os
import sys

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config



if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()
    userclassraw_path = os.path.join(formatted_datasets_path, "userclass_raw")

    if not os.path.exists(userclassraw_path):
        os.makedirs(userclassraw_path)


    for dataset in datasets_list:
        dataset_path = os.path.join(datasets_path, dataset)


        print("dataset : ", dataset)

        # -------------------------------------------------------------
        # --------------------- CRESCI-2017----------------------------
        # -------------------------------------------------------------

        if dataset == "cresci-2017":

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

            datasets_list = ["social_spambots_1", "social_spambots_2","social_spambots_3","genuine_accounts"]
            cresci_2017_subdatasets = os.listdir(dataset_path)

            cresci_2017_subdatasets = [subdataset for subdataset in cresci_2017_subdatasets if subdataset in datasets_list]
            
            for subdataset in cresci_2017_subdatasets:
                subdataset_path = os.path.join(dataset_path, subdataset)
                users_path = os.path.join(subdataset_path, "users.csv")
                tweets_path = os.path.join(subdataset_path, "tweets.csv")

                users_dataframe = pd.read_csv(users_path, encoding="ISO-8859-1",low_memory=False)
                tweets_dataframe = pd.read_csv(tweets_path, encoding="ISO-8859-1",low_memory=False)

                users_dataframe = users_dataframe[user_columns_to_keep]
                tweets_dataframe = tweets_dataframe[tweet_columns_to_keep]

                users_dataframe.rename(columns={'id': 'user_id'}, inplace=True)

                # Convertir la colonne 'created_at' en type datetime si ce n'est pas déjà fait
                tweets_dataframe['created'] = pd.to_datetime(tweets_dataframe['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')

                # Trouver l'indice du dernier tweet pour chaque utilisateur
                last_tweet_indices = tweets_dataframe.groupby('user_id')['created_at'].idxmax()

                # Sélectionner les lignes correspondantes dans le dataframe
                last_tweets_df = tweets_dataframe.loc[last_tweet_indices]
                # drop created_at column sur tweets_dataframe
                tweets_dataframe = tweets_dataframe.drop(columns=['created_at'])


                # Fusionner les deux dataframes sur la colonne 'user_id'
                merged_df = pd.merge(users_dataframe, last_tweets_df, on='user_id')

                # si dataset = genuine_accounts, ajouter une colonne 'label' avec la valeur HUMAN sinon BOT
                if subdataset == "genuine_accounts":
                    merged_df['label'] = "HUMAN"
                else:
                    merged_df['label'] = "BOT"

                # drop created_at_x et created_at_y
                merged_df = merged_df.drop(columns=['created_at_x', 'created_at_y'])


                # Sauvegarder le dataframe fusionné
                output_file = os.path.join(userclassraw_path,dataset+'_'+subdataset + ".csv")
                merged_df.to_csv(output_file, index=False)

