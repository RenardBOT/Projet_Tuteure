import pandas as pd
import numpy as np
import os
import sys

# Retrieving config file
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

#np show all columns
pd.set_option('display.max_columns', None)

def feature_entropy(string):
    if len(string) == 0:
        return 0
    entropy = 0
    for char in set(string):
        p_char = string.count(char) / len(string)
        entropy += - p_char * np.log2(p_char)
    return entropy

def feature_engineering(input_dataframe):
    
    output_dataframe = pd.DataFrame()

    output_dataframe['user_id'] = input_dataframe['user_id']
    output_dataframe['label'] = input_dataframe['label']
    output_dataframe['screen_name_length'] = input_dataframe['screen_name'].apply(lambda x: len(x))
    output_dataframe['default_profile_image'] = input_dataframe['default_profile_image'].apply(lambda x: False if pd.isna(x) else True)
    output_dataframe['screen_name_entropy'] = input_dataframe['screen_name'].apply(lambda x: feature_entropy(x))
    output_dataframe['has_location'] = input_dataframe['location'].apply(lambda x: False if pd.isna(x) else True)
    output_dataframe['total_tweets'] = input_dataframe['statuses_count']
    output_dataframe['followers_count'] = input_dataframe['followers_count']
    output_dataframe['follows_count'] = input_dataframe['friends_count']
    output_dataframe['favourites_count'] = input_dataframe['favourites_count']
    output_dataframe['last_status_retweet'] = input_dataframe['retweeted_status_id'].apply(lambda x: False if x == 0 else True)
    output_dataframe['num_hashtags'] = input_dataframe['num_hashtags']
    output_dataframe['num_mentions'] = input_dataframe['num_mentions']
    output_dataframe['age'] = (pd.to_datetime('today') - pd.to_datetime(input_dataframe['created'])).dt.days
    output_dataframe['avg_tweets_per_day'] = output_dataframe['total_tweets'] / output_dataframe['age']

    return output_dataframe





if __name__ == '__main__':
    datasets_path, datasets_list, formatted_datasets_path = Config().getDatasetsConfig()

    userclassraw_path = os.path.join(formatted_datasets_path, "userclass_raw")
    userclass_path = os.path.join(formatted_datasets_path, "userclass")

    if not os.path.exists(userclassraw_path):
        os.makedirs(userclass_path)
    
    rawlist = os.listdir(userclassraw_path)

    for raw in rawlist:
        raw_path = os.path.join(userclassraw_path, raw)
        userclassraw = pd.read_csv(raw_path)
        output_dataframe = feature_engineering(userclassraw)
        output_path = os.path.join(userclass_path, raw)
        output_dataframe.to_csv(output_path, index=False, header=True)

