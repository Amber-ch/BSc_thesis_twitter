import csv
import os
from pathlib import Path
import pandas
import glob
import numpy as np
from tqdm import tqdm
import re
import random

# Selects 1000 random tweets from the data set and splits it over 2 subsets of 500 tweets
class DataSelector(object):
    ##attributes
    __all_tweets = None
    __top_tweets = None
    __bottom_tweets = None

    # open .csv file as pandas dataframe
    def __init__(self):
        data_folder = Path(__file__).parent
        tweet_path = (data_folder / "data/tweets/filtered/fulltext/filtered+combined_ALL.csv").resolve()
        column_names = ["id_str", "full_text"]
        self.__all_tweets = pandas.read_csv(open(tweet_path), header=0)

    def add_random(self):
        rows = len(self.__all_tweets)
        num_list = list(range(rows))
        random.shuffle(num_list)
        #print(num_list)
        self.__all_tweets["rand"] = num_list
        print(self.__all_tweets)

    def sort_df(self):
        self.__all_tweets.sort_values(by="rand")

    def select_tweets(self):
        self.add_random()
        self.sort_df()
        self.__all_tweets = self.__all_tweets.head(n=1000)
        # create two separate datasets
        self.__top_tweets = self.__all_tweets.head(n=500)
        self.__top_tweets = self.__top_tweets.drop(columns="rand")
        self.__bottom_tweets = self.__all_tweets.tail(n=500)
        self.__bottom_tweets = self.__bottom_tweets.drop(columns="rand")
        # save to .csv
        top_filename = "validation_set_0.csv"
        bottom_filename = "validation_set_1.csv"
        base_path = Path(__file__).parent
        top_path = (base_path / "data/tweets/filtered/validation" / top_filename).resolve()
        bottom_path = (base_path / "data/tweets/filtered/validation" / bottom_filename).resolve()
        self.__top_tweets.to_csv(path_or_buf=top_path, index=False)
        self.__bottom_tweets.to_csv(path_or_buf=bottom_path, index=False)


if __name__ == "__main__":
    select = DataSelector()
    select.select_tweets()