import csv
import os
from pathlib import Path
import pandas
import re
import math

#importing other python files:
#from folder.subfolder.file import class


class HateBaseFilter(object):
    ## Attributes
    __lexicon = []
    __all_tweets = []
    __filtered_tweets = []
    __filename = None
    __tweet_columns = []

    ## Constructor
    def __init__(self):
        # Set path to Hatebase lexicon
        data_folder = Path(__file__).parent
        data_path = (data_folder / "data/hatebase/hatebase_dutch_full_manual_filtering.csv").resolve()
        column_names = ["result__vocabulary_id", "result__term", "result__hateful_meaning", "result__number_of_sightings", "relevance"]
        self.__lexicon = pandas.read_csv(open(data_path, encoding="utf-8"), names=column_names)

        # Return the lexicon as a list
        self.__lexicon.drop([0])
        self.__lexicon = self.__lexicon.result__term.to_list()

        # Set column names for the tweet files
        self.__tweet_columns = ["created_at", "id", "id_str", "full_text"]

    ## Methods
    def set_filename(self, value):
        self.__filename = value

    # returns a pandas dataframe containing all the tweets from a .csv file
    def get_all_tweets(self):
        result = []
        base_path = Path(__file__).parent
        path = (base_path / "data/tweets").resolve()
        print(path)
        for root, dirs, files, in os.walk(path):
            if self.__filename in files:
                result.append(os.path.join(root, self.__filename))

        # invalid filename entered
        if len(result) == 0:
            print("File not found!")
        # valid filename entered
        else:
            print("File found!")
            all_tweets = pandas.read_csv(open(result[0]))
        #print(result)
        #print(self.__all_tweets)


    # returns a .csv file of the tweets ran through the hatebase filter
    #def get_hate_tweets(self):
    #   return 

if __name__ == "__main__":
    filter = HateBaseFilter()

    while True:
        # Write what to do here
        # User input: name of the csv tweet file
        find_filename = input("Enter filename (yyyymmdd):\n")

        # Add ".csv" if not added yet
        substring = ".csv"
        if substring not in find_filename:
            find_filename = find_filename + substring

        print(find_filename)
        filter.set_filename(find_filename)
        filter.get_all_tweets()

