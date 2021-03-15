import csv
import os
from pathlib import Path
import pandas

#importing other python files:
#from folder.subfolder.file import class


class HateBaseFilter(object):
    ## Attributes
    __lexicon = []
    __all_tweets = []
    __filtered_tweets = []
    __filename = None

    ## Constructor
    def __init__(self):
        # Set path to Hatebase lexicon
        data_folder = Path(__file__).parent
        data_path = (data_folder / "..data/hatebase/hatebase_dutch_full_manual_filtering.csv").resolve()
        column_names = ["result__vocabulary_id", "result__term", "result__hateful_meaning", "result__number_of_sightings", "relevance"]
        self.__lexicon = pandas.read_csv(open(hatebase_file), names=column_names)

        # Return the lexicon as a list
        self.__lexicon.drop([0])
        self.__lexicon = self.__lexicon.result__term.to_list()

    ## Methods
    # returns a pandas dataframe containing all the tweets from a .csv file
    def get_all_tweets(self):



    # returns a .csv file of the tweets ran through the hatebase filter
    #def get_hate_tweets(self):
    #   return 

if __name__ == "__main__":
    filter = HateBaseFilter()

    while True:
        # Write what to do here
        # User input: name of the csv tweet file
        master.__filename = input("Enter filename (yyyymmdd):")

