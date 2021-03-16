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
    __all_tweets = None
    __filtered_tweets = None
    __filename = None
    __tweet_columns = []
    __tweet_id = []
    __full_text = []
    __filter_tweet_id = []
    __filter_full_text = []
    __month = None

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
        self.__tweet_columns = ["id_str", "full_text"]

    ## Methods

    def set_filename(self, value):
        # Add ".csv" if not added yet, otherwise it will open the file with IDs only
        extension = ".csv"
        if extension not in value:
            value = value + extension
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
        else:
            print("File found!")
            all_tweets_list = list(csv.reader(open(result[0])))

            # loop over all tweets from the dataset to extract IDs and full texts
            for tweet in all_tweets_list:
                tweet_elements = str(tweet)
                tweet_elements_list = []
                current_tweet__id = []
                current_tweet__full_text = []

                tweet_elements_list = tweet_elements.split(', ')
                for element in tweet_elements_list:
                    if "'id_str:" in element:
                        current_tweet__id.append(element)
                    elif 'full_text' in element:
                        current_tweet__full_text.append(element)

                # only take the first tweet from this list, as this is the id linked to the text
                tweet_id = str(current_tweet__id[0]).replace('id_str:"', '')

                # remove unwanted substrings
                tweet_id = tweet_id.replace('"', '')
                tweet_id = tweet_id.replace("'", '')

                tweet_text = str(current_tweet__full_text[0]).split('"')
                tweet_text = str(tweet_text[1]).replace("'", '')

                self.__tweet_id.append(tweet_id)
                self.__full_text.append(tweet_text)
            
            if(len(self.__tweet_id) != len(self.__full_text)):
                print("Warning: number of tweet IDs do not correspond with number of texts")
                
            # turn lists of IDs and texts into pandas dataframe
            self.__all_tweets = pandas.DataFrame(list(zip(self.__tweet_id, self.__full_text)), columns=['id_str', 'full_text'])
            #print(self.__all_tweets)



    # TODO return a .csv file of the tweets ran through the hatebase filter (ID + full text)
    def get_hate_tweets(self):
        index = 0
        for tweet in self.__full_text:
            for keyword in self.__lexicon:
                if keyword in tweet:
                    self.__filter_full_text.append(tweet)
                    self.__filter_tweet_id.append(self.__tweet_id[index])

            index = index + 1
        self.__filtered_tweets = pandas.DataFrame(list(zip(self.__filter_tweet_id, self.__filter_full_text)), columns=['id_str', 'full_text'])
        
        # give it a new name to distinguish from the original file
        new_filename = self.__filename.replace('.csv', '')
        new_filename = new_filename + '_filtered.csv'
        self.set_filename(new_filename)

        base_path = Path(__file__).parent
        path = (base_path / "data/tweets/filtered" / new_filename).resolve()

        self.__filtered_tweets.to_csv(path_or_buf=path, index=False)
        print(self.__filtered_tweets)
        

if __name__ == "__main__":
    filter = HateBaseFilter()

    while True:
        find_filename = input("Enter filename (yyyymmdd):\n")
        
        filter.set_filename(find_filename)
        filter.get_all_tweets()
        filter.get_hate_tweets()

