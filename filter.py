import csv
import os
from pathlib import Path
import pandas
import glob
import numpy as np
from tqdm import tqdm
import re

class HateBaseFilter(object):
    ## Attributes
    __hatebase_lexicon = []
    __davidson_lexicon = []
    __full_lexicon  = []
    __all_tweets = None
    __filtered_tweets = None
    __filename = None
    __tweet_columns = []
    __tweet_id = []
    __full_text = []
    __text_translation = []
    __filter_tweet_id = []
    __filter_full_text = []
    __filter_text_translation = []
    __filter_keyword_translation = []
    __contains_keyword = []
    __translation_contains_keyword = []
    __translator = google_translator()
    __filter_keywords = []
    __filter_text_and_id = {}

    ## Constructor
    def __init__(self):
        # Set path to Dutch Hatebase lexicon
        data_folder = Path(__file__).parent
        hatebase_path = (data_folder / "data/hatebase/hatebase_dutch_full_manual_filtering.csv").resolve()
        hatebase_column_names = ["result__vocabulary_id", "result__term", "result__hateful_meaning", "result__number_of_sightings", "relevance"]
        self.__hatebase_lexicon = pandas.read_csv(open(hatebase_path, encoding="utf-8"), names=hatebase_column_names)

        # Set path to Davidson et al. (2017) refined Hatebase lexicon
        davidson_path = (data_folder / "davidson_github/lexicons/refined_ngram_dict_trans.csv").resolve()
        davidson_column_names = ["keyword"]
        self.__davidson_lexicon = pandas.read_csv(open(davidson_path, encoding="utf-8"), names=davidson_column_names)

        # Return the lexicons as lists
        self.__hatebase_lexicon.drop([0])
        self.__hatebase_lexicon = self.__hatebase_lexicon.result__term.to_list()
        # Removes the first element (csv header)
        del self.__hatebase_lexicon[0]
        self.__davidson_lexicon.drop([0])
        self.__davidson_lexicon = self.__davidson_lexicon.keyword.to_list()
        del self.__davidson_lexicon[0]
        self.__full_lexicon = list(set(self.__hatebase_lexicon + self.__davidson_lexicon))

        # Set column names for the tweet files
        self.__tweet_columns = ["id_str", "full_text", "text_translation"]

    ## Methods
    def reset(self):
        self.__all_tweets = None
        self.__filtered_tweets = None
        self.__filename = None
        self.__tweet_id = []
        self.__full_text = []
        self.__filter_tweet_id = []
        self.__filter_full_text = []
        self.__contains_keyword = []
        self.__filter_keywords = []

    def set_csv_filename(self, value):
        # Add ".csv" if not added yet, otherwise it will open the file with IDs only
        extension = ".csv"
        if extension not in value:
            value = value + extension
        self.__filename = value

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
        elif int(list(self.__filename)[5])<0:
            print("File found!")
            all_tweets_list = list(csv.reader(open(result[0])))

            # loop over all tweets from the dataset to extract IDs and full texts
            for tweet in tqdm(all_tweets_list):
                tweet_elements = str(tweet)
                tweet_elements_list = []
                current_tweet__id = []
                current_tweet__full_text = []
                current_tweet__translated = []

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
            self.__all_tweets = pandas.DataFrame(np.column_stack([self.__tweet_id, self.__full_text]), columns=['id_str', 'full_text'])

        # tweet files may-december
        else:
            all_tweets_df = pandas.read_csv(filepath_or_buffer=result[0], header=0, quotechar='"', delimiter=',', 
                        quoting=csv.QUOTE_ALL, skipinitialspace=True, error_bad_lines=False, engine='python')
            subset_df = all_tweets_df[['id', 'text']]
            subset_df = subset_df.dropna(axis=0)
            #for index in range(subset_df.shape[1]):
            #    subset_df.at[index, 'text'].replace('"', '')
            self.__all_tweets = subset_df.rename(columns={'id':'id_str', 'text':'full_text'})
            self.__tweet_id = self.__all_tweets['id_str'].tolist()
            self.__full_text = self.__all_tweets['full_text'].tolist()
            clean_tweets = []
            for tweet in self.__full_text:
                if "," in tweet:
                    tweet = str(tweet).replace(",", '')
                if "\n" in tweet:
                    tweet = str(tweet).replace("\n", '')
                if "'" in tweet:
                    tweet = str(tweet).replace("'", '')
                if '"' in tweet:
                    tweet = str(tweet).replace('"', '')
                clean_tweets.append(tweet)
            self.__full_text = []
            self.__full_text = clean_tweets

        if(len(self.__tweet_id) != len(self.__full_text)):
            print("Warning: number of tweet IDs do not correspond with number of texts")
                

    # reads a .csv file with tweet IDs, full text, and translation. Returns a .csv file with the tweets that contain keywords from the lexicon
    def get_hate_tweets(self):
        index = 0
        for tweet in self.__full_text:
            for keyword in self.__full_lexicon:
                if ' '+keyword+' ' in tweet:
                    text_plus_id = []
                    self.__filter_full_text.append(tweet)
                    self.__filter_tweet_id.append(self.__tweet_id[index])
                    self.__filter_keywords.append(keyword)
            index = index + 1

        #print(self.__filter_full_text)

        self.__filtered_tweets = pandas.DataFrame(list(zip(self.__filter_tweet_id, self.__filter_full_text)), columns=['id_str', 'full_text'])
        self.__filtered_tweets.drop_duplicates(inplace=True)

        self.__filter_keywords = pandas.DataFrame(list(zip(self.__filter_keywords)), columns=['keywords'])

        base_path = Path(__file__).parent
        keyword_path = (base_path / "data/tweets/filtered/keywords" / filename).resolve()
        self.__filter_keywords.to_csv(path_or_buf=keyword_path, index=False)

        # give it a new name to distinguish from the original file
        new_filename = self.__filename.replace('.csv', '')
        new_filename = new_filename + '_filtered.csv'
        split_filename = list(new_filename)
        self.set_filename(new_filename)

        if split_filename[4] == '0' and split_filename[5] == '2':
            path = (base_path / "data/tweets/filtered/february" / new_filename).resolve()
        elif split_filename[5] == '3':
            path = (base_path / "data/tweets/filtered/march" / new_filename).resolve()
        elif split_filename[5] == '4':
            path = (base_path / "data/tweets/filtered/april" / new_filename).resolve()
        elif split_filename[5] == '5':
            path = (base_path / "data/tweets/filtered/may" / new_filename).resolve()
        elif split_filename[5] == '6':
            path = (base_path / "data/tweets/filtered/june" / new_filename).resolve()
        elif split_filename[5] == '7':
            path = (base_path / "data/tweets/filtered/july" / new_filename).resolve()
        elif split_filename[5] == '8':
            path = (base_path / "data/tweets/filtered/august" / new_filename).resolve()
        elif split_filename[5] == '9':
            path = (base_path / "data/tweets/filtered/september" / new_filename).resolve()
        elif split_filename[4] == '1' and split_filename[5] == '0':
            path = (base_path / "data/tweets/filtered/october" / new_filename).resolve()
        elif split_filename[4] == '1' and split_filename[5] == '1':
            path = (base_path / "data/tweets/filtered/november" / new_filename).resolve()
        elif split_filename[4] == '1' and split_filename[5] == '2':
            path = (base_path / "data/tweets/filtered/december" / new_filename).resolve()
        
        #self.__filtered_tweets.to_csv(path_or_buf=path, index=False, quoting=csv.QUOTE_NONE, quotechar='', escapechar=' ')
        self.__filtered_tweets.to_csv(path_or_buf=path, index=False)
        print("Filtered:\n", self.__filtered_tweets)

    # merges all files from a folder into a combined csv file
    def merge_files(self, value):
        data_folder = Path(__file__).parent
        if value == '2':
            file_path = (data_folder / "data/tweets/filtered/february/")
            filename = "filtered+combined_february.csv"
        elif value == '3':
            file_path = (data_folder / "data/tweets/filtered/march/").resolve()
            filename = "filtered+combined_march.csv"
        elif value == '4':        
            file_path = (data_folder / "data/tweets/filtered/april/").resolve()
            filename = "filtered+combined_april.csv"
        elif value == '5':        
            file_path = (data_folder / "data/tweets/filtered/may/").resolve()
            filename = "filtered+combined_may.csv"
        elif value == '6':        
            file_path = (data_folder / "data/tweets/filtered/june/").resolve()
            filename = "filtered+combined_june.csv"
        elif value == '7':        
            file_path = (data_folder / "data/tweets/filtered/july/").resolve()
            filename = "filtered+combined_july.csv"
        elif value == '8':        
            file_path = (data_folder / "data/tweets/filtered/august/").resolve()
            filename = "filtered+combined_august.csv"
        elif value == '9':        
            file_path = (data_folder / "data/tweets/filtered/september/").resolve()
            filename = "filtered+combined_september.csv"
        elif value == '10':        
            file_path = (data_folder / "data/tweets/filtered/october/").resolve()
            filename = "filtered+combined_october.csv"
        elif value == '11':        
            file_path = (data_folder / "data/tweets/filtered/november/").resolve()
            filename = "filtered+combined_november.csv"
        elif value == '12':        
            file_path = (data_folder / "data/tweets/filtered/december/").resolve()
            filename = "filtered+combined_december.csv"
        elif value == 'k':
            # TODO merge keyword files
            file_path = (data_folder / "data/tweets/filtered/keywords").resolve()
            filename = "combined_keywords.csv"
        elif value == 't':
            # merge all texts
            file_path = (data_folder / "data/tweets/filtered/fulltext").resolve()
            filename = "filtered+combined_ALL.csv"
        
        os.chdir(file_path)
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*')]
        #combine all files in the list

        combined_csv = pandas.concat([pandas.read_csv(f) for f in all_filenames ])
        print(combined_csv)
        combined_csv.to_csv(filename, index=False, encoding='utf-8-sig')
        

    def print_lexicon(self):
        print(self.__full_lexicon)

    def keyword_analysis(self):
        data_folder = Path(__file__).parent
        file_name = "combined_keywords.csv"
        new_name = "analyzed_keywords.csv"
        file_path = (data_folder / "data/tweets/filtered/keywords" / file_name).resolve()
        new_path = (data_folder / "data/tweets/filtered/keywords" / new_name).resolve()

        keywords_df = pandas.read_csv(filepath_or_buffer=file_path, header=0)
        keywords_df = pandas.pivot_table(keywords_df, index='keywords', aggfunc='size')
        print(keywords_df.head())
        keywords_df.to_csv(path_or_buf=new_path)
        #print(keywords_df)

        

if __name__ == "__main__":
    filter = HateBaseFilter()

    while True:
        #filter.print_lexicon()
        command = input("Enter command (m=merge files, f=filter):\n")
        if(command == 'm'):
            month = input("Which month? (2=feb, 3=march, etc., k=keywords)")
            filter.merge_files(month)
        # run keyword analysis
        elif(command == 'a'):
            filter.keyword_analysis()
        elif(command == 'f'):
            while True:
            # create csv file for specific date
                filename = input("Enter filename (yyyymmdd):")
                filter.set_filename(filename)
                filter.get_all_tweets()
                filter.get_hate_tweets()
                filter.reset()

