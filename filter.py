import csv
import os
from pathlib import Path
import pandas
import glob

#importing other python files:
#from folder.subfolder.file import class


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
    __filter_tweet_id = []
    __filter_full_text = []
    __contains_keyword = []

    ## Constructor
    def __init__(self):
        # Set path to Hatebase lexicon
        data_folder = Path(__file__).parent
        hatebase_path = (data_folder / "data/hatebase/hatebase_dutch_full_manual_filtering.csv").resolve()
        hatebase_column_names = ["result__vocabulary_id", "result__term", "result__hateful_meaning", "result__number_of_sightings", "relevance"]
        self.__hatebase_lexicon = pandas.read_csv(open(hatebase_path, encoding="utf-8"), names=hatebase_column_names)

        # Set path to Davidson et al. (2017) refined Hatebase lexicon
        davidson_path = (data_folder / "davidson_github/lexicons/refined_ngram_dict_translated.csv").resolve()
        davidson_column_names = ["english", "dutch"]
        self.__davidson_lexicon = pandas.read_csv(open(davidson_path, encoding="utf-8"), names=davidson_column_names)

        # Return the lexicons as lists
        self.__hatebase_lexicon.drop([0])
        self.__hatebase_lexicon = self.__hatebase_lexicon.result__term.to_list()
        self.__davidson_lexicon.drop([0])
        self.__english_davidson_lexicon = self.__davidson_lexicon.english.to_list()
        del self.__english_davidson_lexicon[0]
        self.__dutch_davidson_lexicon = self.__davidson_lexicon.dutch.to_list()
        del self.__dutch_davidson_lexicon[0]
        self.__davidson_lexicon = self.__english_davidson_lexicon + self.__dutch_davidson_lexicon
        self.__full_lexicon = self.__hatebase_lexicon + self.__davidson_lexicon

        print(self.__full_lexicon)

        # Set column names for the tweet files
        self.__tweet_columns = ["id_str", "full_text"]

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



    # return a .csv file of the tweets ran through the hatebase filter (ID + full text)
    def get_hate_tweets(self):
        index = 0
        for tweet in self.__full_text:
            for keyword in self.__full_lexicon:
                if (' '+keyword+' ') in tweet:
                    print("contains:", keyword)
                    self.__filter_full_text.append(tweet)
                    self.__filter_tweet_id.append(self.__tweet_id[index])
                    self.__contains_keyword.append(keyword)

            index = index + 1
        self.__filtered_tweets = pandas.DataFrame(list(zip(self.__filter_tweet_id, self.__filter_full_text, self.__contains_keyword)), columns=['id_str', 'full_text', 'keyword'])
        
        # give it a new name to distinguish from the original file
        new_filename = self.__filename.replace('.csv', '')
        split_filename = list(new_filename)
        print(len(split_filename))

        new_filename = new_filename + '_filtered.csv'
        self.set_filename(new_filename)

        base_path = Path(__file__).parent
        if split_filename[5] == '2':
            path = path = (base_path / "data/tweets/filtered/february" / new_filename)
        elif split_filename[5] == '3':
            path = (base_path / "data/tweets/filtered/march" / new_filename).resolve()
        elif split_filename[5] == '4':
            path = (base_path / "data/tweets/filtered/april" / new_filename).resolve()

        self.__filtered_tweets.to_csv(path_or_buf=path, index=False)
        print(self.__filtered_tweets)

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
        
        os.chdir(file_path)
        extension = 'csv'
        all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
        #combine all files in the list
        combined_csv = pandas.concat([pandas.read_csv(f) for f in all_filenames ])
        #export to csv
        combined_csv.to_csv(filename, index=False, encoding='utf-8-sig')
        

if __name__ == "__main__":
    filter = HateBaseFilter()

    while True:
        find_filename = input("Enter filename (yyyymmdd) or command (m=merge files):\n")
        
        if(find_filename == 'm'):
            month = input("Which month? (2=feb, 3=march, 4=april)")
            filter.merge_files(month)
            # TODO merge
        else:
            filter.set_filename(find_filename)
            filter.get_all_tweets()
            filter.get_hate_tweets()
            filter.reset()

