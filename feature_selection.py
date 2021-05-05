import csv
import os
from pathlib import Path
import pandas
import glob
import numpy as np
from tqdm import tqdm
import re
import nltk
from nltk.corpus import stopwords
from nltk.util import ngrams
import spacy
import re
from collections import Counter
import string
import random
import heapq
from itertools import islice
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


#git add . && git commit -am "message"

# DONE validation sets majority vote, remove those without a majority
# DONE remove tweets with label 3 (it means duplicate)
# DONE remove tweets that are duplicate due to validation
# DONE add tweets back in cuz im a dumbass
# DONE data cleaning/preprocessing steps (ngram identification & lemmatization)
# TODO lookup scores with EmoLex
# TODO lookup anger intensity score
# TODO positive/negative sentiment score

class FeatureSelection(object):
    base_folder = None
    data_folder = None
    __test_set = None
    __validation_set = None
    __validation_set_1 = None
    __merge_validation = None
    __all_data = None
    __current_document = None
    __stop_words = []
    __bigrams = []
    __trigrams = []
    __preprocessed_docs = []
    wordfreq = {}
    freq_list = []
    most_freq_list = []
    freq_dict = {}
    most_freq_dict = {}
    most_relevant_list = []
    most_relevant_dict = {}
    __corpus = None
    __idf_values = {}
    __tf_values = {}
    __tf_idf_values = None

    def __init__(self):
        self.base_folder = Path(__file__).parent
        self.data_folder = (self.base_folder / "data/tweets/filtered").resolve()
        file_path = (self.base_folder / "data/tweets/filtered/labelled_test_set.csv").resolve()
        validation_path = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels.csv").resolve()
        validation_path_1 = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels_1.csv").resolve()
        stop_words_1_path = (self.base_folder / "data/stop-words_dutch_1_nl.txt")
        stop_words_2_path = (self.base_folder / "data/stop-words_dutch_2_nl.txt")


        nltk_stopwords = stopwords.words("dutch")
        additional_stopwords = pandas.read_csv(stop_words_1_path, sep=" ", names=["stopword"])
        additional_stopwords = additional_stopwords['stopword'].tolist()
        additional_stopwords_2 = pandas.read_csv(stop_words_2_path, sep=" ", names=["stopword"])
        additional_stopwords_2 = additional_stopwords_2['stopword'].tolist()
        #print("additional", additional_stopwords['stopword'].tolist())
        all_stopwords = list(set(nltk_stopwords + additional_stopwords + additional_stopwords_2))
        self.__stop_words = all_stopwords
        print(len(all_stopwords), all_stopwords)
        #print(len(nltk_stopwords), nltk_stopwords)
        #print(len(additional_stopwords), additional_stopwords)
        #print(self.__stop_words)

        self.__test_set = pandas.read_csv(file_path)
        self.__test_set.drop(columns=['path', '.', '_id', 'brush', 'annotation.0'], inplace=True)
    
        self.__validation_set = pandas.read_csv(validation_path)
        self.__validation_set_1 = pandas.read_csv(validation_path_1)
        #print(self.__validation_set_1)
        

    # Returns the majority vote for the validation sets
    def majority_vote(self, df):
        validation_df = df
        validation_df['custom_id'] = pandas.Series(validation_df['custom_id'], dtype="string")
        validation_df['document'] = pandas.Series(validation_df['document'], dtype="string")
        #print(type(validation_df['custom_id']))
        mode_df = validation_df.mode(axis=1, numeric_only=True)
        mode_df.rename(columns={mode_df.columns[0]: 'first'}, inplace=True)
        mode_df.rename(columns={mode_df.columns[1]: 'second'}, inplace=True)
        mode_df.rename(columns={mode_df.columns[2]: 'third'}, inplace=True)
        #print("mode",mode_df)
        mode_df = mode_df.join(validation_df['document'])
        mode_df = mode_df.join(validation_df['custom_id'])
        mode_df = mode_df[mode_df['second'].isnull()]
        mode_df = mode_df[['first', 'custom_id']]
        #print(mode_df)
        validation_df['majority'] = validation_df.mode(axis=1, numeric_only=True)[0]
        #print(validation_df)

        majority_df = mode_df.merge(validation_df, on='custom_id')
        majority_df.drop(columns='first', inplace=True)
        #print("majority",majority_df)
        return majority_df

    def get_merged_validation(self):
        majority_df = self.majority_vote(self.__validation_set)
        majority_df_1 = self.majority_vote(self.__validation_set_1)
        self.__merge_validation = pandas.concat([majority_df, majority_df_1])
        self.__merge_validation = self.__merge_validation[['custom_id', 'majority', 'document']]
        #print(self.__merge_validation)

    def merge_all(self):
        self.get_merged_validation()
        self.__merge_validation.rename(columns={"majority":"annotation"}, inplace=True)
        self.__test_set = self.__test_set[['custom_id', 'annotation', 'document']]
        self.__all_data = pandas.concat([self.__merge_validation, self.__test_set])
        #print(self.__all_data)
        
    # Removes rows with value NaN & 3 from the original set
    # Test set: 25 tweets after this function
    # First 3 test tweets contain duplicates with the validation set
    def remove_duplicate(self):
        self.__all_data.dropna(inplace=True)
        self.__all_data = self.__all_data[self.__all_data['annotation'] != 3.0]
        self.__all_data['duplicate'] = self.__all_data.duplicated(subset='custom_id')
        self.__all_data.drop_duplicates(subset='custom_id', inplace=True)
        self.__all_data.drop(columns='duplicate', inplace=True)
        # Toggle for final product to save data set
        #write_to = (self.data_folder/"remove_duplicate_dataset.csv").resolve()
        #self.__all_data.to_csv(path_or_buf=write_to, index=False)
        #print(self.__all_data)

    def remove_links(self):
        removed_link = re.sub(r'http\S+', '', self.__current_document)
        return removed_link

    def remove_mention(self):
        removed_mention = re.sub(r'@[A-Za-z0-9]+','', self.__current_document)
        return removed_mention

    def remove_numerals(self):
        removed_numerals = re.sub(r'[0-9]+', '', self.__current_document)
        return removed_numerals

    def remove_interpunction(self):
        remove_interpunction = re.sub(r'[^\w\s]', '', self.__current_document)
        # For some reason _ is not included in the regex
        remove_interpunction = remove_interpunction.replace('_', '')
        return remove_interpunction

    def remove_emoji(self):
        regex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
        regex_pattern = regex_pattern.sub(r'', self.__current_document)
        return regex_pattern

    def remove_double_space(self):
        remove_spaces = re.sub(r'/\s\s+/g', ' ', self.__current_document)
        # Did not remove all double spaces
        remove_spaces = re.sub(r' +', ' ', self.__current_document)
        return remove_spaces
        
    def remove_stopwords(self):
        tokenized = nltk.word_tokenize(self.__current_document)
        remove_stopwords = [token for token in tokenized if token not in self.__stop_words]
        separator = ' '
        remove_stopwords = separator.join(remove_stopwords)
        return remove_stopwords

    def define_bigrams(self):
        tokenized = nltk.word_tokenize(self.__current_document)
        print(list(ngrams(tokenized, 2)))
        self.__bigrams.append(ngrams(tokenized, 2))
        #print(list(self.__bigrams)[:10])

    def count_ngrams(self, ngrams):
        ngram_frequency = collections.Counter(ngrams)
        ngram_frequency.most_common(10)

    # Convert document column to list, preprocess, convert back to dataframe and join full dataframe
    def preprocess(self):
        documents = self.__all_data['document'].to_list()
        #print("docs", documents)
        for tweet in documents:
            #print(tweet)
            self.__current_document = tweet
            self.__current_document = self.remove_links()
            self.__current_document = self.__current_document.lower()
            self.__current_document = self.remove_mention()
            self.__current_document = self.remove_numerals()
            self.__current_document = self.remove_interpunction()
            self.__current_document = self.remove_emoji()
            self.__current_document = self.remove_double_space()
            self.__current_document = self.remove_stopwords()
            #print(self.__current_document, "\n")
            self.__preprocessed_docs.append(self.__current_document)

        # Merge processed tweets with rest of the dataframe
        self.__preprocessed_docs = pandas.DataFrame(self.__preprocessed_docs, columns=['processed'])
        self.__tweet_ids = self.__all_data[['custom_id']]
        self.__preprocessed_docs = self.__preprocessed_docs.join(self.__tweet_ids)
        #print(self.__preprocessed_docs)
        #print(self.__all_data)
        self.__all_data.rename(columns={'document':'unprocessed'}, inplace=True)
        #print(self.__all_data)
        self.__all_data = self.__all_data.merge(self.__preprocessed_docs, on='custom_id')
        print(self.__all_data)

    # Returns a list of N most frequent terms with their counts
    def identify_frequent_words(self):
        self.__corpus = self.__all_data['processed']
        #print(corpus)
        for sentence in self.__corpus:
            #print(sentence)
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                #print(word)
                if token not in self.wordfreq.keys():
                    self.wordfreq[token] = 1
                else:
                    self.wordfreq[token] += 1
     
        # Returns the N most common words with their frequencies as a dict
        self.freq_dict = dict(sorted(self.wordfreq.items(), key=lambda item: item[1], reverse=True))
        self.most_freq_dict = dict(islice(self.freq_dict.items(), 0,19))
        print(self.most_freq_dict)
        #print(self.most_freq_dict)

        # Returns just the N most common words as a list
        self.freq_list = sorted(self.wordfreq, key=self.wordfreq.get, reverse=True)
        #print(self.freq_list)
        self.most_freq_list = heapq.nlargest(20, self.wordfreq, key=self.wordfreq.get)
        #print(self.most_freq_list)

    # Returns a list of the N most relevant terms with their TF-IDF scores
    def identify_relevant_words(self):
        docs = self.__all_data['processed']
        docs = [' '.join(list(docs))]
        #print(' '.join(list(docs)))
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True)
        tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
        df = pandas.DataFrame(tfidf_vectorizer_vectors[0].T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])
        df.sort_values(by=["tfidf"], ascending=False, inplace=True)
        print(df.head(30))



if __name__ == "__main__":
    featselect = FeatureSelection()
    featselect.merge_all()
    featselect.remove_duplicate()
    featselect.preprocess()
    featselect.identify_frequent_words()
    featselect.identify_relevant_words()