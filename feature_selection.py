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
import spacy

# DONE validation sets majority vote, remove those without a majority
# DONE remove tweets with label 3 (it means duplicate)
# DONE remove tweets that are duplicate due to validation
# DONE add tweets back in cuz im a dumbass
# TODO data cleaning/preprocessing steps
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

    def __init__(self):
        self.base_folder = Path(__file__).parent
        self.data_folder = (self.base_folder / "data/tweets/filtered").resolve()
        file_path = (self.base_folder / "data/tweets/filtered/labelled_test_set.csv").resolve()
        validation_path = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels.csv").resolve()
        validation_path_1 = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels_1.csv").resolve()

        self.__test_set = pandas.read_csv(file_path)
        self.__test_set.drop(columns=['path', '.', '_id', 'brush', 'annotation.0'], inplace=True)
    
        self.__validation_set = pandas.read_csv(validation_path)
        self.__validation_set_1 = pandas.read_csv(validation_path_1)
        print(self.__validation_set_1)
        

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
        print(self.__merge_validation)

    def merge_all(self):
        self.get_merged_validation()
        self.__merge_validation.rename(columns={"majority":"annotation"}, inplace=True)
        self.__test_set = self.__test_set[['custom_id', 'annotation', 'document']]
        self.__all_data = pandas.concat([self.__merge_validation, self.__test_set])
        print(self.__all_data)
        
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
        print(self.__all_data)

    #def preprocess(self):
        """
        Hyperlinks
        Lowercase
        Keyword?
        Stopword
        Lemmatization
        Stop clause
        Numerals
        User mentions
        Non-alphabetic

        hyperlink removal regex: (https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)
        lowercase: string.lower()
        word_tokenize(sentence)
        stopword: nltk stopwords
        """




if __name__ == "__main__":
    featselect = FeatureSelection()
    featselect.merge_all()
    featselect.remove_duplicate()