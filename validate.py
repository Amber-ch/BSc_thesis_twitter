from nltk.metrics.agreement import AnnotationTask
import csv
import os
from pathlib import Path
import pandas
import glob
import numpy as np
from tqdm import tqdm
import re

#pandas.set_option('display.float_format', lambda x: '19.f')
base_folder = Path(__file__).parent
validation_path = (base_folder / "data/tweets/filtered/validation").resolve()
udt_labelled_path = (base_folder / "data/tweets/filtered/labelled_tweets_ALL.udt.csv").resolve()
sheet_labelled_path = (base_folder / "data/tweets/filtered/labelled_tweets_sheet.csv").resolve()
set0_annotator0_path = (validation_path / "validation_set_0_0.udt.csv").resolve()
set0_annotator1_path = (validation_path / "validation_set_0_1.udt.csv").resolve()
#set1_annotator0_path = (validation_path / "validation_set_1_0.udt.csv").resolve()
#set1_annotator1_path = (validation_path / "validation_set_1_1.udt.csv").resolve()

# Create the dataframe with validation labels and tweet IDs
udt_labelled_set = pandas.read_csv(udt_labelled_path, header=0)
udt_labelled_set = udt_labelled_set.iloc[1:]
udt_labelled_set = udt_labelled_set.drop(columns=['path', '.', '_id', 'brush', 'annotation.0'])
udt_labelled_set['custom_id'] = udt_labelled_set['custom_id'].astype(int)
udt_labelled_set = udt_labelled_set.dropna()
udt_labelled_set['annotation'] = udt_labelled_set['annotation'].astype(int)
#print(udt_labelled_set)
sheet_labelled_set = pandas.read_csv(sheet_labelled_path, header=0, names=['custom_id', 'document', 'annotation'])
sheet_labelled_set['custom_id'] = sheet_labelled_set['custom_id'].astype(int)
#udt_labelled_set.head()
#print(sheet_labelled_set)
#print(type(sheet_labelled_set))
#print(type(udt_labelled_set))
frames = [sheet_labelled_set, udt_labelled_set]
original_set = pandas.concat(frames)
original_set = original_set.sort_values(by='custom_id')
print("original", original_set)

set0_annotator0 = pandas.read_csv(set0_annotator0_path, header=0)
set0_annotator1 = pandas.read_csv(set0_annotator1_path, header=0)

set0_tweet_id = set0_annotator0['custom_id'].dropna()
#set0_tweet_id = set0_tweet_id.to_frame()
#print(set0_tweet_id)
id_list = list(set0_tweet_id)
int_id_list = []
for id in id_list:
    int_id_list.append(int(id))
int_id_list.sort()
#print(int_id_list)

#set0_tweet_id = set0_tweet_id.astype({'custom_id':'int32'})
#list_id = list(set0_tweet_id)
#print(list_id)
set0_annotator0 = set0_annotator0['annotation'].to_frame()
set0_annotator0.columns = ['label_0']

set0_annotator1 = set0_annotator1['annotation'].to_frame()
set0_annotator1.columns = ['label_1']

labelled_set = pandas.concat([udt_labelled_set, sheet_labelled_set])
labelled_set.head()

merged_df = set0_annotator0.join(set0_annotator1)
merged_df = merged_df.join(set0_tweet_id)
merged_df = merged_df.dropna()
merged_df = merged_df.sort_values(by=['custom_id'])

merged_df = merged_df.astype(int)
print(merged_df)

original_and_validation_df = pandas.concat([merged_df, original_set], ignore_index=True)
original_and_validation_df = original_and_validation_df.sort_values(by='custom_id')
print(original_and_validation_df)
duplicated_column = original_and_validation_df[original_and_validation_df.duplicated(['custom_id'], keep=False)]
#print(list(original_and_validation_df['custom_id']))
#print(len(list(original_and_validation_df['custom_id'])), len(set(list(original_and_validation_df['custom_id']))))
# 11279 - 10845 = 434 duplicates
print(duplicated_column)

original_tweet_id = list(original_set['custom_id'])
#print(original_tweet_id)
#print(int_id_list)
# Fetch the corresponding labels from original set and join with validation labels
#original_label = []
#row = 0
#for id in original_tweet_id:
#    if id in int_id_list:
#        original_label.append(merged_df.iloc[row]['annotation'])
#    row = row + 1
#print(original_label)