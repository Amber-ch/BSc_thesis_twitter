from nltk.metrics.agreement import AnnotationTask
import simpledorff
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
# Toggle for different sets!
#set1_annotator0_path = (validation_path / "validation_set_1_0.udt.csv").resolve()
#set1_annotator1_path = (validation_path / "validation_set_1_1.udt.csv").resolve()

# Create the dataframe with validation labels and tweet IDs
udt_labelled_set = pandas.read_csv(udt_labelled_path, header=0)
udt_labelled_set = udt_labelled_set.iloc[1:]
udt_labelled_set = udt_labelled_set.drop(columns=['path', '.', '_id', 'brush', 'annotation.0', 'document'])
udt_labelled_set['custom_id'] = udt_labelled_set['custom_id'].astype(int)
udt_labelled_set = udt_labelled_set.dropna()
udt_labelled_set['annotation'] = udt_labelled_set['annotation'].astype(int)

sheet_labelled_set = pandas.read_csv(sheet_labelled_path, header=0, names=['custom_id', 'document', 'annotation'])
sheet_labelled_set = sheet_labelled_set.drop(columns='document')
sheet_labelled_set['custom_id'] = sheet_labelled_set['custom_id'].astype(int)

frames = [sheet_labelled_set, udt_labelled_set]
original_set = pandas.concat(frames)
original_set = original_set.sort_values(by='custom_id')
print("original", original_set)

set0_annotator0 = pandas.read_csv(set0_annotator0_path, header=0)
set0_annotator1 = pandas.read_csv(set0_annotator1_path, header=0)

set0_tweet_id = set0_annotator0['custom_id'].dropna()
id_list = list(set0_tweet_id)
int_id_list = []
for id in id_list:
    int_id_list.append(int(id))
int_id_list.sort()


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
duplicated_column = duplicated_column.sort_values(['custom_id', 'annotation'])
original_annotations = duplicated_column[['custom_id','annotation']].dropna()


print(type(original_annotations), original_annotations)
duplicated_column.drop(columns='annotation', inplace=True)
duplicated_column.dropna(inplace=True)
all_labels_df = duplicated_column.merge(original_annotations)
all_labels_df.rename(columns={"annotation": "label_2"}, inplace=True)
labels_only = all_labels_df.drop(columns='custom_id')
print(all_labels_df)

label_0_df = all_labels_df[['label_0', 'custom_id']]
label_0_df.rename(columns={"label_0" : "label"}, inplace=True)
#label_0_df = label_0_df.insert(0, 'annotator', '0')
label_0_df['annotator'] = '0'
label_1_df = all_labels_df[['label_1', 'custom_id']]
label_1_df.rename(columns={"label_1" : "label"}, inplace=True)
#label_1_df = label_1_df.insert(0, 'annotator', '1')
label_1_df['annotator'] = '1'
label_2_df = all_labels_df[['label_2', 'custom_id']]
label_2_df.rename(columns={"label_2" : "label"}, inplace=True)
#label_2_df = label_2_df.insert(0, 'annotator', '2')
label_2_df['annotator'] = '2'

#print(label_0_df,label_1_df,label_2_df)

krippendorff_df = pandas.concat([label_0_df, label_1_df, label_2_df])
#krippendorff_df = krippendorff_df.concat(label_2_df)
print(krippendorff_df)
# Each row corresponds to an annotator, each column corresponds to a tweet
labels_transposed = labels_only.transpose()

def df_to_experiment_annotator_table(df, experiment_col, annotator_col, class_col):
    return df.pivot_table(index=annotator_col, columns=experiment_col, values=class_col, aggfunc="first")

#all_labels_df.pivot_table(index='custom_id', columns='')

krippendorff_pivot = df_to_experiment_annotator_table(krippendorff_df, 'custom_id', 'annotator', 'label')
print(krippendorff_pivot)

#print(labels_transposed_pivot)