# Calculate inter-rater agreement score for the 2 validation sets
# Uses Krippendorff's alpha from the simpledorff package

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

# Labels: 0, 1, 2 [hate speech, offensive, neither]
# krippendorff alpha set 0: 0.2919352663176674
# krippendorff alpha set 1: 0.07924638421273422

# Labels: 0 (stands for 0&1), 2 [hate speech&offensive, neither]
# krippendorff alpha set 0: 0.35487373939669065
# krippendorff alpha set 1: 0.1085997642383959


base_folder = Path(__file__).parent
validation_path = (base_folder / "data/tweets/filtered/validation").resolve()
udt_labelled_path = (base_folder / "data/tweets/filtered/labelled_tweets_ALL.udt.csv").resolve()
sheet_labelled_path = (base_folder / "data/tweets/filtered/labelled_tweets_sheet.csv").resolve()
set0_annotator0_path = (validation_path / "validation_set_0_0.udt.csv").resolve()
set0_annotator1_path = (validation_path / "validation_set_0_1.udt.csv").resolve()
# Toggle for different sets!
set1_annotator0_path = (validation_path / "validation_set_1_0.udt.csv").resolve()
set1_annotator1_path = (validation_path / "validation_set_1_1.udt.csv").resolve()

# Create the dataframe with validation labels and tweet IDs
udt_labelled_set = pandas.read_csv(udt_labelled_path, header=0)
udt_labelled_set = udt_labelled_set.iloc[1:]
udt_labelled_set = udt_labelled_set.drop(columns=['path', '.', '_id', 'brush', 'annotation.0'])
udt_labelled_set['custom_id'] = udt_labelled_set['custom_id'].astype(int)
udt_labelled_set = udt_labelled_set.dropna()
udt_labelled_set['annotation'] = udt_labelled_set['annotation'].astype(int)

sheet_labelled_set = pandas.read_csv(sheet_labelled_path, header=0, names=['custom_id', 'document', 'annotation'])
#sheet_labelled_set = sheet_labelled_set.drop(columns='document')
sheet_labelled_set['custom_id'] = sheet_labelled_set['custom_id'].astype(int)

frames = [sheet_labelled_set, udt_labelled_set]
original_set = pandas.concat(frames)
original_set = original_set.sort_values(by='custom_id')
print("original", original_set)

# Read volunteer validation sets
set0_annotator0 = pandas.read_csv(set0_annotator0_path, header=0)
set0_annotator1 = pandas.read_csv(set0_annotator1_path, header=0)
set1_annotator0 = pandas.read_csv(set1_annotator0_path, header=0)
set1_annotator1 = pandas.read_csv(set1_annotator1_path, header=0)

# Get the IDs from the validation sets
set0_tweet_id = set0_annotator0['custom_id'].dropna()
set1_tweet_id = set1_annotator0['custom_id'].dropna()

# Get the tweets from the validation sets
set0_documents = set0_annotator0['document']
set1_documents = set1_annotator0['document']

"""id_list = list(set0_tweet_id)
int_id_list = []
for id in id_list:
    int_id_list.append(int(id))
int_id_list.sort()"""


set0_annotator0 = set0_annotator0['annotation'].to_frame()
set0_annotator0.columns = ['label_0']

set0_annotator1 = set0_annotator1['annotation'].to_frame()
set0_annotator1.columns = ['label_1']

set1_annotator0 = set1_annotator0['annotation'].to_frame()
set1_annotator0.columns = ['label_0']

set1_annotator1 = set1_annotator1['annotation'].to_frame()
set1_annotator1.columns = ['label_1']

labelled_set = pandas.concat([udt_labelled_set, sheet_labelled_set])
#labelled_set.head()

# Merge labels from both volunteers for each validation set
merged_df = set0_annotator0.join(set0_annotator1)
merged_df = merged_df.join(set0_tweet_id)
merged_df = merged_df.join(set0_documents)
merged_df = merged_df.dropna()
merged_df = merged_df.sort_values(by=['custom_id'])

merged_df_1 = set1_annotator0.join(set1_annotator1)
merged_df_1 = merged_df_1.join(set1_tweet_id)
merged_df_1 = merged_df_1.join(set1_documents)
merged_df_1 = merged_df_1.dropna()
merged_df_1 = merged_df_1.sort_values(by=['custom_id'])

merged_df['custom_id'] = merged_df['custom_id'].astype(int)
merged_df_1['custom_id'] = merged_df_1['custom_id'].astype(int)
print("df",merged_df, merged_df_1)

# Merge validation sets and original set
original_and_validation_df = pandas.concat([merged_df, original_set], ignore_index=True)
original_and_validation_df = original_and_validation_df.sort_values(by='custom_id')

original_and_validation_df_1 = pandas.concat([merged_df_1, original_set], ignore_index=True)
original_and_validation_df_1 = original_and_validation_df_1.sort_values(by='custom_id')

#print(original_and_validation_df)


duplicated_column = original_and_validation_df[original_and_validation_df.duplicated(['custom_id'], keep=False)]
duplicated_column = duplicated_column.sort_values(['custom_id', 'annotation'])
#print("dup", duplicated_column)
original_annotations = duplicated_column[['custom_id','document', 'annotation']].dropna()
#print("here", original_annotations)
#print(duplicated_column)

duplicated_column_1 = original_and_validation_df_1[original_and_validation_df_1.duplicated(['custom_id'], keep=False)]
duplicated_column_1 = duplicated_column_1.sort_values(['custom_id', 'annotation'])
original_annotations_1 = duplicated_column_1[['custom_id','document', 'annotation']].dropna()

#print(type(original_annotations), original_annotations)
duplicated_column = duplicated_column.drop(columns='annotation')
#print(duplicated_column)
duplicated_column.dropna(inplace=True)
#print("clean", duplicated_column)
duplicated_column_1 = duplicated_column_1.drop(columns='annotation')
duplicated_column_1.dropna(inplace=True)
print(duplicated_column_1)
#print("duplicate clean", duplicate_clean)

all_labels_df = duplicated_column.merge(original_annotations)
all_labels_df.rename(columns={"annotation": "label_2"}, inplace=True)
labels_only = all_labels_df.drop(columns='custom_id')
all_labels_df_1 = duplicated_column_1.merge(original_annotations_1)
all_labels_df_1.rename(columns={"annotation": "label_2"}, inplace=True)
labels_only_1 = all_labels_df_1.drop(columns='custom_id')
#print("len", len(all_labels_df))
print("all_labels", all_labels_df)

# Exclude duplicate tweets (marked with label 3)
all_labels_df = all_labels_df[(all_labels_df.label_0 != 3) & (all_labels_df.label_1 != 3) & (all_labels_df.label_2 != 3)]
all_labels_df_1 = all_labels_df_1[(all_labels_df_1.label_0 != 3) & (all_labels_df_1.label_1 != 3) & (all_labels_df_1.label_2 != 3)]
print(all_labels_df, all_labels_df_1)

merge_labels_df = all_labels_df.replace(to_replace=1.0, value=0.0)
merge_labels_df_1 = all_labels_df_1.replace(to_replace=1.0, value=0.0)
#print(merge_labels_df, merge_labels_df_1)

# Gather all labels in one column
label_0_df = all_labels_df[['label_0', 'custom_id']]
label_0_df.rename(columns={"label_0" : "label"}, inplace=True)
label_0_df['annotator'] = '0'
label_1_df = all_labels_df[['label_1', 'custom_id']]
label_1_df.rename(columns={"label_1" : "label"}, inplace=True)
label_1_df['annotator'] = '1'
label_2_df = all_labels_df[['label_2', 'custom_id']]
label_2_df.rename(columns={"label_2" : "label"}, inplace=True)
label_2_df['annotator'] = '2'

label_0_df_1 = all_labels_df_1[['label_0', 'custom_id']]
label_0_df_1.rename(columns={"label_0" : "label"}, inplace=True)
label_0_df_1['annotator'] = '0'
label_1_df_1 = all_labels_df_1[['label_1', 'custom_id']]
label_1_df_1.rename(columns={"label_1" : "label"}, inplace=True)
label_1_df_1['annotator'] = '1'
label_2_df_1 = all_labels_df_1[['label_2', 'custom_id']]
label_2_df_1.rename(columns={"label_2" : "label"}, inplace=True)
label_2_df_1['annotator'] = '2'

merge_0_df = merge_labels_df[['label_0', 'custom_id']]
merge_0_df.rename(columns={"label_0" : "label"}, inplace=True)
merge_0_df['annotator'] = '0'
merge_1_df = merge_labels_df[['label_1', 'custom_id']]
merge_1_df.rename(columns={"label_1" : "label"}, inplace=True)
merge_1_df['annotator'] = '1'
merge_2_df = merge_labels_df[['label_2', 'custom_id']]
merge_2_df.rename(columns={"label_2" : "label"}, inplace=True)
merge_2_df['annotator'] = '2'

merge_0_df_1 = merge_labels_df_1[['label_0', 'custom_id']]
merge_0_df_1.rename(columns={"label_0" : "label"}, inplace=True)
merge_0_df_1['annotator'] = '0'
merge_1_df_1 = merge_labels_df_1[['label_1', 'custom_id']]
merge_1_df_1.rename(columns={"label_1" : "label"}, inplace=True)
merge_1_df_1['annotator'] = '1'
merge_2_df_1 = merge_labels_df_1[['label_2', 'custom_id']]
merge_2_df_1.rename(columns={"label_2" : "label"}, inplace=True)
merge_2_df_1['annotator'] = '2'

krippendorff_df = pandas.concat([label_0_df, label_1_df, label_2_df])
krippendorff_df.sort_values(by='custom_id', inplace=True)

krippendorff_df_1 = pandas.concat([label_0_df_1, label_1_df_1, label_2_df_1])
krippendorff_df_1.sort_values(by='custom_id', inplace=True)

krippendorff_merge = pandas.concat([merge_0_df, merge_1_df, merge_2_df])
krippendorff_merge.sort_values(by='custom_id', inplace=True)

krippendorff_merge_1 = pandas.concat([merge_0_df_1, merge_1_df_1, merge_2_df_1])
krippendorff_merge_1.sort_values(by='custom_id', inplace=True)

labels_path = (validation_path / "validation_labels.csv").resolve()
labels_path_1 = (validation_path / "validation_labels_1.csv").resolve()
merge_path = (validation_path / "merge_validation_labels.csv").resolve()
merge_path_1 = (validation_path / "merge_validation_labels_1.csv").resolve()

write_to = (validation_path / "krippendorff_df.csv").resolve()
write_to_1 = (validation_path / "krippendorff_df_1.csv").resolve()
merge_write = (validation_path / "krippendorff_merge.csv").resolve()
merge_write_1 = (validation_path / "krippendorff_merge_1.csv").resolve()

all_labels_df.to_csv(path_or_buf=labels_path, index=False)
all_labels_df_1.to_csv(path_or_buf=labels_path_1, index=False)
merge_labels_df.to_csv(path_or_buf=merge_path, index=False)
merge_labels_df_1.to_csv(path_or_buf=merge_path_1, index=False)

krippendorff_df.to_csv(path_or_buf=write_to, index=False)
krippendorff_df_1.to_csv(path_or_buf=write_to_1, index=False)
krippendorff_merge.to_csv(path_or_buf=merge_write, index=False)
krippendorff_merge_1.to_csv(path_or_buf=merge_write_1, index=False)
#print(krippendorff_df_1)

#print(krippendorff_df)
krippendorff_alpha = simpledorff.calculate_krippendorffs_alpha_for_df(krippendorff_df, experiment_col='custom_id',annotator_col='annotator',class_col='label')
krippendorff_alpha_1 = simpledorff.calculate_krippendorffs_alpha_for_df(krippendorff_df_1, experiment_col='custom_id', annotator_col='annotator', class_col='label')
print("Validation set 0:",krippendorff_alpha, "Validation set 1:", krippendorff_alpha_1)

krippendorff_merge = simpledorff.calculate_krippendorffs_alpha_for_df(krippendorff_merge, experiment_col='custom_id',annotator_col='annotator',class_col='label')
krippendorff_merge_1 = simpledorff.calculate_krippendorffs_alpha_for_df(krippendorff_merge_1, experiment_col='custom_id',annotator_col='annotator',class_col='label')
print("Merge set 0:", krippendorff_merge, "Merge set 1:", krippendorff_merge_1)
