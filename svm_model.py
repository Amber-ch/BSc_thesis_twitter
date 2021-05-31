import csv
import os
from pathlib import Path
import pandas
import glob
import numpy as np
from tqdm import tqdm
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from mlxtend.plotting import plot_decision_regions
from sklearn import preprocessing
from sklearn import utils
from sklearn.decomposition import PCA as sklearnPCA
import plotly.express as px

# ALL DATA
# MULTICLASS
"""{'fit_time': array([15.73047519, 17.26012921, 15.36619473, 19.10246968, 17.75897312]), 'score_time': array([3.45899391, 3.00638485, 2.9995048 , 3.37442112, 3.05482578]), 'test_accuracy': array([0.36263736, 0.35409035, 0.36568987, 0.35347985, 0.34270006]), 'test_recall_micro': array([0.36263736, 0.35409035, 0.36568987, 0.35347985, 0.34270006]), 'test_recall_macro': array([0.32833236, 0.33194634, 0.33541232, 0.33082764, 0.31736656]), 'test_f1_micro': array([0.36263736, 0.35409035, 0.36568987, 0.35347985, 0.34270006]), 'test_f1_macro': array([0.30560759, 0.32335838, 0.3098009 , 0.32467261, 0.30323775])}
"""

# BINARY
"""{'fit_time': array([5.49048853, 4.96000552, 4.58262348, 4.65388346, 4.78449154]), 'score_time': array([1.1687243 , 1.22753811, 1.04389405, 1.0715313 , 1.11091828]), 'test_accuracy': array([0.46092796, 0.48412698, 0.46886447, 0.47680098, 0.49908369]), 'test_recall_micro': array([0.46092796, 0.48412698, 0.46886447, 0.47680098, 0.49908369]), 'test_recall_macro': array([0.49031062, 0.47917439, 0.48315527, 0.50883058, 0.51761079]), 'test_f1_micro': array([0.46092796, 0.48412698, 0.46886447, 0.47680098, 0.49908369]), 'test_f1_macro': array([0.45426432, 0.46534585, 0.46237216, 0.47518117, 0.49217036])}
"""

# VALIDATED DATA
# MULTICLASS
"""{'fit_time': array([0.1094234 , 0.10933995, 0.10400534, 0.10066509, 0.10766673]), 'score_time': array([0.04234672, 0.04127383, 0.04148579, 0.04400086, 0.04206157]), 'test_accuracy': array([0.41732283, 0.38095238, 0.44444444, 0.42857143, 0.46825397]), 'test_recall_micro': array([0.41732283, 0.38095238, 0.44444444, 0.42857143, 0.46825397]), 'test_recall_macro': array([0.34893083, 0.34464942, 0.32440838, 0.33965315, 0.43518519]), 'test_f1_micro': array([0.41732283, 0.38095238, 0.44444444, 0.42857143, 0.46825397]), 'test_f1_macro': array([0.33120849, 0.33537939, 0.31671021, 0.33454325, 0.40079289])}
"""

# BINARY
"""{'fit_time': array([0.04022789, 0.03042603, 0.02923608, 0.02849388, 0.02926564]), 'score_time': array([0.02234769, 0.01845002, 0.01842356, 0.01951742, 0.01791263]), 'test_accuracy': array([0.51181102, 0.51587302, 0.51587302, 0.57936508, 0.58730159]), 'test_recall_micro': array([0.51181102, 0.51587302, 0.51587302, 0.57936508, 0.58730159]), 'test_recall_macro': array([0.51191151, 0.51920615, 0.48016304, 0.55432099, 0.57738095]), 'test_f1_micro': array([0.51181102, 0.51587302, 0.51587302, 0.57936508, 0.58730159]), 'test_f1_macro': array([0.49525641, 0.51510946, 0.48021911, 0.55226282, 0.56586271])}
"""

# Import dataset (all data)
file_path = (Path(__file__).parent / "data/features/notext_features.csv").resolve()
dataset = pandas.read_csv(file_path)
features_no_label = dataset[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'intensity']]
features_and_annotation = dataset[['annotation', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'intensity']]
target_label = dataset[['annotation']].astype(int)
merged_target_label = target_label.replace(to_replace=1, value=0)
merged_target_label.replace(to_replace=2, value=1, inplace=True)

# Import dataset (validated)
val_file_path = (Path(__file__).parent / "data/features/notext_validated_features.csv").resolve()
val_dataset = pandas.read_csv(val_file_path)
val_features_no_label = val_dataset[['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'intensity']]
val_features_and_annotation = val_dataset[['annotation', 'anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust', 'intensity']]
val_target_label = val_dataset[['annotation']].astype(int)
val_merged_target_label = val_target_label.replace(to_replace=1, value=0)
val_merged_target_label.replace(to_replace=2, value=1, inplace=True)



# Plot data to determine shape of kernel
def PCA_plot_data():
    norm_feature_train = (feature_train - feature_train.min())/(feature_train.max() - feature_train.min())
    pca = sklearnPCA(n_components=2)
    transformed = pandas.DataFrame(pca.fit_transform(norm_feature_train))
    transformed = transformed.join(target_train)
    print("transformed", transformed)
    hatespeech_df = transformed.loc[transformed['annotation']==0]
    offensive_df = transformed.loc[transformed['annotation']==1]
    neither_df = transformed.loc[transformed['annotation']==2]
    print(hatespeech_df, offensive_df, neither_df)
    plt.scatter(hatespeech_df[0], hatespeech_df[1], label="Hate speech", c='red')
    plt.scatter(offensive_df[0], offensive_df[1], label="Offensive", c='blue')
    plt.scatter(neither_df[0], neither_df[1], label="Neither", c='green')
    #plt.autoscale()
    plt.legend()
    plt.show()

def PCA_plot_merged():
    norm_feature_train = (feature_train - feature_train.min())/(feature_train.max() - feature_train.min())
    pca = sklearnPCA(n_components=2)
    transformed = pandas.DataFrame(pca.fit_transform(norm_feature_train))
    transformed = transformed.join(target_train)
    print("transformed", transformed)
    hatespeech_df = transformed.loc[transformed['annotation']==0]
    offensive_df = transformed.loc[transformed['annotation']==1]
    merged_df = pandas.concat([offensive_df, hatespeech_df])
    merged_df.replace(to_replace=1, value=0, inplace=True)
    neither_df = transformed.loc[transformed['annotation']==2]
    neither_df.replace(to_replace=2, value=1, inplace=True)
    #print(hatespeech_df, offensive_df, neither_df)
    plt.scatter(merged_df[0], merged_df[1], label="Hate speech/offensive", c='red')
    plt.scatter(neither_df[0], neither_df[1], label="Neither", c='green')
    #plt.autoscale()
    plt.legend()
    plt.show()

# Plot validated data to determine shape of kernel
def val_PCA_plot_data():
    val_norm_feature_train = (val_feature_train - val_feature_train.min())/(val_feature_train.max() - val_feature_train.min())
    pca = sklearnPCA(n_components=2)
    val_transformed = pandas.DataFrame(pca.fit_transform(val_norm_feature_train))
    val_transformed = val_transformed.join(val_target_train)
    #print("transformed", transformed)
    hatespeech_df = val_transformed.loc[val_transformed['annotation']==0]
    offensive_df = val_transformed.loc[val_transformed['annotation']==1]
    neither_df = val_transformed.loc[val_transformed['annotation']==2]
    #print(hatespeech_df, offensive_df, neither_df)
    plt.scatter(hatespeech_df[0], hatespeech_df[1], label="Hate speech", c='red')
    plt.scatter(offensive_df[0], offensive_df[1], label="Offensive", c='blue')
    plt.scatter(neither_df[0], neither_df[1], label="Neither", c='green')
    #plt.autoscale()
    plt.legend()
    plt.show()

def val_PCA_plot_merged():
    norm_feature_train = (val_feature_train - val_feature_train.min())/(val_feature_train.max() - val_feature_train.min())
    pca = sklearnPCA(n_components=2)
    transformed = pandas.DataFrame(pca.fit_transform(norm_feature_train))
    transformed = transformed.join(val_target_train)
    #print("transformed", transformed)
    hatespeech_df = transformed.loc[transformed['annotation']==0]
    offensive_df = transformed.loc[transformed['annotation']==1]
    merged_df = pandas.concat([offensive_df, hatespeech_df])
    merged_df.replace(to_replace=1, value=0, inplace=True)
    neither_df = transformed.loc[transformed['annotation']==2]
    neither_df.replace(to_replace=2, value=1, inplace=True)
    #print(hatespeech_df, offensive_df, neither_df)
    plt.scatter(merged_df[0], merged_df[1], label="Hate speech/offensive", c='red')
    plt.scatter(neither_df[0], neither_df[1], label="Neither", c='green')
    #plt.autoscale()
    plt.legend()
    plt.show()


# Simple train/test split of 90/10
feature_train, feature_test, target_train, target_test = train_test_split(features_no_label, target_label, test_size=0.1, random_state=109)
merged_feature_train, merged_feature_test, merged_target_train, merged_target_test = train_test_split(features_no_label, merged_target_label, test_size=0.1, random_state=109)

val_feature_train, val_feature_test, val_target_train, val_target_test = train_test_split(val_features_no_label, val_target_label, test_size=0.1, random_state=109)
val_merged_feature_train, val_merged_feature_test, val_merged_target_train, val_merged_target_test = train_test_split(val_features_no_label, val_merged_target_label, test_size=0.1, random_state=109)


# K-fold cross validation (k = 5), leave out 10% as per Davidson
kf = KFold(n_splits=5, random_state=109, shuffle=True)

def multiclass_svm_classifier():
    classifier = OneVsRestClassifier(svm.SVC(decision_function_shape='ovr', class_weight='balanced'))
    classifier.fit(feature_train, target_train)
    scores = cross_validate(classifier, feature_train, target_train, scoring=['accuracy','recall_micro', 'recall_macro','f1_micro', 'f1_macro'], cv=kf)
    predictions = classifier.predict(feature_test)
    print(scores)
    labels = ['Hate speech', 'Offensive', 'Neither']
    disp = plot_confusion_matrix(classifier, feature_test, target_test, display_labels=labels, cmap=plt.cm.Blues, normalize='true')
    plt.show()

def binary_svm_classifier():
    classifier = svm.SVC(class_weight='balanced')
    classifier.fit(merged_feature_train, merged_target_train)
    scores = cross_validate(classifier, merged_feature_train, merged_target_train, scoring=['accuracy','recall_micro', 'recall_macro','f1_micro', 'f1_macro'], cv=kf)
    predictions = classifier.predict(merged_feature_test)
    print(predictions)
    print(scores)
    labels = ['Hate speech/offensive', 'Neither']
    norm_disp = plot_confusion_matrix(classifier, merged_feature_test, merged_target_test, display_labels=labels, cmap=plt.cm.Blues, normalize='true')
    plt.show()

def val_multiclass_svm_classifier():
    classifier = OneVsRestClassifier(svm.SVC(decision_function_shape='ovr', class_weight='balanced'))
    classifier.fit(val_feature_train, val_target_train)
    scores = cross_validate(classifier, val_feature_train, val_target_train, scoring=['accuracy','recall_micro', 'recall_macro','f1_micro', 'f1_macro'], cv=kf)
    predictions = classifier.predict(val_feature_test)
    print(scores)
    labels = ['Hate speech', 'Offensive', 'Neither']
    disp = plot_confusion_matrix(classifier, val_feature_test, val_target_test, display_labels=labels, cmap=plt.cm.Blues, normalize='true')
    plt.show()

def val_binary_svm_classifier():
    classifier = svm.SVC(class_weight='balanced')
    classifier.fit(val_merged_feature_train, val_merged_target_train)
    scores = cross_validate(classifier, val_merged_feature_train, val_merged_target_train, scoring=['accuracy','recall_micro', 'recall_macro','f1_micro', 'f1_macro'], cv=kf)
    predictions = classifier.predict(val_merged_feature_test)
    print(predictions)
    print(scores)
    labels = ['Hate speech/offensive','Neither']
    norm_disp = plot_confusion_matrix(classifier, val_merged_feature_test, val_merged_target_test, display_labels=labels, cmap=plt.cm.Blues, normalize='true')
    plt.show()


if __name__ == "__main__":
    # Toggle for the needed function

    #PCA_plot_data()
    #PCA_plot_merged()
    #val_PCA_plot_data()
    #val_PCA_plot_merged()
    #multiclass_svm_classifier()
    #binary_svm_classifier()
    val_multiclass_svm_classifier()
    #val_binary_svm_classifier()