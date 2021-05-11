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
from nltk.stem.snowball import SnowballStemmer

# TODO rewrite so each step happens for 1 tweet at a time, instead of each step for all tweets
# DONE validation sets majority vote, remove those without a majority
# DONE remove tweets with label 3 (it means duplicate)
# DONE remove tweets that are duplicate due to validation
# DONE add tweets back in cuz im a dumbass
# DONE data cleaning/preprocessing steps (ngram identification & lemmatization)
# DONE lookup scores with EmoLex
# DONE lookup anger intensity score

class FeatureSelection(object):
    test_corpus = None
    base_folder = None
    data_folder = None
    emolex_translation = None
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
    freq_df = None
    freq_list = []
    most_freq_list = []
    freq_dict = {}
    most_freq_dict = {}
    most_relevant = None
    most_relevant_dict = {}
    __corpus = None
    __idf_values = {}
    __tf_values = {}
    __tf_idf_values = None
    emolex_translation_pivot = None
    emolex_translation_df = None
    __emolex_scores = None
    emolex_header = []
    emotions = []
    intensity_lexicon = None

    def __init__(self):
        self.base_folder = Path(__file__).parent
        self.data_folder = (self.base_folder / "data/tweets/filtered").resolve()

        # Toggle for testing!
        #file_path = (self.base_folder / "data/tweets/filtered/labelled_test_set.csv").resolve()
        file_path = (self.base_folder / "data/tweets/filtered/labelled_tweets_ALL.udt.csv").resolve()

        validation_path = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels.csv").resolve()
        validation_path_1 = (self.base_folder / "data/tweets/filtered/validation/merged/validation_labels_1.csv").resolve()
        stop_words_1_path = (self.base_folder / "data/stop-words_dutch_1_nl.txt")
        stop_words_2_path = (self.base_folder / "data/stop-words_dutch_2_nl.txt")
        emolex_path = (self.base_folder / "data/NRC-Emotion-Lexicon/NRC-Emotion-Lexicon-v0.92").resolve()
        emolex_english = (emolex_path/"wordlevel.txt").resolve()
        self.emolex_translation = (emolex_path/"translation.xlsx").resolve()

        self.emolex_header = ["index", "english", "dutch", "anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]
        self.emotions = ["anger", "anticipation", "disgust", "fear", "joy", "negative", "positive", "sadness", "surprise", "trust"]

        self.__emolex_scores = pandas.DataFrame(columns=self.emolex_header)
        nltk_stopwords = stopwords.words("dutch")
        additional_stopwords = pandas.read_csv(stop_words_1_path, sep=" ", names=["stopword"])
        additional_stopwords = additional_stopwords['stopword'].tolist()
        additional_stopwords_2 = pandas.read_csv(stop_words_2_path, sep=" ", names=["stopword"])
        additional_stopwords_2 = additional_stopwords_2['stopword'].tolist()
        #print("additional", additional_stopwords['stopword'].tolist())
        all_stopwords = list(set(nltk_stopwords + additional_stopwords + additional_stopwords_2))
        self.__stop_words = all_stopwords
        #print(len(all_stopwords), all_stopwords)
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
        write_to = (self.data_folder/"remove_duplicate_dataset.csv").resolve()
        self.__all_data.to_csv(path_or_buf=write_to, index=False)
        print(self.__all_data)

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


    def save_words(self, wordtype):
        if wordtype == 'frequent':
            filename = "frequent_words.csv"
            path = (self.base_folder / "data" / filename).resolve()
            self.freq_df.to_csv(path_or_buf=path, index=False)
        elif wordtype == 'relevant':
            filename = "relevant_words.csv"
            path = (self.base_folder / "data" / filename).resolve()
            self.most_relevant.to_csv(path_or_buf=path, index=False)
    
    def save_features(self, feature):
        if feature == 'all':
            filename = "all_features.csv"
        elif feature == 'preprocess':
            filename = "preprocessed.csv"
        elif feature == 'emolex':
            filename = "emolex.csv"
        write_to = (self.base_folder / "data" / filename).resolve()
        self.__all_data.to_csv(path_or_buf=write_to, index=True)
        print(self.__all_data)

    # Convert document column to list, preprocess, convert back to dataframe and join full dataframe
    def preprocess(self):
        documents = self.__all_data['document'].to_list()
        #print("docs", documents)
        for tweet in tqdm(documents):
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
        self.save_features('preprocess')

    # Returns a list of N most frequent terms with their counts
    def identify_frequent_words(self):
        self.__corpus = self.__all_data['processed']
        #print(corpus)
        for sentence in tqdm(self.__corpus):
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
        self.freq_df = pandas.DataFrame(self.freq_dict.items(), columns=["word", "count"])
        self.most_freq_dict = dict(islice(self.freq_dict.items(), 0,29))
        print(self.most_freq_dict)
        #print(self.most_freq_dict)

        # Returns just the N most common words as a list
        self.freq_list = sorted(self.wordfreq, key=self.wordfreq.get, reverse=True)
        #print(self.freq_list)
        self.most_freq_list = heapq.nlargest(20, self.wordfreq, key=self.wordfreq.get)
        #print(self.most_freq_list)
        self.save_words('frequent')

    # Returns a list of the N most relevant terms with their TF-IDF scores
    def identify_relevant_words(self):
        docs = self.__all_data['processed']
        docs = [' '.join(list(docs))]
        #print(' '.join(list(docs)))
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2), use_idf=True)
        tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)
        words_df = pandas.DataFrame(tfidf_vectorizer.get_feature_names(), columns=["word"])
        scores_df = pandas.DataFrame(tfidf_vectorizer_vectors[0].T.todense(), columns=["tfidf"])
        full_df = scores_df.join(words_df)
        full_df.sort_values(by=["tfidf"], ascending=False, inplace=True)
        self.most_relevant = full_df
        print(full_df.head(30))
        self.save_words('relevant')

    def setup_emolex_initial(self):
        self.emolex_translation_df = pandas.read_excel(self.emolex_translation, usecols=["English (en)", "Dutch (nl)", "Positive", "Negative", "Anger", "Anticipation", "Disgust", "Fear", "Joy", "Sadness", "Surprise", "Trust"])
        self.emolex_translation_df = self.emolex_translation_df.rename(columns={"English (en)": "english", "Dutch (nl)": "dutch"})
        self.emolex_translation_df.to_excel(self.emolex_translation)
        print(self.emolex_translation_df.head())

    def setup_emolex(self):
        self.emolex_translation_df = pandas.read_excel(self.emolex_translation)
        #self.emolex_translation_df = self.emolex_translation_df.rename(columns={"Unnamed: 0": "index"})
        self.emolex_translation_df.columns = self.emolex_header
        print(self.emolex_translation_df.head())

    def get_emolex_score(self):
        # Subset for testing
        self.test_corpus = self.__corpus.head(20)
        self.setup_emolex()
        # Used when the word cannot be found in the lexicon
        # TODO paper: explain that lemmatizing can change the meaning, stemming does this to a lesser degree
        stemmer = SnowballStemmer("dutch")
        for sentence in tqdm(self.__corpus):
            print(sentence)
            sentence_words_df = pandas.DataFrame(columns=self.emolex_header)
            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                emolex_token_dutch = self.emolex_translation_df.loc[self.emolex_translation_df['dutch'] == token]
                emolex_token_english = self.emolex_translation_df.loc[self.emolex_translation_df['english'] == token]

                if (emolex_token_dutch.empty == False) | (emolex_token_english.empty == False):
                    sentence_words_df = pandas.concat([sentence_words_df, emolex_token_dutch])
                    sentence_words_df = pandas.concat([sentence_words_df, emolex_token_english])
                else:
                    stemmed_token = stemmer.stem(token)
                    emolex_stem_dutch = self.emolex_translation_df.loc[self.emolex_translation_df['dutch'] == stemmed_token]
                    emolex_stem_english = self.emolex_translation_df.loc[self.emolex_translation_df['english'] == stemmed_token]
                    sentence_words_df = pandas.concat([sentence_words_df, emolex_stem_dutch])
                    sentence_words_df = pandas.concat([sentence_words_df, emolex_stem_english])
            sentence_words_df.drop_duplicates(subset=['english'], inplace=True)
            print(sentence_words_df)
            print("HERE")
            sentence_sum = sentence_words_df.sum(axis=0)
            print("AFTER SUM")
            print(sentence_sum)
            sentence_sum = list(sentence_sum)
            print(len(sentence_sum))
            if len(sentence_sum) == 13:
                sentence_sum = pandas.DataFrame([sentence_sum], columns=self.emolex_header)
            else:
                sentence_sum = pandas.DataFrame(np.zeros((13, 13)), columns=self.emolex_header)
            print(sentence_sum)
            self.__emolex_scores = pandas.concat([self.__emolex_scores, sentence_sum], ignore_index=True)
        self.__emolex_scores = self.__emolex_scores.drop(columns=["index", "dutch", "english"])
        self.__emolex_scores = self.__emolex_scores.dropna()
        print(self.__emolex_scores)
        self.__all_data = self.__all_data.join(self.__emolex_scores)
        print(self.__all_data.head())
        self.save_features('emolex')

    def setup_intensity(self):
        intensity_path = (self.base_folder / "data/NRC-Emotion-Intensity-Lexicon-v1/OneFilePerLanguage/Dutch-nl-NRC-Emotion-Intensity-Lexicon-v1.txt").resolve()
        self.intensity_lexicon = pandas.read_csv(intensity_path, sep='\t')
        self.intensity_lexicon = self.intensity_lexicon.loc[self.intensity_lexicon['emotion'] == 'anger']
        print(self.intensity_lexicon)

    def get_intensity_score(self):
        self.setup_intensity()
        stemmer = SnowballStemmer("dutch")
        intensity_list = []

        for sentence in tqdm(self.__corpus):
            collect_scores_df = pandas.DataFrame(columns=["word", "Dutch-nl", "emotion", "emotion-intensity-score"])

            tokens = nltk.word_tokenize(sentence)
            for token in tokens:
                intensity_token_nl = self.intensity_lexicon.loc[self.intensity_lexicon['Dutch-nl'] == token]
                intensity_token_en = self.intensity_lexicon.loc[self.intensity_lexicon['word'] == token]
                
                if (intensity_token_nl.empty == False) | (intensity_token_en.empty == False):
                    collect_scores_df = pandas.concat([collect_scores_df, intensity_token_nl])
                    collect_scores_df = pandas.concat([collect_scores_df, intensity_token_en])
                else:
                    stemmed_token = stemmer.stem(token)
                    intensity_stem_nl = self.intensity_lexicon.loc[self.intensity_lexicon['Dutch-nl'] == stemmed_token]
                    intensity_stem_en = self.intensity_lexicon.loc[self.intensity_lexicon['word'] == stemmed_token]
                    collect_scores_df = pandas.concat([collect_scores_df, intensity_stem_nl])
                    collect_scores_df = pandas.concat([collect_scores_df, intensity_stem_en])
            collect_scores_df.drop_duplicates(subset=['word'], inplace=True)
            num_words = len(collect_scores_df)
            print(num_words)
            print(collect_scores_df)
            collect_scores_df = collect_scores_df.drop(columns=["word", "Dutch-nl", "emotion"])
            print(collect_scores_df)
            collect_scores_df = collect_scores_df.sum(axis=0)
            try:
                anger_intensity = collect_scores_df.iloc[0] / num_words
            except RuntimeWarning:
                anger_intensity = 0
            print("intensity score:", anger_intensity)
            intensity_list.append(anger_intensity)
            #print("sum", collect_scores_df)
            #collect_scores_df = collect_scores_df / num_words
            #print("score:", collect_scores_df)
        intensity_df = pandas.DataFrame(intensity_list, columns=['intensity'])
        intensity_df['intensity'] = intensity_df['intensity'].fillna(0)
        self.__all_data = self.__all_data.join(intensity_df)
        self.save_features('all')
        #print(self.__all_data.head(20))

    def id_toint(self):
        all_path = (self.base_folder / "data/features/all_features.csv").resolve()
        notext_path = (self.base_folder / "data/features/notext_features.csv").resolve()
        colnames_all = ["custom_id","annotation","unprocessed","processed","anger","anticipation","disgust","fear","joy","negative","positive","sadness","surprise","trust","intensity"]
        colnames_notext = ["custom_id","annotation","anger","anticipation","disgust","fear","joy","negative","positive","sadness","surprise","trust","intensity"]
        all_features_df = pandas.read_csv(all_path)
        all_features_df['custom_id'] = all_features_df['custom_id'].astype(int)
        #all_features_df = all_features_df.drop(columns="Unnamed: 0")
        print(all_features_df)
        notext_df = all_features_df.drop(columns=["unprocessed", "processed"])
        print(notext_df)
        all_features_df.to_csv(path_or_buf=all_path, index=False, columns=colnames_all)
        notext_df.to_csv(path_or_buf=notext_path, index=False, columns=colnames_notext)


if __name__ == "__main__":
    featselect = FeatureSelection()
    #featselect.merge_all()
    #featselect.remove_duplicate()
    #featselect.preprocess()
    #featselect.identify_frequent_words()
    #featselect.identify_relevant_words()
    #featselect.get_emolex_score()
    #featselect.get_intensity_score()
    featselect.id_toint()