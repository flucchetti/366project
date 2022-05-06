import json
import gzip
import pickle
import re
import random
import math
import nltk
import spacy
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import timeit
import string
from nltk.metrics import precision, recall, f_measure
import collections
import itertools
from features import *
from partition import *

'''
This module contains all classifiers
'''

fp = open("data/stop_words.txt", "r")
stop_words = [line.rstrip() for line in fp.readlines()]
# print(stop_words)

# ## init spaCy tokenizer
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
# nlp = English()
# # Create a Tokenizer with the default settings for English
# # including punctuation rules and exceptions
# tokenizer = nlp.tokenizer

# from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")




def classify_pos(train_sents, test_sents, devtest_sents, get_feats=pos_feats):
    '''
    Builds BOW calssifier (SVC or Naive) and prints accuracy
    or most informative feats
    '''
    # get feats
    # test_feats = [(get_feats(sent), label) for sent, label in test_sents if get_feats(sent) != {}]
    devtest_feats = [(get_feats(sent), label) for sent, label in devtest_sents  if get_feats(sent) != {}]
    train_feats = [(get_feats(sent), label) for sent, label in train_sents  if get_feats(sent) != {}]

    # print(devtest_feats[:40])

    # BOW with counts
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform([" ".join(get_feats(sent)) for sent, label in train_sents])
    y_train = [label for sent, label in train_sents]

    X_test_counts = count_vect.transform([" ".join(get_feats(sent)) for sent, label in devtest_sents])
    y_test = [label for sent, label in devtest_sents]

    classify_bowc = LinearSVC()
    classify_bowc.fit(X_train_counts, y_train)

    y_test_predicted = classify_bowc.predict(X_test_counts)

    print("Multinomial results")
    print(metrics.classification_report(y_test, y_test_predicted))

    print(metrics.accuracy_score(y_test, y_test_predicted))




def classify_bow(train_sents, test_sents, devtest_sents, get_feats=bow_feats):
    '''
    Builds BOW calssifier (SVC or Naive) and prints accuracy
    or most informative feats
    '''
    # get feats
    # test_feats = [(get_feats(sent), label) for sent, label in test_sents ]
    devtest_feats = [(get_feats(sent), label) for sent, label in devtest_sents ]
    train_feats = [(get_feats(sent), label) for sent, label in train_sents ]

    #print(devtest_feats[:40])

    # classify_bow = nltk.classify.SklearnClassifier(LinearSVC())
    classify_bow = nltk.NaiveBayesClassifier.train(train_feats)

    classify_bow.train(train_feats)

    print("NLTK Naive Bayes:")
    # accuracy = nltk.classify.accuracy(classify_bow, devtest_feats)
    # print("Accuracy score:", accuracy)

    classify_bow.show_most_informative_features(40)

    ## calculate precision/recall/f1
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(devtest_sents):
        refsets[label].add(i)
        observed = classify_bow.classify(bow_feats(feats))
        testsets[observed].add(i)

    print("For label 1:")
    label = 1
    print( 'Precision:', precision(refsets[label], testsets[label]) )
    print( 'Recall:', recall(refsets[label], testsets[label]) )
    print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )
    print("For label 0:")
    label = 0
    print( 'Precision:', precision(refsets[label], testsets[label]) )
    print( 'Recall:', recall(refsets[label], testsets[label]) )
    print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )
    
    



def classify_bowc(train_sents, test_sents, devtest_sents):
    '''
    Builds BOW with counts calssifier (tf-idf weighing)
    '''
    # BOW with counts
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform([" ".join(sent) for sent, label in train_sents])
    y_train = [label for sent, label in train_sents]

    X_test_counts = count_vect.transform([" ".join(sent) for sent, label in devtest_sents])
    y_test = [label for sent, label in devtest_sents]


    classify_bowc = MultinomialNB()
    classify_bowc.fit(X_train_counts, y_train)

    y_test_predicted = classify_bowc.predict(X_test_counts)

    print("Multinomial results")
    print(metrics.classification_report(y_test, y_test_predicted))

    print(metrics.accuracy_score(y_test, y_test_predicted))



if __name__=="__main__":
    '''
    Loads and partitions data into train-test-devtest
    calls classifiers
    '''
 
    start = timeit.default_timer()

    ## dataset
    ds = 'data/shuffled_data.json.gz'
    num_entries = 1000
    data = load_sents(ds, num_entries)
    
    stop = timeit.default_timer()
    print('Time for loading data: ', stop - start) 

    
    train_sents, test_sents, devtest_sents = partition(data)

    classify_bow(train_sents, test_sents, devtest_sents)

    classify_bowc(train_sents, test_sents, devtest_sents)
    # classify_pos(train_sents, test_sents, devtest_sents)



    