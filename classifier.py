import re
import math
import collections
# from features import *
# from partition import *
import string
import pandas as pd
from  numpy import shape
import scipy as sp
from zlib import crc32

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

import nltk
from nltk.corpus import stopwords

from nltk.metrics import precision, recall, f_measure

from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

stop_words = list(stopwords.words('english'))

'''
This module contains all classifiers
'''

def tokenizer(s):
    t = re.split(r"(\W)", s)
    tokens = [i for i in t 
            if i not in ["\t","\n","\r","\f","\v"," ",""]
            and (i not in string.punctuation)
            # and (i not in stop_words)
            ]
    return tokens

def bigram_feats(sent, need_tokenize=True):
    '''
    '''
    if need_tokenize:
        sent = tokenizer(str(sent))

    bigrams = list(zip(sent[:-1], sent[1:]))

    return {'-'.join(b):1 for b in bigrams}


def pos_feats(sent, need_tokenize=True):
    '''
    Returns bag of words feature (simple no counts just flag)
    WITH TOK
    '''
    bow = {}
    if need_tokenize:
        sent = tokenizer(str(sent))

    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow


def bow_feats(sent, need_tokenize=True):
    '''
    Returns bag of words feature (simple no counts just flag)
    '''
    bow = {}
    if need_tokenize:
        sent = tokenizer(str(sent))
        
    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow


def classify_bow_NB(csv_file, get_feats=bow_feats):
    '''
    Builds BOW calssifier (SVC or Naive) and prints accuracy
    or most informative feats
    '''
    print("classify_bow_NB")
    df = pd.read_csv(csv_file)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)

    X = df['sent']
    y = df["s_spoiler"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    test = list(zip(X_test, y_test))
    train = list(zip(X_train, y_train))

    test_feats = [(get_feats(sent), label) for sent, label in test]
    # devtest_feats = [(get_feats(sent), label) for sent, label in devtest]
    train_feats = [(get_feats(sent), label) for sent, label in train]

    # print(test_feats[:10])

    classify = nltk.NaiveBayesClassifier.train(train_feats)


    print("NLTK Naive Bayes:")

    classify.show_most_informative_features(40)

    ## METRICS
    ## calculate precision/recall/f1
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test):
        """ makes dicts refsets/testsets where for ex:
        {0: {<sents> 1, 3, 42}, 1: {2, 6, 55}}
        recording which sents are at labels 0/1 per
        gold standard vs prediction
        """
        refsets[label].add(i)
        observed = classify.classify(get_feats(feats))
        testsets[observed].add(i)


    for label in [0,1]:
        print("\nFor label", label, ":")
        #The support is the number of occurrences of each class in y_true
        print("support: ", len(refsets[label]))
        # print("support: ", len(testsets[label]))
        prec = precision(refsets[label], testsets[label])
        print( 'Precision:', prec )
        rec = recall(refsets[label], testsets[label])
        print( 'Recall:', rec )
        print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )

    acc = nltk.classify.accuracy(classify, test_feats)
    print("Accuracy score:", acc)


def classify_bow_counts2(csv_file, classifier_type=MultinomialNB, ngram_range=(1,1),get_feats=bow_feats):
    print("classify_bow_counts2")
    # BOW with counts
    count_vect = CountVectorizer()

    df = pd.read_csv(csv_file)

    X = df['sent']
    y = df["s_spoiler"]

    # Xy = list(zip(X, y))

    assert(len(X)==len(y)), "XY MISMATCH"

    size = len(X)
    bite = math.floor(size/10)


    X_train_counts = count_vect.fit_transform([str(sent) for sent in X[bite:]])
    y_train = y[bite:]

    print(str(X[0]))

    X_test_counts = count_vect.transform([str(sent) for sent in X[:bite]])
    y_test = y[:bite]


    ## X_counts is document term matrix where column = every feature in set
    ## and row = sent index, and value at row,col is boolean
    ## indicating if word_col appears in sent_row

    classify = classifier_type()
    classify.fit(X_train_counts, y_train)

    y_test_predicted = classify.predict(X_test_counts)

    X_columns=count_vect.get_feature_names()
    print(X_columns[:40])

    print(classifier_type," results")
    print(metrics.classification_report(y_test, y_test_predicted))

    

## SAME AS classify_many_feats(feat_list=[])
def classify_bow_counts(csv_file, classifier_type=LinearSVC, ngram_range=(1,1)):
    '''
    Builds BOW with counts calssifier (tf-idf weighing)
    '''
    print("classify_bow_counts")
    df = pd.read_csv(csv_file)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)

    ## Count vect
    cvect = CountVectorizer(ngram_range=ngram_range)
    X = cvect.fit_transform(df.sent.values.astype('U'))
    X_columns=cvect.get_feature_names()
    print(X_columns[:40])
    y = df["s_spoiler"]

    # classifier = experiment(X,y, classifier_type)
    ## SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ## CLASSIFIER
    classifier = classifier_type()

    ## TRAIN
    classifier.fit(X_train, y_train)

    #####PREDICT UNSEEN
    texts = ["Harry kills Voldemort at the end.", 
        "I never expected it to end with a big wedding",
        "a b c",
        "what a cliffhanger",
        "i don't believe it",
        "another one",
        "and another",
        "this isn't a spoiler",
        "OMG I can't believe J was K's father!"]
    text_features = cvect.transform(texts)
    print("TF:", text_features)
    predictions = classifier.predict(text_features)
    for text, predicted in zip(texts, predictions):
        print('"{}"'.format(text))
        print("  - Predicted as: '{}'".format(predicted))
        print("")



def classify_many_feats(csv_file, feat_list=[], classifier_type=LinearSVC, ngram_range=(1,1)):
    '''
    Classifies with metadata
    feat_list = list of target feats chosen out of:
    ['userID', 'bookID', 'rating', 
    'date_published','authorID', 'genre',
    'sent', 's_loc']
    '''
    print("classify_many_feats")
    df = pd.read_csv(csv_file)
    ## randomize - not needed because split_test_train randomizes
    # df = dataset.sample(frac=1)


    ### GATHER DATA
    df["userID"] = df["userID"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    df["genre"] = df["genre"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    

    # Create vectorizer for function to use
    cvect = CountVectorizer(ngram_range=ngram_range)

    print("FEATURE_LIST: ", feat_list)
    

    X = sp.sparse.hstack((cvect .fit_transform(df.sent.values.astype('U')), df[feat_list].values),format='csr')
    X_columns=cvect.get_feature_names()+df[feat_list].columns.tolist()
    print(X_columns[:40])
    y = df["s_spoiler"]

    experiment(X,y, classifier_type)



def experiment(X, y, classifier_type):
    '''
    Runs training and predictions, print results
    '''
    print("Total size y,X:" ,y.shape[0],X.shape[0])
    
    ## SPLIT
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    ## CLASSIFIER
    classifier = classifier_type()

    ## TRAIN
    classifier.fit(X_train, y_train)

    ## PREDICT
    # print('TEST', X_test[:10])
    y_test_predicted = classifier.predict(X_test)

    ## RESULTS
    print(classifier, " results")
    
    '''
    Print metrics: accuracy, precision, recall, F1, AUC, ROC AUC
    '''
    print(metrics.classification_report(y_test, y_test_predicted))
    print(metrics.accuracy_score(y_test, y_test_predicted))

    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_test, y_test_predicted, pos_label=1)
    pr_auc = auc(recall, precision)
    print("Precision-Recall AUC: %.2f" % pr_auc)
    # ROC AUC
    fpr, tpr, _ = roc_curve(y_test, y_test_predicted, pos_label=1)
    roc_auc = auc(fpr, tpr)
    print("ROC AUC: %.2f" % roc_auc)

    return classifier






if __name__=="__main__":
    '''
    Loads and partitions data into train-test-devtest
    calls classifiers
    '''
    file = 'data/balanced_revs.csv'
    

    # classify_bow_counts2(file)
    classify_bow_counts(file, classifier_type=MultinomialNB)
    # classify_many_feats(file, classifier_type=MultinomialNB)

    # classify_bow_NB(file, get_feats=bow_feats)

    