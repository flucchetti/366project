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
from nltk.metrics import precision, recall, f_measure, accuracy
import collections
import itertools
from features import *
from partition import *

from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

'''
This module contains all classifiers
'''


# def classify_pos(train_sents, test_sents, devtest_sents, get_feats=pos_feats):
#     '''
#     Builds BOW calssifier (SVC or Naive) and prints accuracy
#     or most informative feats
#     '''
#     # get feats
#     # test_feats = [(get_feats(sent), label) for sent, label in test_sents if get_feats(sent) != {}]
#     devtest_feats = [(get_feats(sent), label) for sent, label in devtest_sents  if get_feats(sent) != {}]
#     train_feats = [(get_feats(sent), label) for sent, label in train_sents  if get_feats(sent) != {}]

#     # print(devtest_feats[:40])

#     # BOW with counts
#     count_vect = CountVectorizer()

#     X_train_counts = count_vect.fit_transform([" ".join(get_feats(sent)) for sent, label in train_sents])
#     y_train = [label for sent, label in train_sents]

#     X_test_counts = count_vect.transform([" ".join(get_feats(sent)) for sent, label in devtest_sents])
#     y_test = [label for sent, label in devtest_sents]

#     classify_bowc = LinearSVC()
#     classify_bowc.fit(X_train_counts, y_train)

#     y_test_predicted = classify_bowc.predict(X_test_counts)

#     print("Multinomial results")
#     print(metrics.classification_report(y_test, y_test_predicted))

#     print(metrics.accuracy_score(y_test, y_test_predicted))


def my_accuracy(reference, test):
    '''FOR DEBUG PURPOSE'''
    return sum(x == y for x, y in zip(reference, test)) / len(test)


def classify_bow(train_sents, test_sents, devtest_sents, get_feats=bow_feats):
    '''
    Builds BOW calssifier (SVC or Naive) and prints accuracy
    or most informative feats
    '''
    # get feats
    # test_feats = [(get_feats(sent), label) for sent, label in test_sents ]
    devtest_feats = [(get_feats(sent), label) for sent, label in devtest_sents ]
    train_feats = [(get_feats(sent), label) for sent, label in train_sents ]

    # classify_bow = nltk.classify.SklearnClassifier(LinearSVC()) ##needs training?
    classify_bow = nltk.NaiveBayesClassifier.train(train_feats)

    classify_bow.train(train_feats)

    print("NLTK Naive Bayes:")

    classify_bow.show_most_informative_features(40)

    ## calculate precision/recall/f1
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(devtest_sents):
        """ makes dicts refsets/testsets where for eg:
        {0: {<sents> 1, 3, 42}, 1: {2, 6, 55}}
        recording which sents are at labels 0/1 per
        gold standard vs prediction
        """
        refsets[label].add(i)
        observed = classify_bow.classify(get_feats(feats))
        testsets[observed].add(i)


    for label in [0,1]:
        print("\nFor label", label, ":")
        print( 'Precision:', precision(refsets[label], testsets[label]) )
        print( 'Recall:', recall(refsets[label], testsets[label]) )
        print( 'Accuracy:', my_accuracy(refsets[label], testsets[label]) )
        print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )

    acc = nltk.classify.accuracy(classify_bow, devtest_feats)
    print("Accuracy score:", acc)

    print(metrics.classification_report(refsets, testsets))




def classify_bow_counts(train_sents, test_sents, devtest_sents):
    '''
    Builds BOW with counts calssifier (tf-idf weighing)
    '''
    # BOW with counts
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform([" ".join(sent) for sent, label in train_sents])
    y_train = [label for sent, label in train_sents]

    X_test_counts = count_vect.transform([" ".join(sent) for sent, label in devtest_sents])
    y_test = [label for sent, label in devtest_sents]

    ## X_counts is document term matrix where column = every feature in set
    ## and row = sent index, and value at row,col is boolean
    ## indicating if word_col appears in sent_row

    classify_bowc = MultinomialNB()
    classify_bowc.fit(X_train_counts, y_train)

    y_test_predicted = classify_bowc.predict(X_test_counts)

    print("Multinomial results")
    print(metrics.classification_report(y_test, y_test_predicted))

    print(metrics.accuracy_score(y_test, y_test_predicted))



def classify_bow_loc(train_sents, test_sents, devtest_sents):
    '''
    sents need to be in format:
    (tokens, spoiler_label, location in review)

    featureset:
    for feat in sent: {feat: tag}

    (featureset, label)
    '''
    # classif = SklearnClassifier(LinearSVC())
    pipeline = Pipeline([('tfidf', TfidfTransformer()),
                        ('chi2', SelectKBest(chi2, k=1000)),
                    ('nb', MultinomialNB())])
    classifier = SklearnClassifier(pipeline)

    # BOW with counts
    count_vect = CountVectorizer()

    # X_train_counts = count_vect.fit_transform([" ".join(sent) for sent, label, loc in train_sents])
    # X_train_locs = [loc for sent, label, loc in train_sents ]
    # y_train = [label for sent, label in train_sents]

    train_count_fst = collections.defaultdict(int)
    train_loc_fst = collections.defaultdict(int)
    train_label_fsts = []

    for sent, label, loc in train_sents:
        s = " ".join(sent)
        ### ???
        train_count_fst[s] = count_vect.fit_transform(s) 
        train_loc_fst[s] = loc


    X_test_counts = count_vect.transform([" ".join(sent) for sent, label, loc in devtest_sents])
    X_test_locs = [loc for sent, label, loc in devtest_sents]
    y_test = [label for sent, label in devtest_sents]
    test_featuresets = []

    '''
    def train(self, labeled_featuresets):
        """
        Train (fit) the scikit-learn estimator.

        :param labeled_featuresets: A list of ``(featureset, label)``
            where each ``featureset`` is a dict mapping strings to either
            numbers, booleans or strings.
        """

        X, y = list(zip(*labeled_featuresets))
        X = self._vectorizer.fit_transform(X)
        y = self._encoder.fit_transform(y)
        self._clf.fit(X, y)

        return self

    def classify_many(self, featuresets):
        """Classify a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :return: The predicted class label for each input sample.
        :rtype: list
        """
        X = self._vectorizer.transform(featuresets)
        classes = self._encoder.classes_
        return [classes[i] for i in self._clf.predict(X)]

    def prob_classify_many(self, featuresets):
        """Compute per-class probabilities for a batch of samples.

        :param featuresets: An iterable over featuresets, each a dict mapping
            strings to either numbers, booleans or strings.
        :rtype: list of ``ProbDistI``
        """
        X = self._vectorizer.transform(featuresets)
        y_proba_list = self._clf.predict_proba(X)
        return [self._make_probdist(y_proba) for y_proba in y_proba_list]
    '''
    classifier.train(train_featuresets)
    # classifier.classify_many(test_featuresets)
    probs = classifier.prob_classify_many(test_featuresets)
    print(probs)



if __name__=="__main__":
    '''
    Loads and partitions data into train-test-devtest
    calls classifiers
    '''
 
    start = timeit.default_timer()

    ## dataset
    ds = 'data/young-adult_data.json.gz'
    ## None entries = all entries
    num_entries = 10000
    data = load_sents(ds, num_entries)
    
    stop = timeit.default_timer()
    print('Time for loading data: ', stop - start) 

    print('Results for: ', ds, "\nNum entries: ", num_entries) 

    train_sents, test_sents, devtest_sents = partition(data)

    classify_bow(train_sents, test_sents, devtest_sents)
    # bow_counts(train_sents, test_sents, devtest_sents)
    # classify_pos(train_sents, test_sents, devtest_sents)

    stop2 = timeit.default_timer()
    print('Time total: ', stop2 - start) 

    print("\n####################\n\n")



    