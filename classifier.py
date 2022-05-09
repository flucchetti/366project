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
from  numpy import shape

from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
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
        print( 'F1:', f_measure(refsets[label], testsets[label], alpha=0.5) )

    acc = nltk.classify.accuracy(classify_bow, devtest_feats)
    print("Accuracy score:", acc)




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

    classify_bowc = LinearSVC()
    classify_bowc.fit(X_train_counts, y_train)

    y_test_predicted = classify_bowc.predict(X_test_counts)

    print("Multinomial results")
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



def classify_loc(train_sents, test_sents, devtest_sents, classifier_type=LinearSVC):
    '''
    sents need to be in format:
    (tokens, spoiler_label, location in review)
    '''
    # classif = SklearnClassifier(LinearSVC())
    # pipeline = Pipeline([
    #                     # ('tfidf', TfidfTransformer()),
    #                     # ('chi2', SelectKBest(chi2, k=1000)),
    #                 ('nb', MultinomialNB())])
    # classifier = SklearnClassifier(pipeline)

    # train_fsets = []

    # for sent, label, loc in train_sents:
    #     feats = bow_feats(sent)
    #     # featureset = {f: loc for f in feats}
    #     featureset = {" ".join(sent) : loc}
    #     train_fsets.append((featureset, label))

    # test_fsets = []

    # for sent, label, loc in devtest_sents:
    #     feats = bow_feats(sent)
    #     # featureset = {f: loc for f in feats}
    #     featureset = {" ".join(sent) : loc}
    #     test_fsets.append(featureset)


    # classifier.train(train_fsets)
    # predictions = classifier.classify_many(test_fsets)

    # # print(train_fsets[:10], "\n\n")
    # # print(test_fsets[:10])
    # # print(predictions)

    # if 1 in predictions:
    #     print("predicted spoiler")
    # else:
    #     print("no spoiler predicted")

    # test_gold = [label for sent, label in devtest_sents]
    # probs = classifier.prob_classify_many(test_fsets)
    # for pdist in classifier.prob_classify_many(test_fsets):
    #     print('%.4f %.4f' % (pdist.prob('x'), pdist.prob('y')))

    classifier = SklearnClassifier(classifier_type())

    train_feats = []
    for sent, label, loc in train_sents:
        feats = bow_feats(sent)
        featureset = {f: loc for f in feats}
        train_feats.append((featureset, label))

    test_feats = []

    for sent, label, loc in devtest_sents:
        feats = bow_feats(sent)
        featureset = {f: loc for f in feats}
        test_feats.append(featureset)

    classifier.train(train_feats)
    predictions = classifier.classify_many(test_feats)
    test_gold = [label for sent, label, loc in devtest_sents]
    print (metrics.classification_report(test_gold, predictions))





# def classify_csv_dict(classifier_type=LinearSVC, num_sents=None):
#     '''
#     bow + loc + user
#     '''
#     data = csv_dict(head=num_sents)
#     classifier = SklearnClassifier(classifier_type())

#     ## classify dict or classify many?


#     # Create vectorizer for function to use
#     vectorizer = CountVectorizer()
#     vect = DictVectorizer()

#     # This is the part I want to fix
#     temp = zip(list(posts.message), list(posts.feature_1), list(posts.feature_2))
#     tokenized = map(lambda x: features(x), temp)
#     X = vect.fit_transform(tokenized)

#     # train_feats = []
#     # for sent, label, loc, user in train_sents:
#     #     feats = bow_feats(sent)
#     #     # featureset = {f: loc for f in feats}
#     #     featureset = {" ".join(sent) : loc}
#     #     train_feats.append((featureset, label))

#     # test_feats = []

#     # for sent, label, loc in devtest_sents:
#     #     feats = bow_feats(sent)
#     #     # featureset = {f: loc for f in feats}
#     #     featureset = {" ".join(sent) : loc}
#     #     test_feats.append(featureset)

#     # classifier.train(train_feats)
#     # predictions = classifier.classify_many(test_feats)
#     # test_gold = [label for sent, label, loc in devtest_sents]
#     # print (metrics.classification_report(test_gold, predictions))
   

def features(data):
        terms = cvect(data[0])
        print(terms)
        d = {'user': data[1], 'loc': data[2]}
        for t in terms:
            t = t.lower()
            # print(t)
            # print(d)
            ##default to 0, norm
            # print(d.get(t, 0))
            d[t] = d.get(t, 0) + 1
            # print(d[t])
        return d

from zlib import crc32

def bytes_to_float(b):
    return float(crc32(b) & 0xffffffff) / 2**32

def str_to_float(s, encoding="utf-8"):
    return bytes_to_float(s.encode(encoding))


def pos_fts_name_only(sent):
    '''
    Unigram feats: only the NNP (proper noun) POS tag
    '''
    # NNP_unigrams = {}
    tags = nltk.pos_tag(sent)
    
    # if not tags: #if tags is empty
    #     return

    unilist, poslist = zip(*tags)

    # if not ("NNP" in poslist):
    #     return
    
    NNP_unigrams=["#Empty"]
    for i, element in enumerate(poslist):
        if element == "NNP":
            uni = unilist[i] 
            NNP_unigrams.append(uni)   

    return " ".join(NNP_unigrams)


if __name__=="__main__":
    '''
    Loads and partitions data into train-test-devtest
    calls classifiers
    '''
    # from sklearn.metrics import auc
    

    start = timeit.default_timer()

    # ## dataset
    # ds = 'data/goodreads_reviews_spoiler.json.gz'
    # ## None entries = all entries
    # num_entries = None
    # data = load_sents(ds, num_entries, get_sents)
    
    # stop = timeit.default_timer()
    # print('Time for loading data: ', stop - start) 

    # print('Results for: ', ds, "\nNum entries: ", num_entries) 

    # train_sents, test_sents, devtest_sents = partition(data)

    # # classify_bow(train_sents, test_sents, devtest_sents)
    # # classify_loc(train_sents, test_sents, devtest_sents)
    # classify_bow_counts(train_sents, test_sents, devtest_sents)
    # # classify_pos(train_sents, test_sents, devtest_sents)


    # import pdb; pdb.set_trace()

   

    file = 'data/reviews_cut_new.csv'
    reader = csv.reader(file)
    
    data = pd.read_csv(file) #, nrows=size)
    size = len(data)
    print(size)

    n = 1000000
    if size > n:
        size = n

    data = pd.read_csv(file, nrows=size)
    # Create vectorizer for function to use
    # cvect = CountVectorizer(ngram_range=(1, 2)).build_tokenizer() 
    # dvect = DictVectorizer()

    bite = math.floor(size/10)

    # test = [data.sent[:bite], data.userID[:bite], data.sloc[:bite]]
    # test_labels = data.has_spoiler[:bite]

    # devtest = [data.sent[bite:2*bite], data.userID[bite:2*bite], data.sloc[bite:2*bite]]
    # devtest_labels = data.has_spoiler[bite:2*bite]

    # train = [data.sent[2*bite:], data.userID[2*bite:], data.sloc[2*bite:]]
    # train_labels = data.has_spoiler[2*bite:]

    # assert(shape(train)[1] == len(train_labels)), (shape(train)[1] ,len(train_labels))
    # assert(shape(test)[1] == len(test_labels)), (shape(test)[1],len(test_labels))
    # assert(shape(devtest)[1] == len(devtest_labels)), (shape(devtest)[1],len(devtest_labels))

    # print("Ratio: ", int(100*(shape(train)[1]/ size)), "-",  
    #     int(100*(shape(devtest)[1]/ size)), "-",  
    #     int(100*(shape(test)[1]/ size)))


    ### TRAIN

    import scipy as sp
    import numpy as np

    # print(data["sent"][:10])
    # data["sent"] = data["sent"].map(lambda s: pos_fts_name_only(s))
    print(data["sent"][:10])
    data["userID"] = data["userID"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    data["genre"] = data["genre"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)

    # data["sloc"] = data["sloc"].map(lambda s: int(s))
    # data["rating"] = data["rating"].map(lambda s: int(s))

    print(data["sloc"][:10])
    print(data["rating"][:30])
    print(data.has_spoiler.dtype)
    print(data.userID.dtype) #obj
    print(data.sloc.dtype) #int
    print(data.rating.dtype) #int


    y = data["has_spoiler"]

    # train = data[bite:]
    # test = data[:bite]
    # print(len(test),len(train))

    # Create vectorizer for function to use
    vectorizer = CountVectorizer(binary=True, ngram_range=(1, 1))

    feat_list= []
    # feat_list =['sloc']
    feat_list = [ ]
    # feat_list = ['userID']
    # feat_list = ['bookID']
    print(feat_list)
    

    X = sp.sparse.hstack((vectorizer.fit_transform(data.sent.values.astype('U')), data[feat_list].values),format='csr')
    X_columns=vectorizer.get_feature_names()+data[feat_list].columns.tolist()

    # count_vect = CountVectorizer()
    # X = count_vect.fit_transform(data.sent.values.astype('U'))

    # print(X_columns[:-10])
    # print(X[:-10])
    # print(data.sent.values.astype('U')[:10])


    # data1 = pd.read_csv('data/shuf_reviews.csv', sep=',', skiprows=range(1,size), nrows=math.floor(size/8))

    # # data['reviewID'] 
    # # data1['reviewID'] 
    # # print(data['reviewID'][:-10])
    # # print(data1['reviewID'][:10])
    # assert(data1['reviewID'][0] != data['reviewID'][0]), "OVERLAP"

    # data1["userID"] = data1["userID"].map(lambda s: float(crc32(s.encode("utf-8")) & 0xffffffff) / 2**32)
    # print(data1.userID.dtype) #obj
    # print(data1.sloc.dtype) #int
    # y_test = data1["has_spoiler"]

    print("FULL:" ,y.shape[0],X.shape[0])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    # print(X_train.shape[0])
    # print(y_train.shape[0])
    # print(X_test.shape[0])
    # print(y_test.shape[0])
    

    # X_test = sp.sparse.hstack((vectorizer.fit_transform(test.sent), test[['userID','sloc']].values),format='csr')
    # X_columns_test=vectorizer.get_feature_names()+test[['userID','sloc']].columns.tolist()
    # print(X_columns_test[:10])
    # print(len(X_columns_test) == len(X_columns))
    # print(len(X_columns_test),len(X_columns))

    # from sklearn_pandas import DataFrameMapper

    # mapper = DataFrameMapper([
    # (['userID', 'sloc'], None),
    # ('sent',CountVectorizer(binary=True, ngram_range=(1, 2)))
    # ])
    # X=mapper.fit_transform(data)

    # X_columns=mapper.features[0][0]+mapper.features[1][1].get_feature_names()

    # train_counts=map(lambda snt, ft1, ft2: features([snt,ft1,ft2]), 
    #             *train)


    # X_train = dvect.fit_transform(train_counts)
    # y_train = train_labels

    # ### TEST
    # devtest_counts=map(lambda snt, ft1, ft2: features([snt,ft1,ft2]), 
    #             *devtest)

    # # print(list(devtest_counts))

    # X_test = dvect.transform(devtest_counts)

    # # print(X_test)

    # y_test = devtest_labels

    # # X_counts is document term matrix where column = every feature in set
    # # and row = sent index, and value at row,col is boolean
    # # indicating if word_col appears in sent_row

    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    # print(X[:10])
    
    # print(X_test[:10])
    
    # print(X.shape[0])
    # print(X_test.shape[0])

    print("predict")
    y_test_predicted = classifier.predict(X_test)

    print("Multinomial results")
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


    stop2 = timeit.default_timer()
    print('Time total: ', stop2 - start) 

    print("\n####################\n\n")



    