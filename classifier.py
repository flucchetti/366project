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
import timeit

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



def tokenize(s):
    t = re.split(r"(\W)", s)
    tokens = list(i for i in t if i not in ["\t","\n","\r","\f","\v"," ",""])
    return tokens


def get_sents(json_entry):
    '''
    Gathers only sentences + spoiler flag from json entry
    '''
    tuples = []
    sents = json_entry['review_sentences']

    for s in sents:
        tokens = tokenize(s[1])
        # tokens = [token.text for token in nlp(str(s))]
        spoiler_flag = s[0]
        # yield?
        tuples.append((tokens,spoiler_flag))

    # print(tuples)
    return tuples
    


def load_data(file_name, head = None):
    '''
    Code from Wan github.
    Returns list of tuples where for each sentence in reviews: (tokens, spoiler_flag)
    '''
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.extend(get_sents(d))
            count += 1
            
            # break if reaches the nth line
            if (head is not None) and (count > head):
                break

    return data

def ordered_load_sents(file_name, head = None):
    '''
    File input is json file where keys are bookIDs and
    each value is all reviews (json entry) for that bookIS.
    Returns a dict where key = bookID
    and value = list of (tokens, spoiler_flag) for all sentences from all reviews 
    of bookID
    '''
    count = 0
    fp = open(file_name,"r")
    book_dict = json.load(fp)
    new_dict = {}

    for key, values in book_dict.items():

        count += len(values)
        # break if reaches the nth line
        if (head is not None) and (count > head):
            break

        tuple_list = []
        for jentry in values:
            s = get_sents(jentry)
            # print(s)
            tuple_list.extend(s)
        
        ## update values
        new_dict[key] = tuple_list

    return new_dict


def get_sents_from_key(book_dict, key_list, seed=10):
    ''' helper '''
    sents = []
    for key in key_list:
        # random.Random(seed).shuffle(book_dict[key])
        sents.extend(book_dict[key])

    return sents


def ordered_partition(sents_dict, seed=10):
    '''
    Takes a dict where key=bookID and values= list of (sents, tuples) for bookID
    partitions sents 10-10-80 into dev-test-train trying to keep
    same bookIDs in same bucket so same book will not overlap between train/test
    '''

    key_list = list(sents_dict.keys())
    random.Random(seed).shuffle(key_list)
    size = len(key_list)
    bite = math.floor(size/10)
    test_ky = key_list[:bite]
    devtest_ky = key_list[bite:2*bite]
    train_ky = key_list[2*bite:]



    test_sents = get_sents_from_key(sents_dict, test_ky)
    train_sents = get_sents_from_key(sents_dict, train_ky)
    devtest_sents = get_sents_from_key(sents_dict, devtest_ky)

    
    print("Number of sentences:")
    total = 0
    for v in sents_dict.values():
        total += len(v)
    print(total)
    print("- Test:", len(test_sents))
    print("- Devtest:", len(devtest_sents))
    print("- Training:", len(train_sents))
    print("Ratio: ", int(100*(len(test_sents)/ total)), "-",  
        int(100*(len(devtest_sents)/ total)), "-",  
        int(100*(len(train_sents)/ total)))

    return train_sents, test_sents, devtest_sents


def partition(sents, seed=10):
    random.Random(seed).shuffle(sents)

    size = len(sents)
    bite = math.floor(size/10)

    test_sents = sents[:bite]
    devtest_sents = sents[bite:2*bite]
    train_sents = sents[2*bite:]

    print("Number of sentences:")
    print("total size:", size)
    print("- Test:", len(test_sents))
    print("- Devtest:", len(devtest_sents))
    print("- Training:", len(train_sents))

    return train_sents, test_sents, devtest_sents


def bow_feats(sent):
    '''
    Returns bag of words feature (simple no counts just flag)
    '''
    bow = {}
    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow


def pos_feats(sents):
    ...

def classify_bow(train_sents, test_sents, devtest_sents):
    '''
    Builds BOW calssifier (SVC or Naive) and prints accuracy
    or most informative feats
    '''
    # get feats
    test_feats = [(bow_feats(sent), label) for sent, label in test_sents]
    devtest_feats = [(bow_feats(sent), label) for sent, label in devtest_sents]
    train_feats = [(bow_feats(sent), label) for sent, label in train_sents]

    # classify_bow = nltk.classify.SklearnClassifier(LinearSVC())
    classify_bow = nltk.NaiveBayesClassifier.train(train_feats)

    classify_bow.train(train_feats)

    accuracy = nltk.classify.accuracy(classify_bow, devtest_feats)
    print("Accuracy score:", accuracy)
    classify_bow.show_most_informative_features(40)



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
    ds = 'data/ordered_data.json'
    num_entries = 150000
    data = ordered_load_sents(ds, num_entries)

    # newf = open("data/test.txt","w")
    # newf.write(str(data))
    # newf.close()
    
    stop = timeit.default_timer()
    print('Time for loading data: ', stop - start) 

    
    train_sents, test_sents, devtest_sents = ordered_partition(data)

    classify_bow(train_sents, test_sents, devtest_sents)

    classify_bowc(train_sents, test_sents, devtest_sents)

    