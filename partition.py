import re
import gzip
import json
import math
import random
import spacy
import nltk
import string
from nltk.corpus import stopwords
import timeit
import csv
import pandas as pd



# from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")

# ## init spaCy tokenizer
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
# nlp = English()
# # Create a Tokenizer with the default settings for English
# # including punctuation rules and exceptions
# tokenizer = nlp.tokenizer

'''
This module contains methods to load tokenized sentences from dataset
and partition into test/train/dev sets
'''

stop_words = list(stopwords.words('english'))

# @DeprecationWarning
def book_load_sents(file_name, head = None):
    '''
    File input is JSON file where keys are bookIDs and
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

# @DeprecationWarning
def book_partition(sents_dict, seed=10):
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


def tokenize(s):
    t = re.split(r"(\W)", s)
    tokens = [i for i in t 
            if i not in ["\t","\n","\r","\f","\v"," ",""]
            and (i not in string.punctuation)
            and (i not in stop_words)]
    return tokens

def spacy_tok_to_str(tok_list):
    res = []
    for spacy_tok in tok_list:
        res.append(spacy_tok.text)
    return res


def get_sents(json_entry):
    '''
    Gathers only sentences + spoiler flag from json entry
    '''
    tuples = []
    sents = json_entry['review_sentences']

    for s in sents:
        if len(s[1]) < 3:
            continue
        tokens = tokenize(s[1])
        # print(tokens)
        # tokens = [token.text for token in nlp(str(s))]
        spoiler_flag = s[0]
        tuples.append((tokens,spoiler_flag))

    # print(tuples)
    return tuples


## TESTING METHOD -- too slow, sub with POS = NNP
def get_sents_no_names(json_entry):
    '''
    Gathers only sentences + spoiler flag from json entry
    '''
    tuples = []
    sents = json_entry['review_sentences']

    for s in sents:
        tokens = tokenize(s[1])
        doc = nlp(s[1])

        t = []
        for tok in tokens:
            if not tok in spacy_tok_to_str(doc.ents):
                t.append(tok)

        # tokens = [token.text for token in nlp(str(s))]
        spoiler_flag = s[0]
        # yield?
        tuples.append((t,spoiler_flag))

    # print(tuples)
    return tuples


def get_sents_loc(json_entry):
    '''
    load sents with relative location in review text

    sents need to be in format:
    (tokens, spoiler_label, location in review)
    '''
    tuples = []
    sents = json_entry['review_sentences']

    for loc,s in enumerate(sents):
        if len(s[1]) < 3:
            continue
        tokens = tokenize(s[1])
        # print(tokens)
        # tokens = [token.text for token in nlp(str(s))]
        spoiler_flag = s[0]
        tuples.append((tokens,spoiler_flag, loc))

    # print(tuples)
    return tuples


def load_sents(file_name, head = None, sents_method=get_sents):
    '''
    Code from Wan github.
    Returns list of tuples where for each sentence in reviews: (tokens, spoiler_flag)
    '''
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.extend(sents_method(d))
            count += 1
            
            # break if reaches the nth line
            if (head is not None) and (count > head):
                break

    return data


def csv_dict(csv_file="data/shuf_tok_reviews.csv", head=None):
    df = pd.read_csv(csv_file)
    start = timeit.default_timer()

    users = df.userID[:head]
    genres = df.genre[:head]
    bookIDs = df.bookID[:head]
    slocs = df.loc[:head]
    labels = df.has_spoiler[:head]
    sents_str = df.sent[:head]
    stop = timeit.default_timer()
    print('Time for fetching csv cols: ', stop - start)

    ## needs lower()
    tok_sents = [eval(s.lower()) for s in sents_str]

    stop2 = timeit.default_timer()
    print('Time for fetching toks: ', stop2 - start)

    return {'userID': users, 
            'genre': genres,
            'bookID': bookIDs,
            'tokens': tok_sents,
            'sloc': slocs,
            'label':labels}

    


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
    print("Ratio: ", int(100*(len(test_sents)/ size)), "-",  
        int(100*(len(devtest_sents)/ size)), "-",  
        int(100*(len(train_sents)/ size)))

    return train_sents, test_sents, devtest_sents



if __name__=="__main__":
    start = timeit.default_timer()

    # ## dataset
    # ds = 'data/goodreads_reviews_spoiler.json.gz'
    # ## None entries = all entries
    # num_entries = 10000
    # data = load_sents(ds, num_entries, get_sents_loc)

    partition_csv()
    
    stop = timeit.default_timer()
    print('Time for loading data: ', stop - start)