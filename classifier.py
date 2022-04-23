import json
import gzip
import pickle
import re
import random
import math
import nltk

pickle_file = "data/reviews.pickle"

# ## init spaCy tokenizer
# from spacy.tokenizer import Tokenizer
# from spacy.lang.en import English
# nlp = English()
# # Create a Tokenizer with the default settings for English
# # including punctuation rules and exceptions
# tokenizer = nlp.tokenizer

# from nltk.tokenize import word_tokenize


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


def make_pickle(data):
    pickle_out = open(pickle_file,"wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()

def load_pickle():
    pickle_in = open(pickle_file,"rb")
    data = pickle.load(pickle_in)
    return data


def partition(sents, seed=10):
    random.Random(seed).shuffle(sents)

    size = len(sents)
    bite = math.floor(size/10)

    test_sents = sents[:bite]
    devtest_sents = sents[bite:2*bite]
    train_sents = sents[2*bite:]

    # print("Number of sentences:")
    # print("total size:", size)
    # print("- Test:", len(test_sents))
    # print("- Devtest:", len(devtest_sents))
    # print("- Training:", len(train_sents))

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



if __name__=="__main__":

    import timeit

    start = timeit.default_timer()

    ## dataset
    ds = 'data/goodreads_reviews_spoiler.json.gz'
    num_entries = 100000
    data = load_data(ds, num_entries)
    # print(data)
    train_sents, test_sents, devtest_sents = partition(data)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  

    test_feats = [(bow_feats(sent), label) for sent, label in test_sents]
    devtest_feats = [(bow_feats(sent), label) for sent, label in devtest_sents]
    train_feats = [(bow_feats(sent), label) for sent, label in train_sents]

    # print(train_feats[:10])
    # print(test_feats[:10])

    classify_bow = nltk.NaiveBayesClassifier.train(train_feats)
    accuracy = nltk.classify.accuracy(classify_bow, test_feats)
    # print("Accuracy score:", accuracy)

    # BOW with counts
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()

    X_train_counts = count_vect.fit_transform([" ".join(sent) for sent, label in train_sents])
    y_train = [label for sent, label in train_sents]

    X_test_counts = count_vect.transform([" ".join(sent) for sent, label in test_sents])
    y_test = [label for sent, label in test_sents]

    from sklearn.naive_bayes import MultinomialNB

    whosaid2 = MultinomialNB()
    whosaid2.fit(X_train_counts, y_train)

    y_test_predicted = whosaid2.predict(X_test_counts)

    from sklearn import metrics
    print("Multinomial results")
    print(metrics.classification_report(y_test, y_test_predicted))

    print(metrics.accuracy_score(y_test, y_test_predicted))

    # make_pickle(data)
    # data2 = load_pickle()
    # print(len(data2))
    # print(data2)
 