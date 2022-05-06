import itertools
import spacy
import nltk
from partition import spacy_tok_to_str

# from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm", exclude=["parser"])
nlp.enable_pipe("senter")

'''
This module contains methods to extract features from sents
'''


def bow_feats_no_names(sent):
    '''
    Returns bag of words feature (simple no counts just flag)
    '''
    bow = {}
    doc = nlp(sent)

    for token in sent:
        if not token in spacy_tok_to_str(doc.ents):
            bow[token.lower()] = 1

    return bow


def bow_feats(sent):
    '''
    Returns bag of words feature (simple no counts just flag)
    '''
    bow = {}
    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow




def pos_feats(sent):
    '''
    Bigram feats: POS tag
    '''
    # NNP: proper noun (overfit?), VBP: verb present tense
    # JJ: adj
    a = ("NNP", "NNS", "NN")
    b = ("VBP", "VBZ", "VBD", "VBG", "VBN","JJ")
    target_pos = list(itertools.product(a,b))
    bigrams = {}
    tags = nltk.pos_tag(sent)

    unilist, poslist = zip(*tags)
    ##problem: hardcoded NNP
    if not ("NN" in poslist or "NNS" in poslist or "NNP" not in poslist):
        return {}

    cartesian1 = list(itertools.product(unilist, unilist))
    cartesian2 = list(itertools.product(poslist, poslist))

    for i, feats in enumerate(cartesian1):
        if cartesian2[i] in target_pos:
            ##problem: hardcoded NNP
            # feats = ("NNP", feats[1])
            bi = "-".join(feats)
            bigrams[bi] = 1

    # print(bigrams)
    ## PROBLEM: a lot of these will be empty (won't always get
    # target_pos). Need to normalize data for it to work?
    return bigrams