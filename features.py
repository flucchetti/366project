import spacy
import re
import string
import nltk
from nltk.corpus import stopwords
stop_words = list(stopwords.words('english'))
from nltk.parse.stanford import StanfordDependencyParser

from nltk.tokenize import word_tokenize
# nlp = spacy.load("en_core_web_sm", exclude=["parser"])
# nlp.enable_pipe("senter")

# ## for dep parsing
# parser = nlp.add_pipe("parser")


'''
This module contains methods to extract features from sents
'''

def tokenizer(s):
    ## nltk tokenizer
    # return word_tokenize(s)

    t = re.split(r"(\W)", s)
    tokens = [i for i in t 
            if i not in ["\t","\n","\r","\f","\v"," ",""]
            and (i not in string.punctuation)
            # and (i not in stop_words)
            ]
    return tokens


## for classify_bow_NB
def bgram_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns bigrams and unigrams from sent 
    '''
    if need_tokenize:
        sent = tokenizer(str(sent))

    bigrams = list(zip(sent[:-1], sent[1:]))

    d = {'-'.join(b).lower() :1 for b in bigrams}
    
    ## add unigrams too
    for tok in sent:
        d[tok.lower()] = 1

    # print(bigrams[:10])
    return d



def bow_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns bow unigrams from sent
    '''
    bow = {}
    if need_tokenize:
        sent = tokenizer(str(sent))
        
    for tokens in sent:
        bow[tokens.lower()] = 1
    return bow



def NE_feats_NB(sent, need_tokenize=True):
    '''
    For NaiveBayes classifier
    Returns Named Entity (NE) unigrams from sent
    '''
    if need_tokenize:
        sent = tokenizer(str(sent))

    NNP_unigrams = {"#EMPTY#":1}
    tags = nltk.pos_tag(sent)
    
    if not tags: #if tags is empty, error
        return 

    unilist, poslist = zip(*tags)
    
    for i, element in enumerate(poslist):
        if element == "NNP":
            uni = unilist[i] 
            NNP_unigrams[uni.lower()] = 1   

    return NNP_unigrams


## for classify_many
def pos_bgram_feats(sent, need_tokenize=True):
    '''
    For classify_bow_counts and classify_many_feats
    Text feature: Parts of Speech bigrams with PLACEHOLDER substitution from sent

    eg. "The white cat walks"
    Returns: (the, white) (white, NN) (NN, walks)
    '''
    if not need_tokenize:
        sent = ' '.join(sent)

    placeholder_bgrams = ["#EMPTY#"]
    tags = nltk.pos_tag(word_tokenize(sent))

    if not tags: #if tags is empty, error
        return 

    ugram_to_pos = dict(tags)
    # print(ugram_to_pos)

    ugrams = [ugr for (ugr, pos) in tags]
    bigrams = list(zip(ugrams[:-1], ugrams[1:]))

    placeholder_bgrams = ['#EMPTY']

    nouns = ['NN', 'NNP']
    verbs = ["VBP", "VBZ", "VBD", "VBG", "VBN"]

    for (ugram1, ugram2) in bigrams:
        ## GUO keep only pairs: JJ-nouns, nouns-verbs 
        # but hide Nouns
        pos1 = ugram_to_pos[ugram1]
        pos2 = ugram_to_pos[ugram2]

        if pos1 in nouns and (pos2 in verbs or pos2 == 'JJ'):
            placeholder_bgrams.append('-'.join([pos1, ugram2.lower()]))
        elif pos2 in nouns and (pos1 in verbs or pos1 == 'JJ'):
            placeholder_bgrams.append('-'.join([ugram1.lower(), pos2]))

    return ' '.join(placeholder_bgrams)


# def dep_bgram_feats(sent, need_tokenize=True):
#     '''
#     dependency parsing
#     with PLACEHOLDER substitution
#     '''
#     doc = nlp("This is a sentence.")
#     processed = parser(doc)
#     NNP_bigrams = {"#EMPTY":1}
#     tags = nltk.pos_tag(sent)
    
#     if not tags: #if tags is empty, error
#         return 

#     ugram_to_pos = dict(tags)

#     bigrams = list(zip(sent[:-1], sent[1:]))

#     placeholder_bgrams = []

#     nouns = ['NN', 'NNP']
#     verbs = ["VBP", "VBZ", "VBD", "VBG", "VBN"]

#     for (ugram1, ugram2) in bigrams:
#         ## GUO keep only pairs: JJ-nouns, nouns-verbs 
#         # but hide Nouns
#         pos1 = ugram_to_pos(ugram1)
#         pos2 = ugram_to_pos(ugram2)

#         if ( pos1 in nouns and 
#             (pos2 in verbs or pos2 == 'JJ')):
#             placeholder_bgrams.append((pos1, ugram2))
#         elif( pos2 in nouns and 
#             (pos1 in verbs or pos1 == 'JJ')):
#             placeholder_bgrams.append((ugram1, pos2))
#         else:
#             placeholder_bgrams.append((ugram1,ugram2))

#     return placeholder_bgrams






# def bow_feats_no_names(sent):
#     '''
#     Returns bag of words feature (simple no counts just flag)
#     '''
#     bow = {}
#     doc = nlp(sent)

#     for token in sent:
#         if not token in spacy_tok_to_str(doc.ents):
#             bow[token.lower()] = 1

#     return bow


# def bow_feats(sent):
#     '''
#     Returns bag of words feature (simple no counts just flag)
#     WITH TOK
#     '''
#     bow = {}
#     for tokens in sent:
#         bow[tokens.lower()] = 1
#     return bow




# def pos_feats(sent):
#     '''
#     Bigram feats: POS tag
#     '''
#     # NNP: proper noun (overfit?), VBP: verb present tense
#     # JJ: adj
#     a = ("NNP", "NNS", "NN")
#     b = ("VBP", "VBZ", "VBD", "VBG", "VBN","JJ")
#     target_pos = list(itertools.product(a,b))
#     bigrams = {}
#     tags = nltk.pos_tag(sent)

#     unilist, poslist = zip(*tags)
#     ##problem: hardcoded NNP
#     if not ("NN" in poslist or "NNS" in poslist or "NNP" not in poslist):
#         return {}

#     cartesian1 = list(itertools.product(unilist, unilist))
#     cartesian2 = list(itertools.product(poslist, poslist))

#     for i, feats in enumerate(cartesian1):
#         if cartesian2[i] in target_pos:
#             ##problem: hardcoded NNP
#             # feats = ("NNP", feats[1])
#             bi = "-".join(feats)
#             bigrams[bi] = 1

#     # print(bigrams)
#     ## PROBLEM: a lot of these will be empty (won't always get
#     # target_pos). Need to normalize data for it to work?
#     return bigrams