import json
import gzip
import random
from termios import VLNEXT
import timeit
import csv 
from partition import tokenize as tokenizer
import nltk

from flask_sqlalchemy import get_debug_queries
import pandas as pd
'''
This module contains methods to pre-process the dataset
and save the new cleaned dataset to a file in /data
'''

@DeprecationWarning
def load_shuffled(file_name, head = None):
    '''
    Code from Wan github.
    Returns list of tuples where for each sentence in reviews: (tokens, spoiler_flag)
    Shuffles json entries
    '''
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.append(d)
            count += 1
            # break if reaches the nth entry - only for dev
            if (head is not None) and (count > head):
               break
    #shuffle data and return up to head
    seed=10
    random.Random(seed).shuffle(data)  
    return data

@DeprecationWarning
def save_shuffled():
    '''
    Saves shuffled json entry dataset to file
    '''
    start = timeit.default_timer()

    ## dataset
    ds = 'data/goodreads_reviews_spoiler.json.gz'
    num_entries = 10
    data = load_shuffled(ds)
    
    ##write data 
    newf = open("data/shuffled_data.json","w")
    for d in data:
        newf.write(json.dumps(d) + "\n")
    newf.close

    stop = timeit.default_timer()

    print('Time: ', stop - start)  



def load_book_order(file_name, head = None):
    '''
    Creates dict from dataset where key = book id
    and value = list of reviews for book
    '''

    book_dict = {}
    count = 0
    with gzip.open(file_name) as fin:
        for l in fin:
            entry = json.loads(l)

            ## break when reach desired size - only for dev
            count += 1
            if head and count > head:
                break

            key = entry['book_id']

            if not key in book_dict.keys():
                ## create new list {entry} at key = book_id
                book_dict[key] = [entry]
            else:
                ## append entry to list at key = book_id
                book_dict[key].append(entry)
            
    return book_dict


def save_book_order():
    '''
    Saves dataset divided by book order to file
    '''
    start = timeit.default_timer()

    ## dataset
    ds = 'data/goodreads_reviews_spoiler.json.gz'
    num_entries = None
    book_dict = load_book_order(ds, num_entries)
    
    ##write data to json file
    newf = open("data/book_order_data.json","w")
    newf.write(json.dumps(book_dict))
    newf.close()

    # read_f = open("data/book_order_data.json","r")
    # ## check if loads in json
    # jobj = json.load(read_f)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  


def make_genre_dict(genre_ds="data/goodreads_book_genres_initial.json"):
    '''
    Takes WAN genre dataset (JSON) and makes a genre-to-bookIDs dict
    '''
    genre_to_bookID = {}

    for line in open(genre_ds,"r"):
        # print(line)
        jent = json.loads(line)
        bookID = jent["book_id"]

        ## get primary genre out of list
        ## most popular genre classification has highest value
        g = jent["genres"]

        if g == {}:
            continue

        genre = sorted(g, key=g.get, reverse=True)[0].split(",")[0]

        ##limit to these most popular genres:
        if genre not in ["fantasy", "mystery", "romance", "children", "young-adult"]:
            continue

        if not genre in genre_to_bookID.keys():
            ## new list
            genre_to_bookID[genre] = [bookID]
        else:
            genre_to_bookID[genre].append(bookID)

    # ## order genres by size of booklist
    # sorted(genre_to_bookID, key=lambda k: len(genre_to_bookID[k]), reverse=True)

    # # print size of booklist for each genre
    # for k,v in genre_to_bookID.items():
    #     print(k, ": ", len(v))

    return genre_to_bookID


def invert_mapping(d):
    inv_d = {}

    for k, v in d.items():
        for i in v:
            inv_d[i] = k

    return inv_d


def load_genre(bookID_ds, genre_ds, head = None):
    '''
    Creates dict from bookID JSON dataset where key = genre id
    and value = list of reviews for books with that genre
    '''
    count = 0
    fp = open(bookID_ds,"r")
    book_dict = json.load(fp)
    genre_dict = {}

    # init bookID-to-genre dict from dataset
    genre_to_booklist = make_genre_dict(genre_ds)
    book_to_genre = invert_mapping(genre_to_booklist)

    for bookID, jentries in book_dict.items():

        # break if reaches the nth entry - for DEV only
        count += len(jentries)
        if (head is not None) and (count > head):
            break

        if not bookID in book_to_genre.keys():
            ## discard if not in target genres
            continue

        key = book_to_genre[bookID]

        for jent in jentries:
            # print(jent)
            if key not in genre_dict.keys():
                genre_dict[key] = [jent]
            else:
                genre_dict[key].append(jent)

    return genre_dict


def save_genre(num_entries=None):
    '''
    Saves dataset divided by genreID to different genre file
    '''
    start = timeit.default_timer()

    ## dataset
    book_ds = "data/book_order_data.json"
    genre_ds= "data/goodreads_book_genres_initial.json"
    genre_dict = load_genre(book_ds, genre_ds, num_entries)
    
    ##write data to genre.json files
    for genre in genre_dict.keys():
        newf = open("data/"+genre+"_data.json","w")
        for review in genre_dict[genre]:
            newf.write(json.dumps(review))
            newf.write("\n")
        newf.close()

    stop = timeit.default_timer()

    print('Time: ', stop - start)  

def pos_feats_name_only(sent):
    '''
    Unigram feats: only the NNP (proper noun) POS tag
    '''
    # NNP_unigrams = {}
    tags = nltk.pos_tag(tokenizer(sent))
    
    if not tags: #if tags is empty
        return

    unilist, poslist = zip(*tags)

    if not ("NNP" in poslist):
        return
    
    NNP_unigrams=[]
    for i, element in enumerate(poslist):
        if element == "NNP":
            uni = unilist[i] 
            NNP_unigrams.append(uni)   

    return "-".join(NNP_unigrams)


def make_csv(file_name, bookID_to_genre, head=None, tokenize=False, csv_file='data/reviews.csv'):
    '''
    Note: genre is only 5 cat
    '''
    count = 0
    fp = open(csv_file, 'w', encoding='UTF8')
    header = ['reviewID', 'userID', 'bookID', 'genre', 'rating', 'sent', 'sloc', 'has_spoiler']
    ## date, stars?
    writer = csv.writer(fp)
    writer.writerow(header)

    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)

            reviewID = d['review_id']
            userID = d['user_id']
            bookID = d['book_id']
            stars = d['rating']


            ## only 5 most pop genres
            if bookID not in bookID_to_genre.keys():
                continue
            genre = bookID_to_genre[bookID]

            rev_info = [reviewID, userID, bookID, genre, stars]
            for i, sent in enumerate(d['review_sentences']):
                sent_info=[]
                if not tokenize:
                    sent_info = [sent[1], i, sent[0]]
                else:
                    s = str(sent[1])
                    toks = pos_feats_name_only(s)
                    if not toks:
                        continue
                    sent_info = [toks, i, sent[0]]
                ##write to csv 
                writer.writerow(rev_info + sent_info)

            count += 1
            # break if reaches the nth entry - only for dev
            if (head is not None) and (count > head):
               break

    print("Num entries: " , count)



def make_csv_cut(file_name,  bookID_to_genre, head=None, csv_file='data/reviews_cut_new.csv'):
    '''
    even split
    '''
    count = 0
    count_sp = 0
    count_nosp = 0
    fp = open(csv_file, 'w', encoding='UTF8')
    header = ['reviewID', 'userID', 'bookID', 'genre','rating', 'sent', 'sloc', 'has_spoiler']
    ## date, stars?
    writer = csv.writer(fp)
    writer.writerow(header)

    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)

            reviewID = d['review_id']
            userID = d['user_id']
            bookID = d['book_id']
            stars = d['rating']

            ## only 5 most pop genres
            if bookID not in bookID_to_genre.keys():
                continue
            genre = bookID_to_genre[bookID]

            rev_info = [reviewID, userID, bookID, genre, stars]
            for i, sent in enumerate(d['review_sentences']):
                sent_info=[]
                if sent[0] == 1:
                    count_sp += 1
                else:
                    continue

                sent_info = [sent[1], i, sent[0]]
                writer.writerow(rev_info + sent_info)

            count += 1
            # break if reaches the nth entry - only for dev
            if (head is not None) and (count > head):
               break

    fin.close()

    with gzip.open(file_name) as f:
        for l in f:
            d = json.loads(l)
            
            reviewID = d['review_id']
            userID = d['user_id']
            bookID = d['book_id']
            stars = d['rating']

            ## only 5 most pop genres
            if bookID not in bookID_to_genre.keys():
                continue
            genre = bookID_to_genre[bookID]

            rev_info = [reviewID, userID, bookID, genre, stars]
            for i, sent in enumerate(d['review_sentences']):
                sent_info=[]
                if count_nosp > count_sp:
                    print("COUNT:", count_nosp)
                    return

                if sent[0] == 0:
                    count_nosp += 1
                else:
                    continue

                sent_info = [sent[1], i, sent[0]]
                # else:
                #     s = str(sent[1])
                #     toks = pos_feats_name_only(s)
                #     if not toks:
                #         continue
                #     sent_info = [toks, i, sent[0]]
                ##write to csv 
                writer.writerow(rev_info + sent_info)

            count += 1
            # break if reaches the nth entry - only for dev
            if (head is not None) and (count > head):
               break

    print("Num entries: " , count)


if __name__=="__main__":
    bookID_to_genre = invert_mapping(make_genre_dict())
    make_csv_cut("data/goodreads_reviews_spoiler.json.gz", bookID_to_genre, head=None)
    # df = pd.read_csv('data/tok_reviews.csv')
    ## use eval to get toks
    print("done create")

    file_name = 'data/reviews_cut_new.csv'    
    sp_count = 0
    nosp_count = 0
    # fp = csv.reader(file_name)
    # for line in fp:
    #     print(line[-1])
    #     if line[-1] == "0":
    #         nosp_count += 1
    #     else:
    #         sp_count += 1

    # print(sp_count , nosp_count)

    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            
            # do something here with `row`
            if row[-1] == "0":
                nosp_count += 1
            elif row[-1] == "1":
                sp_count += 1

    print(sp_count , nosp_count)
    print(sp_count + nosp_count)
    print(sp_count / (sp_count + nosp_count))
