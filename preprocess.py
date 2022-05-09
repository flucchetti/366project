import json
import gzip
import csv

'''
This module contains methods to pre-process the dataset
and save the new cleaned balanced dataset to a csv file in /data
'''
## GLOBAL VARS

## WAN datasets: book, genre, review
review_ds = 'data/goodreads_reviews_spoiler.json.gz'
book_info_ds = 'data/goodreads_books.json.gz'
genre_ds = 'data/goodreads_book_genres_initial.json'

## number of spoiler sentences in review_ds
TOT_NUM_SPOIL = 569724


## INIT DICTS

book_to_genre = {}
book_to_author = {}
book_to_date_pub = {}


## MAKE DICTS

for line in open(genre_ds,"r"):
        jent = json.loads(line)
        bookID = jent["book_id"]

        ## get most popular genre label out of list of possible genres
        top_genre = jent["genres"]
        if top_genre == {}:
            continue

        ## genres have multiple names, pick first as ID
        ## eg. mystery-crime
        genre = sorted(top_genre, key=top_genre.get, reverse=True)[0].split(",")[0]

        # ##limit to these most popular genres:
        # if genre not in ["fantasy", "mystery", "romance", "children", "young-adult"]:
        #     continue

        book_to_genre[bookID] = genre



with gzip.open(book_info_ds) as f:
        for jentry in f:
            book_info = json.loads(jentry)
            bookID = book_info['book_id']
            ## get first auth if exists
            if len(book_info['authors']) == 0:
                continue

            auth = book_info['authors'][0]['author_id']
            date_pub = book_info['publication_year']

            book_to_author[bookID] = auth
            book_to_date_pub[bookID] = date_pub




def make_balanced_csv(dataset, csv_file, num_revs=None):
    '''
    Create csv
    '''
    entry_count = 0
    num_spoil = 0
    num_no_spoil = 0

    csvf = open(csv_file, 'w', encoding='UTF8')
    header = ['reviewID', 'userID', 'bookID', 'rating', 
            'date_published','authorID', 'genre',
            'sent', 's_loc', 's_spoiler']
    
    writer = csv.writer(csvf)
    writer.writerow(header)

    with gzip.open(dataset) as f:
        for jentry in f:
            jrev = json.loads(jentry)

            ## Skip entries with no spoilers
            if jrev['has_spoiler'] == 'False':
                continue

            entry_count += 1
            reviewID = jrev['review_id']
            userID = jrev['user_id']
            bookID = jrev['book_id']
            stars = jrev['rating']


            if (bookID not in book_to_author.keys()
                or bookID not in book_to_date_pub.keys()
                or bookID not in book_to_genre.keys()):
                ## no entry for bookID, skip
                continue

            # genre = '##'
            # date_pub = '##'
            # auth = '##'
            genre = book_to_genre[bookID]
            date_pub = book_to_date_pub[bookID]
            auth = book_to_author[bookID]


            rev_info = [reviewID, userID, bookID, stars, date_pub, auth, genre]

            ## per sentence data
            for i, sent in enumerate(jrev['review_sentences']):
                sent_info=[]

                if sent[0] == 1:
                    num_spoil += 1
                elif sent[0] == 0:
                    num_no_spoil += 1

                ## reached 50-50 split: stop gathering no-spoil sents
                if num_no_spoil > TOT_NUM_SPOIL and sent[0] == 0:
                    continue

                sent_info = [sent[1], i, sent[0]]
                writer.writerow(rev_info + sent_info)

            
            # FOR DEV ONLY - break if reaches the nth entry
            if (num_revs is not None) and (entry_count > num_revs):
               break

    print("Num entries: " , entry_count)



def ratio_spoilers(csv_file):
    '''
    Counts ratio spoiler=1 to spoiler=0 in csv
    '''
    sp_count = 0
    nosp_count = 0
    with open(csv_file, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[-1] == "0":
                nosp_count += 1
            elif row[-1] == "1":
                sp_count += 1

    print("SP", "NO SP")
    print(sp_count , nosp_count)
    print('SUM', sp_count + nosp_count)
    print(sp_count / (sp_count + nosp_count))




if __name__=="__main__":
    csv_file = 'data/balanced_revs.csv'
    # make_balanced_csv(review_ds, csv_file)
    # ratio_spoilers(csv_file)






# def load_shuffled(file_name, head = None):
#     '''
#     Code from Wan github.
#     Returns list of tuples where for each sentence in reviews: (tokens, spoiler_flag)
#     Shuffles json entries
#     '''
#     count = 0
#     data = []
#     with gzip.open(file_name) as fin:
#         for l in fin:
#             d = json.loads(l)
#             data.append(d)
#             count += 1
#             # break if reaches the nth entry - only for dev
#             if (head is not None) and (count > head):
#                break
#     #shuffle data and return up to head
#     seed=10
#     random.Random(seed).shuffle(data)  
#     return data


# def save_shuffled():
#     '''
#     Saves shuffled json entry dataset to file
#     '''
#     start = timeit.default_timer()

#     ## dataset
#     ds = 'data/goodreads_reviews_spoiler.json.gz'
#     num_entries = 10
#     data = load_shuffled(ds)
    
#     ##write data 
#     newf = open("data/shuffled_data.json","w")
#     for d in data:
#         newf.write(json.dumps(d) + "\n")
#     newf.close

#     stop = timeit.default_timer()

#     print('Time: ', stop - start)  



# def load_book_order(file_name, head = None):
#     '''
#     Creates dict from dataset where key = book id
#     and value = list of reviews for book
#     '''

#     book_dict = {}
#     count = 0
#     with gzip.open(file_name) as fin:
#         for l in fin:
#             entry = json.loads(l)

#             ## break when reach desired size - only for dev
#             count += 1
#             if head and count > head:
#                 break

#             key = entry['book_id']

#             if not key in book_dict.keys():
#                 ## create new list {entry} at key = book_id
#                 book_dict[key] = [entry]
#             else:
#                 ## append entry to list at key = book_id
#                 book_dict[key].append(entry)
            
#     return book_dict


# def save_book_order():
#     '''
#     Saves dataset divided by book order to file
#     '''
#     start = timeit.default_timer()

#     ## dataset
#     ds = 'data/goodreads_reviews_spoiler.json.gz'
#     num_entries = None
#     book_dict = load_book_order(ds, num_entries)
    
#     ##write data to json file
#     newf = open("data/book_order_data.json","w")
#     newf.write(json.dumps(book_dict))
#     newf.close()

#     # read_f = open("data/book_order_data.json","r")
#     # ## check if loads in json
#     # jobj = json.load(read_f)

#     stop = timeit.default_timer()

#     print('Time: ', stop - start)  


# def make_genre_dict(genre_ds="data/goodreads_book_genres_initial.json"):
#     '''
#     Takes WAN genre dataset (JSON) and makes a genre-to-bookIDs dict
#     '''
#     genre_to_bookID = {}

#     for line in open(genre_ds,"r"):
#         # print(line)
#         jent = json.loads(line)
#         bookID = jent["book_id"]

#         ## get primary genre out of list
#         ## most popular genre classification has highest value
#         g = jent["genres"]

#         if g == {}:
#             continue

#         genre = sorted(g, key=g.get, reverse=True)[0].split(",")[0]

#         ##limit to these most popular genres:
#         if genre not in ["fantasy", "mystery", "romance", "children", "young-adult"]:
#             continue

#         if not genre in genre_to_bookID.keys():
#             ## new list
#             genre_to_bookID[genre] = [bookID]
#         else:
#             genre_to_bookID[genre].append(bookID)

#     # ## order genres by size of booklist
#     # sorted(genre_to_bookID, key=lambda k: len(genre_to_bookID[k]), reverse=True)

#     # # print size of booklist for each genre
#     # for k,v in genre_to_bookID.items():
#     #     print(k, ": ", len(v))

#     return genre_to_bookID


# def invert_mapping(d):
#     inv_d = {}

#     for k, v in d.items():
#         for i in v:
#             inv_d[i] = k

#     return inv_d


# def load_genre(bookID_ds, genre_ds, head = None):
#     '''
#     Creates dict from bookID JSON dataset where key = genre id
#     and value = list of reviews for books with that genre
#     '''
#     count = 0
#     fp = open(bookID_ds,"r")
#     book_dict = json.load(fp)
#     genre_dict = {}

#     # init bookID-to-genre dict from dataset
#     genre_to_booklist = make_genre_dict(genre_ds)
#     book_to_genre = invert_mapping(genre_to_booklist)

#     for bookID, jentries in book_dict.items():

#         # break if reaches the nth entry - for DEV only
#         count += len(jentries)
#         if (head is not None) and (count > head):
#             break

#         if not bookID in book_to_genre.keys():
#             ## discard if not in target genres
#             continue

#         key = book_to_genre[bookID]

#         for jent in jentries:
#             # print(jent)
#             if key not in genre_dict.keys():
#                 genre_dict[key] = [jent]
#             else:
#                 genre_dict[key].append(jent)

#     return genre_dict


# def save_genre(num_entries=None):
#     '''
#     Saves dataset divided by genreID to different genre file
#     '''
#     start = timeit.default_timer()

#     ## dataset
#     book_ds = "data/book_order_data.json"
#     genre_ds= "data/goodreads_book_genres_initial.json"
#     genre_dict = load_genre(book_ds, genre_ds, num_entries)
    
#     ##write data to genre.json files
#     for genre in genre_dict.keys():
#         newf = open("data/"+genre+"_data.json","w")
#         for review in genre_dict[genre]:
#             newf.write(json.dumps(review))
#             newf.write("\n")
#         newf.close()

#     stop = timeit.default_timer()

#     print('Time: ', stop - start)  



# def make_csv(file_name, bookID_to_genre, head=None, tokenize=False, csv_file='data/reviews.csv'):
#     '''
#     Note: genre is only 5 cat
#     '''
#     count = 0
#     fp = open(csv_file, 'w', encoding='UTF8')
#     header = ['reviewID', 'userID', 'bookID', 'genre', 'rating', 'sent', 'sloc', 'has_spoiler']
#     ## date, stars?
#     writer = csv.writer(fp)
#     writer.writerow(header)

#     with gzip.open(file_name) as fin:
#         for l in fin:
#             d = json.loads(l)

#             reviewID = d['review_id']
#             userID = d['user_id']
#             bookID = d['book_id']
#             stars = d['rating']


#             ## only 5 most pop genres
#             if bookID not in bookID_to_genre.keys():
#                 continue
#             genre = bookID_to_genre[bookID]

#             rev_info = [reviewID, userID, bookID, genre, stars]
#             for i, sent in enumerate(d['review_sentences']):
#                 sent_info=[]
#                 if not tokenize:
#                     sent_info = [sent[1], i, sent[0]]
#                 else:
#                     s = str(sent[1])
#                     toks = pos_feats_name_only(s)
#                     if not toks:
#                         continue
#                     sent_info = [toks, i, sent[0]]
#                 ##write to csv 
#                 writer.writerow(rev_info + sent_info)

#             count += 1
#             # break if reaches the nth entry - only for dev
#             if (head is not None) and (count > head):
#                break

#     print("Num entries: " , count)