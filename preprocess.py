import json
import gzip
import random
import timeit

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
    random.shuffle(data)  
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
    newf = open("data/ordered_data.json","w")
    newf.write(json.dumps(book_dict))
    newf.close()

    # read_f = open("data/ordered_data.json","r")
    # ## check if loads in json
    # jobj = json.load(read_f)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  



if __name__=="__main__":

   ...

