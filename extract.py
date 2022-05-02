import json
import gzip
import random
import timeit


def load_data(file_name, head = -1):
    '''
    Code from Wan github.
    Returns list of tuples where for each sentence in reviews: (tokens, spoiler_flag)
    '''
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.append(d)
            count += 1
            # break if reaches the nth line
            #if (head is not None) and (count > head):
            #    break
    #shuffle data and return up to head
    random.shuffle(data)  
    return data#[:head]


def order_by_book(file_name, head = None):
    '''
    Creates dict from dataset where key = book id
    and value = list of reviews for book
    '''

    book_dict = {}
    count = 0
    with gzip.open(file_name) as fin:
        for l in fin:
            entry = json.loads(l)

            ## break when reach desired size
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


if __name__=="__main__":

    # start = timeit.default_timer()

    # ## dataset
    # ds = 'data/goodreads_reviews_spoiler.json.gz'
    # num_entries = 10
    # data = load_data(ds)
    
    # ##write data 
    # newf = open("data/shuffled_data.json","w")
    # for d in data:
    #     newf.write(json.dumps(d) + "\n")
    # newf.close

    # stop = timeit.default_timer()

    # print('Time: ', stop - start)  

    start = timeit.default_timer()

    ## dataset
    ds = 'data/goodreads_reviews_spoiler.json.gz'
    num_entries = None
    book_dict = order_by_book(ds, num_entries)
    
    ##write data 
    newf = open("data/ordered_data.json","w")
    newf.write(json.dumps(book_dict))
    newf.close()

    # read_f = open("data/ordered_data.json","r")
    # ## check if loads in json
    # jobj = json.load(read_f)

    stop = timeit.default_timer()

    print('Time: ', stop - start)  
