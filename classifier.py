import json
import gzip
import pickle

pickle_file = "data/reviews.pickle"


def load_data(file_name, head = None):
    '''
    Code from Wan github.
    '''
    count = 0
    data = []
    with gzip.open(file_name) as fin:
        for l in fin:
            d = json.loads(l)
            data.append(d)
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

if __name__=="__main__":
    ## dataset
    ds = 'data/goodreads_reviews_spoiler.json.gz'
    num_entries = 10000
    data = load_data(ds, num_entries)
    # print(data)
    print(len(data))

    # make_pickle(data)
    # data2 = load_pickle()
    # print(len(data2))
    # print(data2)
 