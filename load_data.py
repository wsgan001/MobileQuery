'''
    This is a data helper...
'''
import numpy as np
from collections import OrderedDict


# sample data
def load_data():
    # sample locs
    locs = np.random.randint(0, 1000, size=1000)
    queries = np.random.randint(0, 1000, size=1000)
    users = np.random.randint(0, 500, size=1000)

    # params about data size
    data_params = {}
    num_l, num_u, num_q = max(locs) + 1, max(users) + 1, max(queries) + 1
    data_params.update({'n_locs': num_l, 'n_users': num_u, 'n_queries': num_q})
    # put pre-trained loc features
    data_params['loc_embs'] = None

    dim_l, dim_q = 100, 150
    dim_u = dim_q - dim_l
    data_params.update({'dim_l': dim_l, 'dim_u': dim_u, 'dim_q': dim_q})

    # split dataset
    n_size = len(queries)
    sidx = np.random.permutation(n_size)
    train_p = 0.8
    n_train = int(np.round(n_size * train_p))
    train, test = sidx[:n_train], sidx[n_train:]

    # locaton context
    win_size = 5
    loc_seqs = np.random.randint(0, num_l, size=(num_l, win_size))
    # query sentence words (equal length with padding)
    query_seqs = np.random.randint(0, num_q, size=(num_q, 20))

    train_set = [[users[i] for i in train],
                 [loc_seqs[locs[i]] for i in train],
                 [query_seqs[queries[i]] for i in train]]

    # map: (u, l) --> q
    test_map = OrderedDict()
    for i in test:
        kk = (users[i], locs[i])
        vv = test_map.get(kk, set())
        vv.add(queries[i])
        test_map.update({kk: vv})
    ul_pair = zip(*test_map.keys())

    # sample negative queries for each (u, l) pair
    sampled_query_seqs = [np.random.randint(0, num_q, size=(100, 20))
                          for _ in test]
    test_set = (list(ul_pair[0]), list(ul_pair[1]), sampled_query_seqs,
                [list(v) for v in test_map.values()])

    return (data_params, train_set, test_set)
