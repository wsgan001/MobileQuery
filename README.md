Mobile Query Recommendation
===
$: `python main.py`

**Note:** Modification to load_data.py is needed if you want to run this program.

Requirements
---
python 2.7, theano(above v0.8)
Data format
---
Define num of users, locations, queries as `I, J, K`   
Define train, test sets size as `M, N`  
Define location context window size as `C`  
Define word sequence max length as `S`  
Define sampled_queries size as `T`   

train_set: 3-element tuple (users, locations, queries)  
- users: vector, users indices, shape: `[M]`
- locations: matrix, context window corespondding to each location, shape: `[M, C]`
- queries: matrix, word sequence corespondding to each query, shape: `[M, S]`

test_set: 4-element tuple (users, locations, sample_queries, actual_queries)  
- users: vector, users indices, shape: `[N]`
- locations: matrix, context window corespondding to each location, shape: `N, C]`
- queries: tensor3, each matrix represents word sequence corespondding to every sample query, shape: `[N, T, S]`
- acutal_queries: python list of list, each list represents queries corespondding to (user, location) pair

Parameters
---
- num_u, num_l, num_q: num of users, locations, queries
- dim_u, dim_l, dim_q: dimension of features for user, location, query
- loc_feats: location features
- max_epochs=5: The maximum number of epoch to run
- decay_c=0: Weight decay for weights
- lrate=1e-4: Learning rate for sgd (not used for adadelta and rmsprop)
- batch_size=16: The batch size during training
- valid_batch_size=64: The batch size used for test set

