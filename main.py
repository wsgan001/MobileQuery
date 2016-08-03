import logging
import time
from collections import OrderedDict
import theano
from theano import config
from theano import tensor as T
import numpy as np

import load_data
from utils import *
from utils import sgd as optimizer


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")

    if shuffle:
        np.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def init_params(options):
    params = OrderedDict()
    # similarity matrix
    params['Wsim'] = np.random.rand(options['dim_u'] + options['dim_l'],
                                    options['dim_q']).astype(config.floatX)
    params['emb_u'] = np.random.rand(options['n_users'],
                                     options['dim_u']).astype(config.floatX)
    loc_embs = options['loc_embs']

    # load pre-trained loc features
    if isinstance(loc_embs, np.ndarray):
        params['emb_l'] = loc_embs.astype(config.floatX)
    else:
        params['emb_l'] = np.random.rand(options['n_locs'],
                                         options['dim_l']).astype(config.floatX)
    params['emb_q'] = np.random.rand(options['n_queries'],
                                     options['dim_q']).astype(config.floatX)

    # rnn params
    params['W'] = np.random.rand(options['dim_q'],
                                 options['dim_q']).astype(config.floatX)
    params['b'] = np.random.rand(options['dim_q']).astype(config.floatX)
    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def _p(pp, name):
    return '%s_%s' % (pp, name)


def build_model(tparams, options):
    u = T.vector('u', dtype='int64')
    l = T.matrix('l', dtype='int64')
    q = T.matrix('q', dtype='int64')

    dim_q = options['dim_q']
    n_samples, n_times = q.shape[0], q.shape[1]
    x_q = tparams['emb_q'][q.flatten()].reshape([n_times,
                                                 n_samples,
                                                 dim_q])

    win_size = l.shape[1]
    emb_l = tparams['emb_l'][l.flatten()].reshape([win_size,
                                                   n_samples,
                                                   options['dim_l']])
    # mean
    emb_l = emb_l.sum(axis=0) / win_size

    def rnn_unit(x_, h_):
        wx = tparams['W']
        h_t = T.nnet.sigmoid(T.dot(x_, wx) + T.dot(h_, wx) + tparams['b'])
        s_t = T.nnet.sigmoid(T.dot(h_, wx) + tparams['b'])
        return [h_t, s_t]

    [emb_q, _], _ = theano.scan(fn=rnn_unit, sequences=x_q,
                                outputs_info=[T.alloc(floatX(0.),
                                                      n_samples,
                                                      dim_q),
                                              None],
                                n_steps=n_times)
    emb_q = emb_q.sum(axis=0) / n_times

    emb = T.concatenate([tparams['emb_u'][u], emb_l], axis=1)
    scores = (emb_q * T.dot(emb, tparams['Wsim'])).sum(axis=1)
    cost = T.nnet.relu(1. - scores).sum()

    f_score = theano.function([u, l, q], scores, name='f_score')

    return (u, l, q, f_score, cost)


def calc_precision_recall(actuals, queries, scores):
    ranks = sorted(range(len(queries)), key=lambda x: scores[x])
    # compute precision/recall 1 to 10
    precisions = [p_at_k(actuals, ranks, k+1) for k in xrange(10)]
    recalls = [r_at_k(actuals, ranks, k+1) for k in xrange(10)]
    return (precisions, recalls)


def pred_error(f_pred, data, iterator, verbose=False):
    valid_err = 0
    valid_p = np.zeros(10).astype(config.floatX)
    valid_r = np.zeros(10).astype(config.floatX)
    for _, valid_index in iterator:
        u_set = [data[0][t] for t in valid_index]
        l_set = [data[1][t] for t in valid_index]
        queries = [data[2][t] for t in valid_index]
        actuals = [data[3][t] for t in valid_index]

        for (u, l, a, qs) in zip(u_set, l_set, actuals, queries):
            num_q = len(qs)
            scores = f_pred(np.tile(u, num_q),
                            np.tile(l, (num_q, 1)), qs)
            p, r = calc_precision_recall(a, qs, scores)
            valid_p += p
            valid_r += r

    valid_p /= len(data[0])
    valid_r /= len(data[0])
    valid_err = 1. - valid_p[0]

    print 'precison@k: %s' % (','.join(str(p) for p in valid_p))
    print 'recall@k: %s' % (','.join(str(r) for r in valid_r))
    logging.info('precison@k: %s' % (','.join(str(p) for p in valid_p)))
    logging.info('recall@k: %s' % (','.join(str(r) for r in valid_r)))

    return valid_err


def train_model(
    max_epochs=5,  # The maximum number of epoch to run
    decay_c=0.,  # Weight decay for weights
    lrate=1e-4,  # Learning rate for sgd (not used for adadelta and rmsprop)
    batch_size=16,  # The batch size during training
    valid_batch_size=64,  # The batch size used for test set
):
    model_options = locals().copy()
    dataparams, train, test = load_data.load_data()
    model_options.update(dataparams)

    print('Building model...')
    params = init_params(model_options)
    tparams = init_tparams(params)
    (u, l, q, f_score, cost) = build_model(tparams, model_options)
    args = [u, l, q]

    def _l2_regularizer(decay_c):
        decay_c = theano.shared(np.asarray(decay_c, dtype=config.floatX),
                                name='decay_c')
        l2r = 0.
        for kk, vv in tparams.items():
            l2r += (tparams[kk] ** 2).sum()
        return decay_c * l2r

    cost += _l2_regularizer(decay_c)

    grads = T.grad(cost, wrt=list(tparams.values()))
    # f_grad = theano.function([x], grads, name='f_grad')
    lr = T.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads, args, cost)

    kf = get_minibatches_idx(len(train[0]), batch_size,
                             shuffle=True)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size,
                                  shuffle=True)

    print('Training...')
    uidx = 0
    start = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0
            logging.info('Time: %s' % (time.time() - start))

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size,
                                     shuffle=True)

            for _,  train_index in kf:
                uidx += 1
                u = [train[0][t] for t in train_index]
                l = [train[1][t] for t in train_index]
                q = [train[2][t] for t in train_index]
                n_samples += len(u)

                cost = f_grad_shared(u, l, q)
                f_update(lrate)

                print('Epoch ', eidx, 'Update ', uidx,
                      'Cost ', cost)
                logging.info('------ Epoch: %d, Update(cls): %d -------'
                             % (eidx, uidx))
                pred_error(f_score, test, kf_test)

                logging.info('------------------------------------')

    except KeyboardInterrupt:
        print("Training interupted")


if __name__ == '__main__':
    logging.basicConfig(filename='test.log', level=logging.DEBUG)
    train_model(max_epochs=12, decay_c=1e-4)

