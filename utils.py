import theano
import numpy as np
import theano.tensor as tensor
from theano import config


def floatX(data):
    return np.asarray(data, dtype=config.floatX)


def sgd(lr, tparams, grads, x, cost):
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function(x, cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adagrad(lr, tparams, grads, x, cost, epsilon=1e-6):
    """Adagrad updates
    Parameters
    ----------
    lr : float or symbolic scalar
        The learning rate controlling the size of update steps
    epsilon : float or symbolic scalar
        Small value added for numerical stability
    """

    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    accu_grads = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
                  for k, p in tparams.items()]
    accu_up = [(accu, accu + g ** 2) for accu, g in zip(accu_grads, grads)]

    f_grad_shared = theano.function(x, cost, updates=gsup + accu_up,
                                    name='adagrad_f_grad_shared')

    updir = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
             for k, p in tparams.items()]
    updir_new = [(ud, lr * g / tensor.sqrt(accu + epsilon))
                 for ud, g, accu in zip(updir, gshared, accu_grads)]
    param_up = [(p, p - udn[1]) for p, udn in zip(tparams.values(), updir_new)]

    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               name='adagrad_f_update')

    return f_grad_shared, f_update


def p_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(k)
    return result


def r_at_k(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    result = len(act_set & pred_set) / float(len(actual))
    return result
