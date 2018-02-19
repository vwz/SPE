#encoding=utf-8
from __future__ import print_function
import os

import six.moves.cPickle as pickle  # @UnresolvedImport
# import six.moves as moves

from collections import OrderedDict
import sys
import time

import numpy
import theano
from theano import config 
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


# theano.config.floatX = 'float32'
# Set the random number generators' seeds for consistency
# SEED = 123
# numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable


def _p(pp, name):
    return '%s_%s' % (pp, name)

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    nsteps = state_below.shape[0] 
    if state_below.ndim == 3:
        n_samples = state_below.shape[1] 
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_): 
        preact = tensor.dot(h_, tparams['lstm_U']) 
        preact += x_ 

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dimension_lstm'])) # input gate 
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dimension_lstm'])) # forget gate 
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dimension_lstm'])) # output gate 
        c = tensor.tanh(_slice(preact, 3, options['dimension_lstm'])) # init cell 

        c = f * c_ + i * c 
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c) 
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c
    state_below = (tensor.dot(state_below, tparams['lstm_W']) + tparams['lstm_b'])

    dim_proj = options['dimension_lstm']
    rval, updates = theano.scan(_step, 
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.), 
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.), 
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps) # maxlen
    return rval[0] 

def build_model(tparams, options, x, mask):

    proj = lstm_layer(tparams, x, options,
                                            prefix='lstm',
                                            mask=mask)
    return  proj


def get_lstm(
    model_options, # model config parameters
    tparams, # theano shared variables
    x, # a sequence
    x_mask, # mask matrix
):

    proj = build_model(tparams, model_options, x, x_mask)
    return proj
