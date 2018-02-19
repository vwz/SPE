#encoding=utf-8
'''
Created on 2017年1月13日
@author: Liu Zemin
Functions and Application : 

'''

import numpy
import theano
from theano import tensor


def autoencoderCalculateModel(tparams):
    """
        aotuencoder compute model
    """
    
    x=tensor.vector('x',dtype=theano.config.floatX)  # @UndefinedVariable
    
    y=tensor.nnet.sigmoid(tensor.dot(tparams['w'],x)+tparams['b1'])
    
    # 返回结果
    return x, y
    