#encoding=utf-8
'''
Functions and Application : 
autoencoder model
'''

import numpy
import theano
from theano import tensor


def autoencoderModel(options, tparams):
    """
        autoencoder model
    """
    
    x=tensor.matrix('x',dtype=theano.config.floatX)  # @UndefinedVariable
    
    def _objectiveFunc(index, loss):
        y=tensor.nnet.sigmoid(tensor.dot(tparams['w'],x[index])+tparams['b1'])
#         _x=tensor.nnet.sigmoid(tensor.dot(tensor.transpose(tparams['w']),y)+tparams['b2'])
        _x=tensor.nnet.sigmoid(tensor.dot(tparams['w2'],y)+tparams['b2'])
        p=((x[index]-_x)**2).sum()
        return loss+p
        
    rval, update = theano.scan(
                               _objectiveFunc,
                               sequences=tensor.arange(x.shape[0]), 
                               outputs_info=tensor.constant(0., dtype=theano.config.floatX), # @UndefinedVariable 
                               )
      
    cost=rval[-1]
#     cost=0
   
    cost+=options['decay_w']*tparams['w'].norm(2)
    cost+=options['decay_w']*tparams['w2'].norm(2)
    cost+=options['decay_b1']*tparams['b1'].norm(2)
    cost+=options['decay_b2']*tparams['b2'].norm(2)

    return x, cost
    
    