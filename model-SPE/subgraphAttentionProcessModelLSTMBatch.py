#encoding=utf-8
'''
Functions and Application : 
attention based metagraph process model
'''

import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse
import lstmModel
# theano.config.floatX = 'float32'

def metagraphAttentionProcessModel(options,tparams):
    """
        the MPE process model
    """
    metagraphEmbeddings=tensor.matrix('metagraphEmbeddings',dtype=theano.config.floatX) # @UndefinedVariable # shape=#(metagraph)*len(metaEmbeding)
    subPaths_matrix=tensor.matrix('subPaths_matrix',dtype='int64') # shape=maxlen*#(sub-paths)
    subPaths_mask=tensor.matrix('subPaths_mask',dtype=theano.config.floatX)  # @UndefinedVariable # shape=maxlen*#(sub-paths)
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable # #(words)*word_dimension
    
    metagraphBeta=tensor.exp(tensor.dot(tensor.nnet.sigmoid(tensor.dot(metagraphEmbeddings, tparams['Q_A'])+tparams['b_A']), tparams['eta_A']))
        
    def _processSubpathsBatch():
        x=subPaths_matrix 
        x_mask=subPaths_mask
        
        wordEmb=wordsEmbeddings[x] # shape=len(path)*1*len(wordsEmbeddings)
        tmp=wordEmb*metagraphBeta
        softmax0=tmp/tmp.sum(axis=-1, keepdims=True) # shape=len(path)*1*len(wordsEmbeddings) 
        subpaths=((softmax0*wordEmb)[:,:,:,None]*metagraphEmbeddings).sum(axis=2)
            
            
        h3Dmatrix=lstmModel.get_lstm(options, tparams, subpaths, x_mask)
            
        beta1=tensor.exp(tensor.dot(tensor.nnet.sigmoid(tensor.dot(h3Dmatrix, tparams['Q_B'])+tparams['b_B']), tparams['eta_B']))
        temp=x_mask*beta1
        softmax1=temp/temp.sum(axis=0, keepdims=True) 
        # shape=dimension_lstm*0
        pathsEmb=(softmax1[:,:,None]*h3Dmatrix).sum(axis=0)
            
        return pathsEmb
    
    rval2=_processSubpathsBatch()
    beta=tensor.dot(tensor.nnet.sigmoid(tensor.dot(rval2, tparams['Q_C'])+tparams['b_C']), tparams['eta_C'])
    softmax2=tensor.nnet.softmax(beta)[0] 
    embx=(softmax2[:,None]*rval2).sum(axis=0) # shape=dimension_lstm*0
            
    score=tensor.dot(embx,tparams['w'])
        
    return metagraphEmbeddings, subPaths_matrix, subPaths_mask, wordsEmbeddings, score