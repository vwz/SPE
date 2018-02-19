#encoding=utf-8
'''
Functions and Application : 
the MPE model
'''
import os
import numpy
import theano
from theano import tensor
from theano.ifelse import ifelse
import lstmModel

def metagraphAttentionModel(options,tparams):
    """
        the MPE model
    """
    metagraphEmbeddings=tensor.matrix('metagraphEmbeddings',dtype=theano.config.floatX) # @UndefinedVariable # shape=#(metagraph)*len(metaEmbeding)
    trainingParis=tensor.tensor3('trainingParis',dtype='int64') # 3D tensor,shape=#(triples)*4*2
    subPaths_matrix=tensor.matrix('subPaths_matrix',dtype='int64') # shape=maxlen*#(sub-paths)
    subPaths_mask=tensor.matrix('subPaths_mask',dtype=theano.config.floatX)  # @UndefinedVariable # shape=maxlen*#(sub-paths)
    wordsEmbeddings=tensor.matrix('wordsEmbeddings',dtype=theano.config.floatX)  # @UndefinedVariable # #(words)*word_dimension
    
    metagraphBeta=tensor.exp(tensor.dot(tensor.nnet.sigmoid(tensor.dot(metagraphEmbeddings, tparams['Q_A'])+tparams['b_A']), tparams['eta_A']))
    
    def _processTriple(fourPairs,lossSum):
        
        def _processSubpathsBatch(start, end):
            x=subPaths_matrix[:,start:end] # shape=maxlen*nsamples
            x_mask=subPaths_mask[:,start:end] 
            
            wordEmb=wordsEmbeddings[x] # shape=maxlen*nsamples*len(wordsEmbeddings)
            tmp=wordEmb*metagraphBeta
            softmax0=tmp/tmp.sum(axis=-1, keepdims=True) # shape=maxlen*nsamples*len(wordsEmbeddings) 
            subpaths=((softmax0*wordEmb)[:,:,:,None]*metagraphEmbeddings).sum(axis=2)
            
            h3Dmatrix=lstmModel.get_lstm(options, tparams, subpaths, x_mask)
            beta1=tensor.exp(tensor.dot(tensor.nnet.sigmoid(tensor.dot(h3Dmatrix, tparams['Q_B'])+tparams['b_B']), tparams['eta_B'])) 
            temp=x_mask*beta1 
            # softmax1 shape=maxlen*nsamples
            softmax1=temp/temp.sum(axis=0, keepdims=True)
            # shape=nsamples*lstm_dimension
            pathsEmb=(softmax1[:,:,None]*h3Dmatrix).sum(axis=0)
            
            return pathsEmb 
        
        def iftFunc():
            embx=numpy.zeros(options['dimension_lstm'],).astype(theano.config.floatX)  # @UndefinedVariable 
            return embx
         
        def iffFunc(start,end):
            embx=None
            rval2=_processSubpathsBatch(start, end)
            # beta shape=paths_num * 0
            beta=tensor.dot(tensor.nnet.sigmoid(tensor.dot(rval2, tparams['Q_C'])+tparams['b_C']), tparams['eta_C'])
            softmax2=tensor.nnet.softmax(beta)[0] 
            embx=(softmax2[:,None]*rval2).sum(axis=0) # shape=dimension_lstm*0
            
            return embx
        
        # get emb1
        start=fourPairs[0][0] 
        end=fourPairs[1][1] 
        emb1=None 
        emb1=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) 
        # get emb2
        start=fourPairs[2][0] 
        end=fourPairs[3][1]
        emb2=None 
        emb2=ifelse(tensor.eq(start,end),iftFunc(),iffFunc(start,end)) 
            
        loss=0
        param=options['objective_function_param'] 
        if options['objective_function_method']=='sigmoid': 
            loss=-tensor.log(tensor.nnet.sigmoid(param*(tensor.dot(emb1,tparams['w'])-tensor.dot(emb2,tparams['w'])))) # sigmoid
        else: # hinge-loss
            value=param + tensor.dot(emb2,tparams['w']) - tensor.dot(emb1,tparams['w'])
            loss=value*(value>0)
        
        return loss+lossSum
        
    rval,update=theano.scan(
                            _processTriple,
                            sequences=trainingParis, 
                            outputs_info=tensor.constant(0., dtype=theano.config.floatX), # @UndefinedVariable 
                            )
    cost=rval[-1]
    
    cost+=options['decay_Q_A']*(tparams['Q_A'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['b_A'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['eta_A'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['lstm_W'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['lstm_U'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['lstm_b'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['Q_B'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['b_B'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['eta_B'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['Q_C'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['b_C'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['eta_C'] ** 2).sum()
    cost+=options['decay_Q_A']*(tparams['w'] ** 2).sum()
    
    # return MPE model
    return metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, wordsEmbeddings, cost
    