#encoding=utf-8
'''
Functions and Application : 
autoencoder training for metagraph embedding
'''

import numpy
import theano
from theano import tensor
from collections import OrderedDict
import time
import six.moves.cPickle as pickle  # @UnresolvedImport
from theano import config
import dataToolsForAutoencoder
import autoencoderModel
import autoencoderCalculate

# Set the random number generators' seeds for consistency
SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)  # @UndefinedVariable


def gradientDescentGroup(learning_rate, tparams, grads, x, cost):
    update=[(shared,shared-learning_rate*g) for g,shared in zip(grads,tparams.values())]
    func=theano.function([x],cost,updates=update,on_unused_input='ignore')
    return func

def adadelta(lr, tparams, grads, x, cost):
    """
    An adaptive learning rate optimizer
    ----------
    lr : Theano SharedVariable
        Initial learning rate
    tpramas: Theano SharedVariable
        Model parameters
    grads: Theano variable
        Gradients of cost w.r.t to parameres
    x: Theano variable
        Model inputs
    mask: Theano variable
        Sequence mask
    y: Theano variable
        Targets
    cost: Theano variable
        Objective fucntion to minimize

    Notes
    -----
    For more information, see [ADADELTA]_.

    .. [ADADELTA] Matthew D. Zeiler, *ADADELTA: An Adaptive Learning
       Rate Method*, arXiv:1212.5701.
    """
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.items()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.items()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.items()]
    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]
    f_grad_shared = theano.function([x], cost, updates=zgup + rg2up,
                                    on_unused_input='ignore',
                                    name='adadelta_f_grad_shared')
    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) 
             for ru2, ud in zip(running_up2, updir)] 
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)] 
    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def init_sharedVariables(options):
    """
        init shared variables
    """
    print 'init shared Variables......'
    params = OrderedDict()
    w = numpy.random.rand(options['dimension2'], options['dimension1']) 
    w = w*2.0-1.0 
    params['w']=w.astype(config.floatX)  # @UndefinedVariable
    w2 = numpy.random.rand(options['dimension1'], options['dimension2']) 
    w2 = w2*2.0-1.0 
    params['w2']=w2.astype(config.floatX)  # @UndefinedVariable


    b1 = numpy.random.rand(options['dimension2'], ) 
    b1 = b1*2.0-1.0 
    params['b1']=b1.astype(config.floatX)  # @UndefinedVariable
    b2 = numpy.random.rand(options['dimension1'], ) 
    b2 = b2*2.0-1.0 
    params['b2']=b2.astype(config.floatX)  # @UndefinedVariable
    return params

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

def load_params(path, params):
    """
    load the trained parameters from file
    """
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def autoencoderTraining(
                              metagraphStructuralSimFilePath='', # metagraph structural similarity file
                              dimension1=981, # dimension of input vectors
                              dimension2=30, # dimension of the result
                              lrate=0.0001, # learning rate
                              max_epochs=100, # epochs
                              decay_w=0.0001, # decay
                              decay_b1=0.0001, # decay
                              decay_b2=0.0001, # decay
                              batch_size=20, # training batch size
                              is_shuffle_for_batch=True, # is shuffle for batch
                              dispFreq=5, # display frequence
                              saveFreq=5, # parameters save frequence
                              saveto='', # save destination
                              embeddingsSaveFile='', # result file
                              ):
    """
       training method
    """
    model_options = locals().copy()
    
    """
        get all data
    """
    # 的structural similarity matrix
    x_allData=dataToolsForAutoencoder.readMetagraphStructuralSim(metagraphStructuralSimFilePath)
    
    # batch size
    allBatches=dataToolsForAutoencoder.get_minibatches_idx(len(x_allData), batch_size, is_shuffle_for_batch)
    
    """
        init shared variables
    """
    params=init_sharedVariables(model_options) 
    tparams=init_tparams(params) 
    print 'Generate SelectSignificantFeatures model ......'
    x, cost=autoencoderModel.autoencoderModel(model_options, tparams)
    
    print 'Generate gradients ......'
    grads=tensor.grad(cost,wrt=list(tparams.values()))
    
    print 'Using Adadelta to generate functions ......'
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update=adadelta(lr, tparams, grads, x, cost)
    
    """
        training
    """
    start_time = time.time() 
    print 'start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    best_p = None 
    history_cost=[] 
    uidx=0 # update index
    for eidx in range(max_epochs):
        for _, batch in allBatches:
            uidx+=1
            x_data=[x_allData[i] for i in batch]
            x_data=numpy.asarray(x_data)
            
            cost_data=f_grad_shared(x_data)
            f_update(lrate)
            
            if numpy.isnan(cost_data) or numpy.isinf(cost_data):
                print('bad cost detected: ', cost_data)
                return 
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch =', eidx, ',  Update =', uidx, ',  Cost =', cost_data
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                if best_p is not None: 
                    params = best_p
                else: 
                    params = unzip(tparams)
                
                numpy.savez(saveto, history_errs=history_cost, **params)
                pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print('Done')
    
    end_time = time.time() 
    print 'end time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
    print 'Training finished! Cost time == ', end_time-start_time,' s'
    
    x,y=autoencoderCalculate.autoencoderCalculateModel(tparams) 
    calculateF=theano.function([x],y)
    output = open(embeddingsSaveFile, 'w')
    output.write(bytes(dimension1)+'\t'+bytes(dimension2)+'\n') 
    for i in range(len(x_allData)):
        embedding=calculateF(x_allData[i])
        output.write(bytes(i)+'\t') 
        for j in embedding: 
            output.write(bytes(j)+'\t')
        output.write('\n')
    output.flush()
    output.close()
    print 'Complete writing !!!'
    
root_dir='D:/dataset/icde2016/dataset/metagraph-structural-similarity/'
datasetName='linkedin'
if __name__=='__main__':
    autoencoderTraining(
                              metagraphStructuralSimFilePath=root_dir+datasetName+'-metagraph.sim', 
                              dimension1=173, # dimension of input vectors
                              dimension2=64, # dimension of the result
                              lrate=0.1, # learning rate
                              max_epochs=50, # max epochs
                              decay_w=0.1, # decay
                              decay_b1=0.1, # decay
                              decay_b2=0.1, # decay
                              batch_size=50, # batch size
                              is_shuffle_for_batch=True, # is shuffle for batch
                              dispFreq=5, # display frequence
                              saveFreq=5, # save frequence
                              saveto=root_dir+datasetName+'.saveto.npz', # parameters save file
                              embeddingsSaveFile=root_dir+datasetName+'-metagraph.embedding', # embedding results
                              )