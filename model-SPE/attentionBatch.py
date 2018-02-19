#encoding=utf-8
'''
Functions and Application : 
MPE model
'''
import numpy
import theano
from theano import tensor
import dataProcessTools
from theano import config
from collections import OrderedDict
import time
import six.moves.cPickle as pickle  # @UnresolvedImport
import gc
import subgraphAttentionModelLSTMBatch
# theano.config.floatX = 'float32'

SEED = 123
numpy.random.seed(SEED)

def numpy_floatX(data):
    return numpy.asarray(data, dtype=theano.config.floatX)  # @UndefinedVariable

def gradientDescentGroup(learning_rate,tparams,grads,metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, subPaths_lens, wordsEmbeddings, cost):
    update=[(shared,shared-learning_rate*g) for g,shared in zip(grads,tparams.values())]
    func=theano.function([metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, subPaths_lens, wordsEmbeddings],cost,updates=update,on_unused_input='ignore',mode='FAST_RUN')
    return func

def adadelta(lr, tparams, grads, metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, wordsEmbeddings, cost):
    """
    An adaptive learning rate optimizer
    Parameters
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
    f_grad_shared = theano.function([metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, wordsEmbeddings], cost, updates=zgup + rg2up,
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


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0., name='%s_grad' % k)
               for k, p in tparams.items()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update

def ortho_weight(ndim):
    """
        init a matrix by svd
    """
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(theano.config.floatX)  # @UndefinedVariable

def init_params_weight(row,column):
    """
   init a matrix
    """
    W = numpy.random.rand(row, column) 
    W = W*2.0-1.0 
    return W.astype(theano.config.floatX)  # @UndefinedVariable


def init_sharedVariables(options):
    """
        inti the shared variables
    """
    print 'init shared Variables......'
    params = OrderedDict()
    # Q_A
    Q_A=init_params_weight(options['metagraph_embedding_dimension'],options['dimension_A']) # (-1,1)
    params['Q_A']=Q_A
    # b_A
    b_A=numpy.random.rand(options['dimension_A'], ) 
    params['b_A']=b_A
    # eta_A
    eta_A=numpy.random.rand(options['dimension_A'], ) 
    params['eta_A']=eta_A
    
    
    lstm_W=numpy.concatenate([
                              init_params_weight(options['metagraph_embedding_dimension'],options['dimension_lstm']),
                              init_params_weight(options['metagraph_embedding_dimension'],options['dimension_lstm']),
                              init_params_weight(options['metagraph_embedding_dimension'],options['dimension_lstm']),
                              init_params_weight(options['metagraph_embedding_dimension'],options['dimension_lstm'])
                              ],axis=1) 
    params['lstm_W'] = lstm_W
    lstm_U = numpy.concatenate([ortho_weight(options['dimension_lstm']),
                           ortho_weight(options['dimension_lstm']),
                           ortho_weight(options['dimension_lstm']),
                           ortho_weight(options['dimension_lstm'])], axis=1)
    params['lstm_U'] = lstm_U
    lstm_b = numpy.zeros((4 * options['dimension_lstm'],))
    params['lstm_b'] = lstm_b.astype(theano.config.floatX)  # @UndefinedVariable
    
    # Q_B
    Q_B=init_params_weight(options['dimension_lstm'],options['dimension_B']) # (-1,1)
    params['Q_B']=Q_B
    # b_B
    b_B=numpy.random.rand(options['dimension_B'], ) # (0,1)
    params['b_B']=b_B
    # eta_B
    eta_B=numpy.random.rand(options['dimension_B'], ) # (0,1)
    params['eta_B']=eta_B
    
    # Q_C
    Q_C=init_params_weight(options['dimension_lstm'],options['dimension_C']) # (-1,1)
    params['Q_C']=Q_C
    # b_C
    b_C=numpy.random.rand(options['dimension_C'], ) # (0,1)
    params['b_C']=b_C
    # eta_C
    eta_C=numpy.random.rand(options['dimension_C'], ) # (0,1)
    params['eta_C']=eta_C
    
    w = numpy.random.rand(options['dimension_lstm'], ) # (0,1)
    params['w']=w.astype(theano.config.floatX)  # @UndefinedVariable
    
    return params

def init_tparams(params): # set shared variables
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams
    
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params

main_dir='D:/dataset/test/icde2016_metagraph/'
def metagraphAttentionTraining(
                               
                     trainingDataFile=main_dir+'facebook.splits/train.10/train_classmate_1', # the full path of training data file
                     metagraphEmbeddings_path='', # the file path of metagraph embeddings
                     wordsEmbeddings_data=None, # words embeddings
                     wordsEmbeddings_path=main_dir+'facebook/nodesFeatures', # the file path of words embeddings
                     wordsSize=1000000, # the size of words vocabulary
                     subpaths_map=None, # contains sub-paths
                     subpaths_file=main_dir+'facebook/subpathsSaveFile',# the file which contains sub-paths
                     maxlen_subpaths=1000, # the max length for sub-paths
                     maxlen=100,  # Sequence longer then this get ignored 
                     batch_size=10, # use a batch for training. This is the size of this batch.
                     is_shuffle_for_batch=True, # if need shuffle for training
                     objective_function_method='sigmoid', # loss function, we use sigmoid here
                     objective_function_param=0, # the parameter in loss function, beta
                     lrate=0.0001, # learning rate
                     max_epochs=100, # the max epochs for training
                     
                     dispFreq=5, # the frequences for display
                     saveFreq=5, # the frequences for saving the parameters
                     saveto=main_dir+'facebook/path2vec-modelParams.npz', # the path for saving parameters. It is generated by main_dir, dataset_name, suffix, class_name and index.
                     
                     # all dimensions parameters
                     metagraph_embedding_dimension=10, # metagraph embedding dimension 
                     dimension_A=10, # the dimension of attention when computing the m-node embedding
                     dimension_lstm=10, # dimension of lstm parameters
                     dimension_B=10, # the dimension of attention when computing the m-path embedding
                     dimension_C=10, # the dimension of attention when computing the m-paths embedding
                     
                     # decay parameters
                     decay_Q_A=0.001,
                     decay_b_A=0.001,
                     decay_eta_A=0.001,
                     decay_lstm_W=0.001, 
                     decay_lstm_U=0.001,
                     decay_lstm_b=0.001, 
                     decay_Q_B=0.001,
                     decay_b_B=0.001,
                     decay_eta_B=0.001,
                     decay_Q_C=0.001,
                     decay_b_C=0.001,
                     decay_eta_C=0.001,
                     decay_w=0.001, 
                     
                               ):
    # get all parameters
    model_options = locals().copy()
    
    if wordsEmbeddings_data is None: 
        if wordsEmbeddings_path is not None:
            wordsEmbeddings_data,dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            print 'There is not path for wordsEmbeddings, exit!!!'
            exit(0) 
    
    if subpaths_map is None: 
        if subpaths_file is not None: 
            subpaths_map=dataProcessTools.loadAllSubPathsRomove0Path(subpaths_file, maxlen_subpaths, wordsEmbeddings_data)
        else:
            print 'There is not path for sub-paths, exit!!!'
            exit(0)
    
    metagraphEmbedding_data, metagraphDimension, metagraphSize=dataProcessTools.getMetagraphEmbeddings(metagraphEmbeddings_path)
    
    trainingData,trainingPairs_data=dataProcessTools.getTrainingData(trainingDataFile)
    allBatches=dataProcessTools.get_minibatches_idx(len(trainingData), batch_size, is_shuffle_for_batch)
    
    '''
        init shared variables
    '''
    params=init_sharedVariables(model_options) 
    tparams=init_tparams(params)
    print 'Generate models ......'
    
    metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, wordsEmbeddings, cost=subgraphAttentionModelLSTMBatch.metagraphAttentionModel(model_options, tparams)
    
    print 'Generate gradients ......'
    grads=tensor.grad(cost,wrt=list(tparams.values()))
    print 'Using Adadelta to generate functions ......'
    this_time = time.time()
    print 'Start to compile and optimize, time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(this_time))
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update=adadelta(lr, tparams, grads, metagraphEmbeddings, trainingParis, subPaths_matrix, subPaths_mask, wordsEmbeddings, cost)
    
    print 'Start training models ......'
    best_p = None 
    history_cost=[] # not use
    
    start_time = time.time() 
    print 'start time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))
    uidx=0 
    for eidx in range(max_epochs):
        for _, batch in allBatches: 
            uidx += 1
            # prepare data for this model
            trainingDataForBatch=[trainingData[i] for i in batch]
            trainingPairsForBatch=[trainingPairs_data[i] for i in batch]
            triples_matrix_data, subPaths_matrix_data, subPaths_mask_data, subPaths_lens_data=dataProcessTools.prepareDataForTraining(trainingDataForBatch, trainingPairsForBatch, subpaths_map)
            cost=0
            cost=f_grad_shared(metagraphEmbedding_data, triples_matrix_data, subPaths_matrix_data, subPaths_mask_data,wordsEmbeddings_data)
            f_update(lrate)
            
            trainingDataForBatch=None 
            trainingPairsForBatch=None
            del triples_matrix_data
            del subPaths_matrix_data
            del subPaths_mask_data
            del subPaths_lens_data
            
            if numpy.isnan(cost) or numpy.isinf(cost):
                print('bad cost detected: ', cost)
                return 
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch =', eidx, ',  Update =', uidx, ',  Cost =', cost
                this_time = time.time() 
                print 'Time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(this_time))
            if saveto and numpy.mod(uidx, saveFreq) == 0:
                print('Saving...')
                if best_p is not None: 
                    params = best_p
                else: 
                    params = unzip(tparams)
                
                numpy.savez(saveto, history_errs=history_cost, **params)
                pickle.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                print('Done')
        gc.collect()
        
    end_time = time.time() 
    print 'end time ==',time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))
    print 'Training finished! Cost time == ', end_time-start_time,' s'
            
    