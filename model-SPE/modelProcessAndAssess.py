#encoding=utf-8
'''
Functions and Application : 
Get the process model and test
'''

import numpy
import theano
from theano import tensor
from collections import OrderedDict
import dataProcessTools
import toolsFunction
import evaluateTools
import gc
import subgraphAttentionProcessModelLSTMBatch


def get_metagraphAttentionModel(
                    
                     model_params_path='', # model save path
                     metagraph_embedding_dimension=10, # metagraph embedding  dimension
                     dimension_A=10, # the dimension of attention when computing the m-node embedding
                     dimension_lstm=10, # dimension of lstm parameters
                     dimension_B=10, # the dimension of attention when computing the m-path embedding
                     dimension_C=10, # the dimension of attention when computing the m-paths embedding
                      ):
    """
    get the MPE process Model
    """
    model_options = locals().copy()
    
    tparams = OrderedDict()
    tparams['Q_A']=None
    tparams['b_A']=None
    tparams['eta_A']=None
    tparams['lstm_W']=None
    tparams['lstm_U']=None
    tparams['lstm_b']=None
    tparams['Q_B']=None
    tparams['b_B']=None
    tparams['eta_B']=None
    tparams['Q_C']=None
    tparams['b_C']=None
    tparams['eta_C']=None
    tparams['w']=None
    tparams=load_params(model_params_path, tparams) 
    
    metagraphEmbeddings, subPaths_matrix, subPaths_mask, wordsEmbeddings, score=subgraphAttentionProcessModelLSTMBatch.metagraphAttentionProcessModel(model_options, tparams)
    func=theano.function([metagraphEmbeddings, subPaths_matrix, subPaths_mask, wordsEmbeddings], score) 
    return func 

def load_params(path, params):
    """
    load all the parameters
    """
    pp = numpy.load(path) 
    for kk, vv in params.items():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params

def compute_metagraphAttention(
                     wordsEmbeddings=None, # words embeddings
                     wordsEmbeddings_path=None, # the file path of words embeddings
                     metagraphEmbeddings_path=None, # the file path of metagraph embeddings
                     wordsSize=0, # the size of words vocabulary
                     subpaths_map=None, # contains sub-paths
                     subpaths_file=None, # the file which contains sub-paths
                     maxlen_subpaths=1000, # the max length for sub-paths
                     
                     test_data_file='', # test data file
                     top_num=10, # top num in experiments
                     ideal_data_file='', # ideal data file
                     func=None, # the MPE process model
                   ):
    """
        evaluate the MPE model
    """
    model_options = locals().copy()
    
    if wordsEmbeddings is None: 
        if wordsEmbeddings_path is not None: 
            wordsEmbeddings,dimension,wordsSize=dataProcessTools.getWordsEmbeddings(wordsEmbeddings_path)
        else: 
            print 'There is not path for wordsEmbeddings, exit!!!'
            exit(0) 

    if subpaths_map is None: 
        if subpaths_file is not None:
            subpaths_map=dataProcessTools.loadAllSubPathsRomove0Path(subpaths_file, maxlen_subpaths, wordsEmbeddings)
        else: 
            print 'There is not path for sub-paths, exit!!!'
            exit(0)
            
    metagraphEmbedding_data, metagraphDimension, metagraphSize=dataProcessTools.getMetagraphEmbeddings(metagraphEmbeddings_path)

    line_count=0 
    test_map={} 
    print 'Compute MAP and nDCG for file ',test_data_file
    with open(test_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            query=int(arr[0]) 
            map={} 
            for i in range(1,len(arr)): 
                candidate=int(arr[i]) 
                subPaths_matrix_data,subPaths_mask_data,subPaths_lens_data=dataProcessTools.prepareDataForTest(query, candidate, subpaths_map)
                if subPaths_matrix_data is None and subPaths_mask_data is None and subPaths_lens_data is None: 
                    map[candidate]=-1000. 
                else: 
                    value=func(metagraphEmbedding_data, subPaths_matrix_data, subPaths_mask_data, wordsEmbeddings)
                    map[candidate]=value
                del subPaths_matrix_data
                del subPaths_mask_data
                del subPaths_lens_data
            tops_in_line=toolsFunction.mapSortByValueDESC(map, top_num)
            test_map[line_count]=tops_in_line 
            line_count+=1 
            map=None
            gc.collect()
                
    
    line_count=0 
    ideal_map={}
    with open(ideal_data_file) as f: 
        for l in f: 
            arr=l.strip().split()
            arr=[int(x) for x in arr] 
            ideal_map[line_count]=arr[1:] 
            line_count+=1 
    
    MAP=evaluateTools.get_MAP(top_num, ideal_map, test_map)
    MnDCG=evaluateTools.get_MnDCG(top_num, ideal_map, test_map)
    
    return MAP,MnDCG
    
    