#encoding=utf-8
'''
Functions and Application : 
data process methods
'''

import numpy
import theano
from theano import tensor

def readMetagraphStructuralSim(filepath):
    """
        read metagraph structural similarity 
    """
    sim=None 
    dimension=0 
    row=0 
    column=0 
    with open(filepath) as f:
        for l in f:
            tmp=l.strip().split()
            if len(tmp)>0:
                if len(tmp)==1: 
                    dimension=int(tmp[0])
                    sim=numpy.ones((dimension,dimension)).astype(theano.config.floatX)  # @UndefinedVariable
                    continue
                else:
                    row=int(tmp[0])
                    column=int(tmp[1])
                    sim[row][column]=float(tmp[2])
                    sim[column][row]=float(tmp[2])
    return sim
                    
    
    
def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """
    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

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


if __name__=='__main__':
    filepath='d:/test/write'
    sim=readMetagraphStructuralSim(filepath)
    print sim[0]
    print sim

