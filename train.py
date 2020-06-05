#!/usr/bin/env python3

import sys
if sys.version_info[0] < 3:
    raise Exception("Python 3 is required.")

import time

import numpy as np

import util

#import omnifold
import energyflow
from OmniFold import omnifold

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--padding', action='store_true')
    
    args = parser.parse_args()

    # read input
    inputfile = 'data/ttbar.npz'
    npzfile = np.load(inputfile, allow_pickle=True, encoding='bytes')
    data = npzfile['arr_0']

    #print(data)
    #print(data.dtype.names)
    
    #weights = data['w']

    npzfile.close()

    # format and prepare data
    preprocdata = util.prepare_data_omnifold(data)

    #print(preprocdata.shape)
    #print(preprocdata[3])

    variables = ['lep_pt','met_met', 'met_phi', 'mttReco']
    preprocdata2 = util.prepare_data_multifold(data, variables)

    print(preprocdata2.shape)
    print(preprocdata2[333])


    #how to deal with event weights of simulated sample? both reco and truth
    #what about negative weights?
    #how to deal with background if do event by event reweight
