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
