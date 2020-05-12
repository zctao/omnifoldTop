#!/usr/bin/env python3

import numpy as np

def get_fourvector_array(pt_arr, eta_arr, phi_arr, e_arr, padding=True):
    """
    Combine and format pt, eta, phi, and e arrays into array of four vectors

    Each of the input array can be either an array of float or array of arrays
    with a shape of (nevents,) 

    If padding is set to True, pad the output array into homogeneous ndarray 
    with zeros, i.e. each event will have the same number of four vectors 
    Otherwise the output array will be an array of objects
    """

    # Check if the input arrays are of the same dimension
    if (not (pt_arr.shape==eta_arr.shape and pt_arr.shape==phi_arr.shape and pt_arr.shape==e_arr.shape)):
        raise Exception("Input arrays have to be of the same dimension.")

    # Stack arrays
    v4_arr = np.stack([pt_arr, eta_arr, phi_arr, e_arr], axis=1)
    
    # Determine if input arrays are arrays of floats or objects
    if pt_arr.dtype != np.dtype('O'):
        # array of float i.e. one four vector per event
        return v4_arr
    else:
        # array of objects (arrays)
        # i.e. each event can contain various number of four vectors

        stackarr = lambda v : np.stack([v[0],v[1],v[2],v[3]], axis=1)
        
        if not padding:
            v4_arr_new = np.asarray([stackarr(v4) for v4 in v4_arr])
            return v4_arr_new
        else:
            nv4max = max(len(v4[0]) for v4 in v4_arr)
            v4_arr_new = np.asarray([np.concatenate((stackarr(v4), np.zeros([nv4max-len(v4[0]),4]))) for v4 in v4_arr])
            return v4_arr_new


def prepare_data_omnifold(ntuple, padding=True):
    """
    ntuple: structure array from root tree

    return an numpy array of the shape (nevents, nparticles, 4)
    """

    # Total number of events
    nevents = len(ntuple)
    
    # jets
    jetv4 = get_fourvector_array(ntuple['jet_pt'], ntuple['jet_eta'], ntuple['jet_phi'], ntuple['jet_e'], padding=True)
    # shape: (nevents, NJetMax, 4)

    # lepton
    lepv4 = get_fourvector_array(ntuple['lep_pt'], ntuple['lep_eta'], ntuple['lep_phi'], ntuple['lep_e'], padding=True)
    # shape: (nevents, 4)

    # MET
    metv4 = get_fourvector_array(ntuple['met_met'], np.zeros(nevents), ntuple['met_phi'], ntuple['met_met'], padding=True)
    # shape: (nevent, 4)

    data = np.concatenate([np.stack([lepv4],axis=1), np.stack([metv4],axis=1), jetv4], axis=1)

    return data

#def prepare_data_multifold(ntuple, variables):
