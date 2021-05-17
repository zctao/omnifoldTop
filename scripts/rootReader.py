#!/usr/bin/env python3
import time
import tracemalloc
import uproot
import numpy as np
import awkward as ak
import numpy.lib.recfunctions as rfn
#import IPython

def getArrayFields(array):
    if isinstance(array, ak.Array):
        return array.fields
    elif isinstance(array, np.ndarray):
        return list(array.dtype.names)
    else:
        print("ERROR: don't know how to deal with array of the type", type(array))
        return []

def MeVtoGeV(array):
    fieldnames = getArrayFields(array)
    for fname in fieldnames:
        # jet_pt, jet_e, met_met, mwt, lep_pt, lep_m
        isObjectVar = fname in ['jet_pt', 'jet_e', 'met_met', 'mwt', 'lep_pt', 'lep_m']
        # MC_*_afterFSR_[pt,m,E]
        isPartonVar = fname.startswith('MC_') and ( fname.endswith('_pt') or fname.endswith('_m') or fname.endswith('_E') or fname.endswith('_Ht') or fname.endswith('_pout'))

        if isObjectVar or isPartonVar:
            try:
                array[fname] /= 1000.
            except TypeError: # cannot do /= on awkward array
                array[fname] = array[fname] / 1000.

    return array

def normalizeWeights(array, weight_name):
    # For now
    try:
        array[weight_name] /= np.mean(array[weight_name])
    except TypeError: # cannot do /= on awkward array
        array[weight_name] = array[weight_name] / np.mean(array[weight_name])

    return array

def readRoot(**parsed_args):
    # read input files
    treename_reco = 'reco'
    treename_truth = parsed_args['truth_level']

    intrees_reco = [fname+':'+treename_reco for fname in parsed_args['input_files']]
    intrees_truth = [fname+':'+treename_truth for fname in parsed_args['input_files']]

    # check events are truth matched
    if parsed_args['check_match']:
        reco_ids = uproot.lazy(intrees_reco, filter_name=['runNumber', 'eventNumber', 'isMatched'])
        truth_ids = uproot.lazy(intrees_truth, filter_name=['runNumber', 'eventNumber', 'isMatched'])

        runNum_reco = reco_ids.runNumber[reco_ids.isMatched==1]
        evtNum_reco = reco_ids.eventNumber[reco_ids.isMatched==1]

        runNum_truth = truth_ids.runNumber[truth_ids.isMatched==1]
        evtNum_truth = truth_ids.eventNumber[truth_ids.isMatched==1]

        assert(np.all(runNum_reco == runNum_truth) and np.all(evtNum_reco == evtNum_truth))

    # read branches
    branches_reco = [parsed_args['weight_name'], "isMatched"]
    branches_truth = ['isMatched']
    if parsed_args['truth_level'] == 'parton':
        branches_reco += ['klfitter_logLikelihood', 'klfitter_bestPerm_*']
        branches_truth += ['MC_ttbar_afterFSR_*', 'MC_thad_afterFSR_*', 'MC_tlep_afterFSR_*']
    else: # particle
        branches_reco += ["PseudoTop_*", "lep_pt", "jet_n"] #FIXME
        branches_truth += ["PseudoTop_*", "lep_pt", "jet_n"] #FIXME

    arrays_reco = uproot.lazy(intrees_reco, filter_name=branches_reco)
    arrays_truth = uproot.lazy(intrees_truth, filter_name=branches_truth)

    # convert units
    arrays_reco = MeVtoGeV(arrays_reco)
    arrays_truth = MeVtoGeV(arrays_truth)

    # normalizeWeights()

    arrays_reco_matched = arrays_reco[arrays_reco.isMatched==1]
    arrays_truth_matched = arrays_truth[arrays_truth.isMatched==1]

    if parsed_args['pad_unmatched']:
        # place unmatched reco events at the end and pad truth array with dummpy events to the same length
        arrays_reco_unmatched = arrays_reco[arrays_reco.isMatched==0]
        nparr_reco = ak.concatenate([arrays_reco_matched, arrays_reco_unmatched]).to_numpy()

        nparr_truth_matched = arrays_truth_matched.to_numpy()
        nparr_truth = np.pad(nparr_truth_matched, (0, len(arrays_reco_unmatched)), constant_values=-999.)
    else:
        nparr_reco = arrays_reco_matched.to_numpy()
        nparr_truth = arrays_truth_matched.to_numpy()

    # apply reco level selections
    if parsed_args['truth_level'] == 'parton':
        sel = nparr_reco['klfitter_logLikelihood'] > -52.0
        nparr_reco = nparr_reco[sel]
        nparr_truth = nparr_truth[sel]
    else:
        # cut on PseudoTop for now
        pass

    if parsed_args['pad_unmatched']:
        # place unmatched truth events at the end and pad reco array with dummy events
        arrays_truth_unmatched = arrays_truth[arrays_truth.isMatched==0]
        nparr_truth = np.concatenate([nparr_truth, arrays_truth_unmatched.to_numpy()])
        nparr_reco = np.pad(nparr_reco, (0, len(arrays_truth_unmatched)), constant_values=-999.)

    assert(len(nparr_reco)==len(nparr_truth))

    #####
    if parsed_args['truth_level'] == 'parton':
        # somehow some branches of the parton events have value nan or inf
        # e.g. 'MC_thad_afterFSR_y'
        sel_notnan = np.invert(np.isnan(nparr_truth['MC_thad_afterFSR_y']))
        nparr_reco = nparr_reco[sel_notnan]
        nparr_truth = nparr_truth[sel_notnan]
        assert(not np.any(np.isnan(nparr_truth['MC_thad_afterFSR_y'])))
    #####

    # rename truth array fields if needed
    truth_prefix = 'MC_' if parsed_args['truth_level'] == 'parton' else 'PL_'
    newnames = {}
    for fname in nparr_truth.dtype.names:
        if not fname.startswith(truth_prefix):
            newnames[fname] = truth_prefix+fname
    nparr_truth = rfn.rename_fields(nparr_truth, newnames)

    # join reco and truth array
    nparr_all = rfn.merge_arrays([nparr_reco, nparr_truth], flatten=True)
    #IPython.embed()

    # write to disk
    np.savez(parsed_args['output_name'], nparr_all)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('input_files', nargs='+', type=str,
                        help="list of input root files")
    parser.add_argument('-t', '--truth-level', choices=['parton', 'particle'],
                        required=True, help="Parton or particle level")
    parser.add_argument('-w', '--weight-name', default='totalWeight_nominal',
                        help="Name of the weight column")
    parser.add_argument('-o', '--output-name', default='ntuple.npz',
                        help="Output file name")
    parser.add_argument('-c', '--check-match', action='store_true',
                        help="Check if the events are truth matched")
    parser.add_argument('-p', '--pad-unmatched', action='store_true',
                        help="Pad unmatched reco and truth arrays")

    args = parser.parse_args()

    tracemalloc.start()

    tstart = time.time()
    readRoot(**vars(args))
    tdone = time.time()
    print("readRoot took {:.2f} seconds".format(tdone - tstart))

    mcurrent, mpeak = tracemalloc.get_traced_memory()
    print("Current memory usage is {:.1f} MB; Peak was {:.1f} MB".format(mcurrent * 10**-6, mpeak * 10**-6))
