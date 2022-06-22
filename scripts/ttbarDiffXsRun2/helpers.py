"""
Helper functions to deal with TTbar DiffXs Run 2 measurement files
"""
import os
import uproot

fileNameMap = {
    'th_pt' : 'truth_4j2b_ljets_PseudoTop_Reco_top_had_pt_central_ljets.root',
    'th_y' : 'truth_4j2b_ljets_PseudoTop_Reco_top_had_y_central_ljets.root',
    'th_y_abs' : 'truth_4j2b_ljets_PseudoTop_Reco_top_had_abs_y_central_ljets.root',
    'tl_pt' : 'truth_4j2b_ljets_PseudoTop_Reco_top_lep_pt_central_ljets.root',
    'tl_y' : 'truth_4j2b_ljets_PseudoTop_Reco_top_lep_y_central_ljets.root',
    'tl_y_abs' : 'truth_4j2b_ljets_PseudoTop_Reco_top_lep_abs_y_central_ljets.root',
    'mtt' : 'truth_4j2b_ljets_PseudoTop_Reco_ttbar_m_central_ljets.root',
    'ptt' : 'truth_4j2b_ljets_PseudoTop_Reco_ttbar_pt_central_ljets.root',
    'ytt' : 'truth_4j2b_ljets_PseudoTop_Reco_ttbar_y_central_ljets.root',
    'ytt_abs' : 'truth_4j2b_ljets_PseudoTop_Reco_ttbar_abs_y_central_ljets.root',
    't1_pt' : 'truth_4j2b_ljets_PseudoTop_Reco_leading_top_pt_central_ljets.root',
    't2_pt' : 'truth_4j2b_ljets_PseudoTop_Reco_subleading_top_pt_central_ljets.root',
}

def get_filepath(observable, binned_correction_dir):

    if not os.path.isdir(binned_correction_dir):
        print(f"ERROR: cannot find directory {binned_correction_dir}")
        return None

    fname = fileNameMap.get(observable)
    if fname is None:
        print(f"WARNING: correction is not available for {observable}")
        return None

    # look for the file in binned_correction_dir
    fpath = None
    for cwd, subdirs, files in os.walk(binned_correction_dir):
        if fname in files:
            fpath = os.path.join(cwd, fname)
            break

    if fpath is None:
        print(f"ERROR: cannot find file {fname} in {binned_correction_dir}")

    return fpath

def read_hist_from_file(filepath, histname):
    h = None

    with uproot.open(filepath) as f:
        if histname in f:
            h = f[histname].to_hist()
        else:
            print(f"ERROR: no {histname} in {filepath}")

    return h

def get_acceptance_correction(observable, binned_correction_dir):

    fpath = get_filepath(observable, binned_correction_dir)

    if fpath is None:
        return None
    else:
        return read_hist_from_file(fpath, 'Acceptance')

def get_efficiency_correction(observable, binned_correction_dir):

    fpath = get_filepath(observable, binned_correction_dir)

    if fpath is None:
        return None
    else:
        return read_hist_from_file(fpath, 'Efficiency')
