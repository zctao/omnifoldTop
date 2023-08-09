import numpy as np
import math
import uproot
import os

from datahandler import DataHandlerBase, filter_variable_names
import plotter

import logging
logger = logging.getLogger('datahandler_root')

def MeVtoGeV(array):
    """
    Convert unit from MeV to GeV

    Parameters
    ----------
    array : numpy structured array

    """
    for fname in list(array.dtype.names):
        # jet_pt, jet_e, met_met, mwt, lep_pt, lep_m
        isObjectVar = fname in ['jet_pt', 'jet_e', 'met_met', 'mwt', 'lep_pt', 'lep_m']
        # MC_*_afterFSR_[pt,m,E, Ht, pout]
        isPartonVar = fname.startswith('MC_') and (
            fname.endswith('_pt') or fname.endswith('_m') or
            fname.endswith('_E') or fname.endswith('_Ht') or
            fname.endswith('_pout')
            )

        if isObjectVar or isPartonVar:
            array[fname] /= 1000.

def load_arrays(
    file_names,
    tree_name,
    variable_names = None
    ):
    """
    Load data from a list of ROOT files
    Return a structured numpy array of data

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of root files to load
    tree_name : str
        Name of the tree in root files
    variable_names : list of str
        List of variables to read. If not provided, read all available variables

    Returns
    -------
    A numpy ndarray for event features
    """
    if isinstance(file_names, str):
        file_names = [file_names]

    data_array = uproot.concatenate(
        [f"{fname}:{tree_name}" for fname in file_names],
        variable_names
        )

    # convert awkward array to numpy array for now
    data_array = data_array.to_numpy()

    # convert units
    MeVtoGeV(data_array)

    return data_array

def read_weight_array(
        filename, # str, name of the root file to read
        tree_name, # str, name of TTree to read
        weight_nominal, # str, name of the nominal weight
        weight_component = None, # str, name of the weight component for normalization
        weight_variation = None, # str, name of the weight from systematic uncertainty variation
        weight_index = None, # int, index of the weight variation in case it is a vector  
    ):

    with uproot.open(filename) as f:
        events = f[tree_name]

        # nominal event weights
        if not weight_nominal in events:
            logger.error(f"Unkown branch {weight_nominal} in {filename}")
            return None

        warr = events[weight_nominal].array().to_numpy()

        warr_comp = None
        if weight_component is not None:
            if not weight_component in events:
                logger.warn(f"No branch {weight_component} in {filename}. Will use nominal event weights.")
            else:
                warr_comp = events[weight_component].array().to_numpy()
                assert(warr.shape == warr_comp.shape)

        warr_syst = None
        if weight_variation is not None:
            if not weight_variation in events:
                logger.warn(f"No branch {weight_variation} in {filename}. Will use nominal event weights.")
            else:
                if weight_index is None:
                    warr_syst = events[weight_variation].array().to_numpy()
                else:
                    warr_syst = events[weight_variation].array().to_numpy()[:,weight_index]
                assert(warr.shape == warr_syst.shape)

    #warr *= warr_syst / warr_comp
    if warr_comp is not None and warr_syst is not None:
        sf = np.zeros_like(warr, float)
        np.divide(warr_syst, warr_comp, out = sf, where = warr_comp!=0)
        warr *= sf

    return warr

def load_weights(
        file_names,
        tree_name,
        weight_name,
        weight_type = 'nominal'
    ):
    """
    Load event weights from ROOT files

    Parameters
    ----------
    file_names : str or file-like object; or sequence of str or file-like objects
        List of root files to load
    tree_name : str
        Name of the tree in root files
    weight_name : str
        Name of the TTree branch that stores the nominal event weights
    weight_type : str
        Type of the event weights variations. Default: nominal

    Returns
    -------
    A numpy ndarray of event weights
    """

    # parse weight_type
    if weight_type != 'nominal':
        # Examples of expected 'weight_type':
        # 'weight_pileup_UP' or 'weight_bTagSF_DL1r_70_eigenvars_B_up:5'
        if len(weight_type.split(':')) > 1:
            # The weight variation branch is a vector of float
            weight_syst, index_w = weight_type.split(':')
            weight_syst = weight_syst.strip()
            index_w = int(index_w.strip())
        else:
            weight_syst = weight_type
            index_w = None

        # component of the nominal weights corresponding to the weight variation
        # All components of the nominal weights (hard code here for now)
        weight_comp = None
        all_weight_components = ["weight_bTagSF_DL1r_70", "weight_jvt", "weight_leptonSF", "weight_pileup", "weight_mc"]
        for wname in all_weight_components:
            if weight_syst.startswith(wname):
                weight_comp = wname
                break

        # A special case for the MC generator weight variations
        if weight_syst == 'mc_generator_weights':
            weight_comp = 'weight_mc'

        if weight_comp is None: # something's wrong
            raise RuntimeError(f"Unknown base component for event weight {weight_type}")

    else:
        weight_syst = None
        weight_comp = None
        index_w = None

    # loop over input files
    if isinstance(file_names, str):
        file_names = [file_names]

    weights_arr = np.empty(shape=(0,))

    # read the weight array one file at a time because not all files contain the same type of weight arrays e.g. data-driven fakes do not have pileup weights 
    for fname in file_names:

        weights_arr = np.concatenate([
            weights_arr,
            read_weight_array(
                fname,
                tree_name,
                weight_nominal = weight_name,
                weight_component = weight_comp,
                weight_variation = weight_syst,
                weight_index = index_w
            )
            ])

    return weights_arr

def select_reco(file_names, treename='reco'):
    branches = ['isDummy', "normalized_weight"]
    # For KLFitter
    # branches += ['klfitter_logLikelihood']

    array_reco = uproot.concatenate(
        [f"{fname}:{treename}" for fname in file_names],
        branches).to_numpy()

    passcuts = array_reco['isDummy'] == 0
    passcuts &= (array_reco['normalized_weight'] != 0)
    #passcuts &= (array_reco['klfitter_logLikelihood'] > -52.)

    return passcuts

def select_parton(file_names, treename='parton'):
    branches = ['isDummy', "MC_thad_afterFSR_y"]

    array_parton = uproot.concatenate(
        [f"{fname}:{treename}" for fname in file_names],
        branches).to_numpy()

    passcuts = array_parton['isDummy'] == 0
    passcuts &= (~np.isnan(array_parton['MC_thad_afterFSR_y']))

    return passcuts

def compute_dphi(phi_arr1, phi_arr2):

    if not isinstance(phi_arr1, np.ndarray) or not isinstance(phi_arr2, np.ndarray):
        dphi = (phi_arr1 - phi_arr2).to_numpy()
    else:
        dphi = phi_arr1 - phi_arr2

    sel_gt_pi = dphi > math.pi
    sel_lt_mpi = dphi < -math.pi

    dphi[sel_gt_pi] -= 2*math.pi
    dphi[sel_lt_mpi] += 2*math.pi

    return dphi

def compute_dR(phi_arr1, y_arr1, phi_arr2, y_arr2):

    dphi = compute_dphi(phi_arr1, phi_arr2)

    dy = y_arr1 - y_arr2

    return np.sqrt(dphi * dphi + dy * dy)

def match_top_dR(
    file_names,
    maxDR = 0.8,
    treename_reco='reco',
    treename_truth='parton',
    plot_dir=None
    ):
    aliases_reco = {
        'th_y' : 'PseudoTop_Reco_top_had_y',
        'th_phi' : 'PseudoTop_Reco_top_had_phi',
        'tl_y' : 'PseudoTop_Reco_top_lep_y',
        'tl_phi' : 'PseudoTop_Reco_top_lep_phi'
    }

    aliases_truth = {
        'th_y' : 'MC_thad_afterFSR_y',
        'th_phi' : 'MC_thad_afterFSR_phi',
        'tl_y' : 'MC_tlep_afterFSR_y',
        'tl_phi' : 'MC_tlep_afterFSR_phi'
    }

    array_reco = uproot.concatenate(
        [f"{fname}:{treename_reco}" for fname in file_names],
        ['th_y', 'th_phi', 'tl_y', 'tl_phi'],
        aliases = aliases_reco
    ).to_numpy()

    array_truth = uproot.concatenate(
        [f"{fname}:{treename_truth}" for fname in file_names],
        ['th_y', 'th_phi', 'tl_y', 'tl_phi'],
        aliases = aliases_truth
    ).to_numpy()

    dR_th = compute_dR(array_reco['th_phi'], array_reco['th_y'], array_truth['th_phi'], array_truth['th_y'])

    dR_tl = compute_dR(array_reco['tl_phi'], array_reco['tl_y'], array_truth['tl_phi'], array_truth['tl_y'])

    passcuts = (dR_th < maxDR) & (dR_tl < maxDR)

    if plot_dir:
        # plot distributions of th_dR and tl_dR
        plotter.plot_hist(
            os.path.join(plot_dir, 'dR_th'), [dR_th],
            nbins = 50, bin_margin = 0.02,
            xlabel = "$\\Delta R(t_{had}^{reco}, t_{had}^{part})$", ylabel = "Events"
        )

        plotter.plot_hist(
            os.path.join(plot_dir, 'dR_tl'), [dR_tl],
            nbins = 50, bin_margin = 0.02,
            xlabel = "$\\Delta R(t_{lep}^{reco}, t_{lep}^{part})$", ylabel = "Events"
        )

    return passcuts

class DataHandlerROOT(DataHandlerBase):
    """
    Load data from root files

    Parameters
    ----------
    filepaths :  str or sequence of str
        List of root file names to load
    variable_names : list of str, optional
        List of reco level variable names to read. If not provided, read all
    variable_names_mc : list of str, optional
        List of truth level variable names to read. If not provided, read all
    weight_type : str, default 'nominal'
        Type of event weights to load for systematic uncertainty variations
    treename_reco : str, default 'reco'
        Name of the reconstruction-level tree
    treename_truth : str, default 'parton'
        Name of the truth-level tree. If empty or None, skip loading the 
        truth-level tree
    """
    def __init__(
        self,
        filepaths,
        variable_names=[],
        variable_names_mc=[],
        treename_reco='reco',
        treename_truth='parton',
        weight_name_nominal='normalized_weight',
        weight_type='nominal',
        matchDR = None, # float
        plot_dir = None, # str
        ):

        super().__init__()

        ######
        # load data from root files
        logger.debug("Load data array from reco trees")
        variable_names = filter_variable_names(variable_names)
        self.data_reco = load_arrays(filepaths, treename_reco, variable_names)

        if variable_names_mc:
            logger.debug("Load data array from truth trees")
            variable_names_mc = filter_variable_names(variable_names_mc)
            self.data_truth = load_arrays(filepaths, treename_truth, variable_names_mc)

        ######
        # event weights
        logger.debug("Load weight arrays")
        self.weights = load_weights(
            filepaths, treename_reco,
            weight_name = weight_name_nominal, weight_type = weight_type
        )

        if variable_names_mc:
            # event weights
            #self.weights_mc = load_weights(
            #    filepaths, treename_truth,
            #    weight_name = weight_name_nominal+'_mc', weight_type = weight_type
            #)
            self.weights_mc = self.weights.copy()

        ######
        # event selection flags
        self.pass_reco = select_reco(filepaths, treename=treename_reco)
        if variable_names_mc:
            self.pass_truth = select_parton(filepaths, treename=treename_truth)

        if matchDR is not None and self.pass_truth is not None:
            self.pass_truth &= match_top_dR(
                filepaths,
                maxDR = matchDR,
                treename_reco = treename_reco,
                treename_truth = treename_truth,
                plot_dir = plot_dir
                )

        ######
        # sanity check
        assert(len(self.data_reco)==len(self.weights))
        assert(len(self.data_reco)==len(self.pass_reco))
        if self.data_truth is not None:
            assert(len(self.data_reco)==len(self.data_truth))
            assert(len(self.data_truth)==len(self.weights_mc))
            assert(len(self.data_truth)==len(self.pass_truth))