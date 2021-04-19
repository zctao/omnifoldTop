import numpy as np
import pandas as pd
from util import parse_input_name, normalize_histogram
# for now
import external.OmniFold.modplot as modplot

def load_dataset(file_names, array_name='arr_0', allow_pickle=True, encoding='bytes', weight_columns=[]):
    """
    Load and return a structured numpy array from a list of npz files
    """
    data = None
    if not isinstance(file_names, list):
        file_names = [file_names]
    if not isinstance(weight_columns, list):
        weight_columns = [weight_columns]

    for fname in file_names:
        fn, rwfactor = parse_input_name(fname)

        npzfile = np.load(fn, allow_pickle=allow_pickle, encoding=encoding)
        di = npzfile[array_name]
        if len(di)==0:
            raise RuntimeError('There is no events in input file {}'.format(fname))

        # rescale total event weights for this input file
        if rwfactor != 1.:
            for wname in weight_columns:
                try:
                    di[wname] *= rwfactor
                except ValueError:
                    print('Unknown field name {}'.format(wname))
                    continue

        if data is None:
            data = di
        else:
            data  = np.concatenate([data, di])
        npzfile.close()

    return data

class DataHandler(object):
    def __init__(self, filepaths, wname='w', truth_known=True,
                 variable_names=None, vars_dict={}, array_name='arr_0'):
        self.weight_name = wname # name of event weights
        self.truth_known = truth_known

        # load data from npz files to numpy array
        tmpDataArr = load_dataset(filepaths, array_name=array_name, weight_columns=wname)
        assert(tmpDataArr is not None)

        if not variable_names:
            # if no variable name list is provided, read everything
            variable_names = list(tmpDataArr.dtype.names)
        else:
            # add wname to the list
            if wname and not wname in variable_names:
                variable_names.append(wname)

            variable_names = self._filter_variable_names(variable_names)

            # check all variable names are available
            for vname in variable_names:
                if not vname in tmpDataArr.dtype.names:
                    raise RuntimeError("Unknown variable name {}".format(vname))

        # convert all fields to float
        dtypes = [(vname, vars_dict.get('vtype','float')) for vname in variable_names]
        self.data = np.array(tmpDataArr[variable_names], dtype=dtypes)

        # sum of event weights
        self.sumw = len(self.data)
        if wname and wname in self.data.dtype.names:
            self.sumw = self.data[wname].sum()

    def get_nevents(self):
        return len(self.data)

    def get_variable_arr(self, variable):
        # return a view (NOT copy) of self.data if possible
        # otherwise, try to make a new array from self.data
        # the output shape is (len(self.data), )

        if variable in self.data.dtype.names:
            return self.data[variable]
        # special cases
        elif '_px' in variable:
            var_pt = variable.replace('_px', '_pt')
            var_phi = variable.replace('_px', '_phi')
            arr_pt = self.get_variable_arr(var_pt)
            arr_phi = self.get_variable_arr(var_phi)
            return arr_pt * np.cos(arr_phi)
        elif '_py' in variable:
            var_pt = variable.replace('_py', '_pt')
            var_phi = variable.replace('_py', '_phi')
            arr_pt = self.get_variable_arr(var_pt)
            arr_phi = self.get_variable_arr(var_phi)
            return arr_pt * np.sin(arr_phi)
        elif '_pz' in variable:
            var_pt = variable.replace('_pz', '_pt')
            var_eta = variable.replace('_pz', '_eta')
            arr_pt = self.get_variable_arr(var_pt)
            arr_eta = self.get_variable_arr(var_eta)
            return arr_pt * np.sinh(arr_eta)
        else:
            raise RuntimeError("Unknown variable {}. \nAvailable variable names: {}".format(variable, self.data.dtype.names))

    def get_weights(self, unweighted=False, bootstrap=False, normalize=False, rw_type=None, vars_dict={}):
        if unweighted or not self.weight_name:
            weights = np.ones(len(self.data))
        else:
            # always return a copy of the original weight array in self.data
            weights = self.get_variable_arr(self.weight_name).copy()
            assert(weights.base is None)

        # reweight sample if needed
        if rw_type is not None:
            weights *= self._reweight_sample(rw_type, vars_dict)

        # normalize to len(self.data)
        if normalize:
            weights /= np.mean(weights)

        if bootstrap:
            weights *= np.random.poisson(1, size=len(weights))

        return weights

    def get_features_array(self, features):
        ndim_features = np.asarray(features).ndim
        if ndim_features == 1:
            # ndarray of shape (n_events, n_features)
            X = np.vstack([self.get_variable_arr(varname) for varname in features]).T
            return X
        else:
            # ndarray of shape (n_events, <feature shape>)
            X = np.stack([self.get_features_array(varnames) for varnames in features], axis=1)
            return X

    def get_dataset(self, features, label, standardize=False):
        """ features: a list of variable names
            label: int for class label
            Return:
                X: numpy array of the shape (n_events, n_features)
                Y: numpy array for event label of the shape (n_events,)
        """
        X = self.get_features_array(features)

        if standardize:
            Xmean = np.mean(X, axis=0)
            Xstd = np.std(X, axis=0)
            X -= Xmean
            X /= Xstd

        # label
        Y = np.full(len(X), label)

        return X, Y

    def get_correlations(self, variables):
        df = pd.DataFrame({var:self.get_variable_arr(var) for var in variables}, columns=variables)
        correlations = df.corr()
        return correlations

    def get_histogram(self, variable, weights, bin_edges, normalize=False):
        """
        If weights is a 1D array of the same length as the variable array, return a histogram and its error
        If weights is a 2D array or a list of 1D array, return an array of histograms and an array of their errors
        """
        if isinstance(weights, np.ndarray):
            if weights.ndim == 1: # if weights is a 1D array
                varr = self.get_variable_arr(variable)
                # check the weight array length is the same as the variable array
                assert(len(varr) == len(weights))
                hist, hist_err = modplot.calc_hist(varr, weights=weights, bins=bin_edges, density=False)[:2]
                if normalize:
                    normalize_histogram(bin_edges, hist, hist_err)
                return hist, hist_err
            elif weights.ndim == 2: # make the 2D array into a list of 1D array
                return self.get_histogram(variable, list(weights), bin_edges, normalize)
            else:
                raise RuntimeError("Only 1D or 2D array or a list of 1D array of weights can be processed.")
        elif isinstance(weights, list): # if weights is a list of 1D array
            hists, hists_err = [], []
            for w in weights:
                h, herr = self.get_histogram(variable, w, bin_edges, normalize)
                hists.append(h)
                hists_err.append(herr)
            return np.asarray(hists), np.asarray(hists_err)
        else:
            raise RuntimeError("Unknown type of weights: {}".format(type(weights)))

    def _reweight_sample(self, rw_type, vars_dict):
        if not rw_type:
            return 1.
        elif rw_type == 'linear_th_pt':
            # truth-level hadronic top pt
            assert('th_pt' in vars_dict)
            assert(self.truth_known)
            varname_thpt = vars_dict['th_pt']['branch_mc']
            th_pt = self.get_variable_arr(varname_thpt)
            # reweight factor
            rw = 1. + 1/800.*th_pt
            return rw
        elif rw_type == 'gaussian_bump':
            # truth-level ttbar mass
            assert('mtt' in vars_dict)
            assert(self.truth_known)
            varname_mtt = vars_dict['mtt']['branch_mc']
            mtt = self.get_variable_arr(varname_mtt)
            #reweight factor
            k = 0.5
            m0 = 800.
            sigma = 100.
            rw = 1. + k*np.exp( -( (mtt-m0)/sigma )**2 )
            return rw
        elif rw_type == 'gaussian_tail':
            assert('mtt' in vars_dict)
            assert(self.truth_known)
            varname_mtt = vars_dict['mtt']['branch_mc']
            mtt = self.get_variable_arr(varname_mtt)
            #reweight factor
            k = 0.5
            m0 = 2000.
            sigma = 1000.
            rw = 1. + k*np.exp( -( (mtt-m0)/sigma )**2 )
            return rw
        else:
            raise RuntimeError("Unknown reweighting type: {}".format(rw_type))

    def _filter_variable_names(self, variable_names):
        varnames_skimmed = set()

        for vname in variable_names:
            if '_px' in vname:
                vname_pt = vname.replace('_px', '_pt')
                vname_phi = vname.replace('_px', '_phi')
                varnames_skimmed.add(vname_pt)
                varnames_skimmed.add(vname_phi)
            elif '_py' in vname:
                vname_pt = vname.replace('_py', '_pt')
                vname_phi = vname.replace('_py', '_phi')
                varnames_skimmed.add(vname_pt)
                varnames_skimmed.add(vname_phi)
            elif '_pz' in vname:
                vname_pt = vname.replace('_pz', '_pt')
                vname_eta = vname.replace('_pz', '_eta')
                varnames_skimmed.add(vname_pt)
                varnames_skimmed.add(vname_eta)
            else:
                varnames_skimmed.add(vname)

        return list(varnames_skimmed)

# Toy data
class DataToy(DataHandler):
    def __init__(self, nevents, mu=0., sigma=1.):
        self.weight_name = ''
        self.truth_known = True

        # generate toy data
        # truth level
        var_truth = np.random.normal(mu, sigma,nevents)

        # detector smearing
        epsilon = 1. # smearing width
        var_reco = np.array([(x + np.random.normal(0, epsilon)) for x in var_truth])

        self.data = np.array([(x,y) for x,y in zip(var_reco, var_truth)],
                             dtype=[('x_reco','float'), ('x_truth','float')])
