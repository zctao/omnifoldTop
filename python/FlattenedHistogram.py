import os
import copy
import hist

import histogramming as myhu

import logging
logger = logging.getLogger('FlattenedHistogram')
logger.setLevel(logging.INFO)

class FlattenedHistogram2D():
    def __init__(
        self,
        binning_d={}, # dict, binning dictionary
        varname_x='',
        varname_y=''
        ):

        self._xhists = {}

        ybin_edges = []
        for ybin_label in binning_d:
            if not ybin_label.startswith("y_bin"):
                continue

            # y bins
            ylow, yhigh = binning_d[ybin_label]["edge"]
            if not ybin_edges:
                ybin_edges.append(ylow)
            else:
                assert(ylow==ybin_edges[-1])
            ybin_edges.append(yhigh)

            # make x histogram
            xbin_edges = binning_d[ybin_label]['x_bins']
            self._xhists[ybin_label] = hist.Hist(hist.axis.Variable(xbin_edges), storage=hist.storage.Weight())

            if varname_x:
                self._xhists[ybin_label].axes[0].label = varname_x

        # make a y histogram
        self._yhist = None
        if ybin_edges:
            self._yhist = hist.Hist(hist.axis.Variable(ybin_edges), storage=hist.storage.Weight())

            if varname_y:
                self._yhist.axes[0].label = varname_y

        if binning_d and (not self._xhists or not self._yhist):
            logger.error("Something is wrong with the binning config:")
            logger.error(f"{binning_d}")
            raise RuntimeError("Fail to initialize FlattenedHistogram2D")

    def fill(self, xarr, yarr, weight=1.):
        # first separate data arrays into y bins
        ybin_edges = self._yhist.axes[0].edges
        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"

            ysel = (yarr >= ylow) & (yarr < yhigh)
            xarr_sel = xarr[ysel]
            warr_sel = weight[ysel] if not isinstance(weight, float) else weight

            self._xhists[ybin_label].fill(xarr_sel, weight=warr_sel)

        # also fill y hist
        self._yhist.fill(yarr, weight=weight)

    def norm(self, flow=True, density=False):
        if flow:
            return myhu.get_hist_norm(self._yhist, density=density, flow=True)
        else:
            # get sum from every y bin slice
            n = 0.
            for ybin_label in self._xhists:
                n += myhu.get_hist_norm(self._xhists[ybin_label], density=density, flow=False)
            return n

    def rescale(self, norm=1., flow=True, density=False):
        old_norm = self.norm(flow=flow, density=density)

        self._yhist *= norm / old_norm

        for ybin_label in self._xhists:
            self._xhists[ybin_label] *= norm / old_norm

    def make_density(self):
        self._yhist /= myhu.get_hist_widths(self._yhist)

        for ybin_label in self._xhists:
            bwidths = myhu.get_hist_widths(self._xhists[ybin_label])
            self._xhists[ybin_label] /= bwidths

    def write(self, f_write, directory):
        f_write[os.path.join(directory, '_yhist')] = self._yhist
        for ybin_label in self._xhists:
            f_write[os.path.join(directory, ybin_label)] = self._xhists[ybin_label]

    def copy(self):
        return copy.deepcopy(self)

    def set_xlabel(self, label):
        for ybin in self._xhists:
            self._xhists[ybin].axes[0].label = label

    def set_ylabel(self, label):
        if self._yhist is None:
            logger.warn(f"No y hist for setting label")
        else:
            self._yhist.axes[0].label = label

    def plot(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, hists_d):
        h_inst = cls()

        for k in hists_d:
            if k == '_yhist':
                h_inst._yhist = hists_d[k]
            elif k.startswith('y_bin'):
                h_inst._xhists[k] = hists_d[k]

        return h_inst

    @classmethod
    def calc_hists(
        cls,
        xarr,
        yarr,
        binning_d,
        weights=1.,
        varname_x='',
        varname_y='',
        norm=None,
        density=False
        ):

        h_inst = cls(binning_d, varname_x, varname_y)
        h_inst.fill(xarr, yarr, weight=weights)

        if norm is not None:
            h_inst.rescale(norm, flow=True, density=False)

        if density:
            h_inst.make_density()

        return h_inst

    @staticmethod
    def average(histograms_list):
        if not histograms_list:
            return None

        elif len(histograms_list) == 1:
            return histograms_list[0]

        else:
            h_average = histograms_list[0].copy()

            h_average._yhist = myhu.average_histograms([h._yhist for h in histograms_list])

            for ybin_label in h_average._xhists:
                h_average._xhists[ybin_label] = myhu.average_histograms([h._xhists[ybin_label] for h in histograms_list])

            return h_average

class FlattenedHistogram3D():
    def __init__(
        self,
        binning_d={}, # dict, binning dictionary
        varname_x='',
        varname_y='',
        varname_z=''
        ):

        self._xyhists = {}

        zbin_edges = []

        for zbin_label in binning_d:
            if not zbin_label.startswith("z_bin"):
                continue

            # z bins
            zlow, zhigh = binning_d[zbin_label]["edge"]
            if not zbin_edges:
                zbin_edges.append(zlow)
            else:
                assert(zlow==zbin_edges[-1])
            zbin_edges.append(zhigh)

            binning_xy_d = binning_d[zbin_label]
            self._xyhists[zbin_label] = FlattenedHistogram2D(binning_xy_d, varname_x, varname_y)

        # make a z histogram
        self._zhist = None
        if zbin_edges:
            self._zhist = hist.Hist(hist.axis.Variable(zbin_edges), storage=hist.storage.Weight())

            if varname_z:
                self._zhist.axes[0].label = varname_z

        if binning_d and (not self._zhist or not self._xyhists):
            logger.error("Something is wrong with the binning config:")
            logger.error(f"{binning_d}")
            raise RuntimeError("Fail to initialize FlattenedHistogram3D")

    def fill(self, xarr, yarr, zarr, weight=1.):
        # first separate data arrays into z bins
        zbin_edges = self._zhist.axes[0].edges
        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            zsel = (zarr >= zlow) & (zarr < zhigh)

            xarr_sel = xarr[zsel]
            yarr_sel = yarr[zsel]
            warr_sel = weight[zsel] if not isinstance(weight, float) else weight

            self._xyhists[zbin_label].fill(xarr_sel, yarr_sel, weight=warr_sel)

        # also fill z hist
        self._zhist.fill(zarr, weight=weight)

    def norm(self, flow=True, density=False):
        if flow:
            return myhu.get_hist_norm(self._zhist, density=density, flow=True)
        else:
            # get sum from every z bin slice
            n = 0.
            for zbin_label in self._xyhists:
                n += self._xyhists[zbin_label].norm(flow=False, density=density)
            return n

    def rescale(self, norm=1., flow=True, density=False):
        old_norm = self.norm(flow=flow, density=density)

        self._zhist *= norm / old_norm

        for zbin_label in self._xyhists:
            norm_zbin = self._xyhists[zbin_label].norm(flow=flow, density=density)
            norm_zbin = norm_zbin * norm / old_norm
            self._xyhists[zbin_label].rescale(norm_zbin, flow=flow, density=density)

    def make_density(self):
        self._zhist /= myhu.get_hist_widths(self._zhist)

        for zbin_label in self._xyhists:
            self._xyhists[zbin_label].make_density()

    def set_xlabel(self, label):
        for zbin in self._xyhists:
            self._xyhists[zbin].set_xlabel(label)

    def set_ylabel(self, label):
        for zbin in self._xyhists:
            self._xyhists[zbin].set_ylabel(label)

    def set_zlabel(self, label):
        if self._zhist is None:
            logger.warn(f"No z hist for setting label")
        else:
            self._zhist.axes[0].label = label

    def write(self, f_write, directory):
        f_write[os.path.join(directory, '_zhist')] = self._zhist
        for zbin_label in self._xyhists:
            self._xyhists[zbin_label].write(f_write, os.path.join(directory, zbin_label))

    def copy(self):
        return copy.deepcopy(self)

    def plot(self):
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, hists_d):
        h_inst = cls()

        for k in hists_d:
            if k == '_zhist':
                h_inst._zhist = hists_d[k]
            elif k.startswith('z_bin'):
                h_inst._xyhists[k] = FlattenedHistogram2D.from_dict(hists_d[k])

        return h_inst

    @classmethod
    def calc_hists(
        cls,
        xarr,
        yarr,
        zarr,
        binning_d,
        weights=1.,
        varname_x='',
        varname_y='',
        varname_z='',
        norm=None,
        density=False
        ):

        h_inst = cls(binning_d, varname_x, varname_y, varname_z)
        h_inst.fill(xarr, yarr, zarr, weight=weights)

        if norm is not None:
            h_inst.rescale(norm, flow=True, density=False)

        if density:
            h_inst.make_density()

        return h_inst

    @staticmethod
    def average(histograms_list):
        if not histograms_list:
            return None

        elif len(histograms_list) == 1:
            return histograms_list[0]
        
        else:
            h_average = histograms_list[0].copy()

            h_average._zhist = myhu.average_histograms([h._zhist for h in histograms_list])

            for zbin_label in h_average._xyhists:
                h_average._xyhists[zbin_label] = FlattenedHistogram2D.average([h._xyhists[zbin_label] for h in histograms_list])

            return h_average

def average_histograms(histograms_list):
    if not histograms_list:
        return None
    elif len(histograms_list) == 1:
        return histograms_list[0]
    elif all([isinstance(h, FlattenedHistogram2D) for h in histograms_list]):
        return FlattenedHistogram2D.average(histograms_list)
    elif all([isinstance(h, FlattenedHistogram3D) for h in histograms_list]):
        return FlattenedHistogram3D.average(histograms_list)
    else:
        raise RuntimeError("Don't know how to average")
