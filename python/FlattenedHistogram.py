import os
import copy
import hist
import numpy as np

import histogramming as myhu
import plotter

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

    def __len__(self):
        return len(self._xhists)

    def __iter__(self):
        return iter(self._xhists)

    def __getitem__(self, bin_label):
        return self._xhists[bin_label]

    def __setitem__(self, bin_label, newh1d):
        self._xhists[bin_label] = newh1d

    def __add__(self, other_fh2d):
        new_fh2d = self.copy()
        new_fh2d._yhist += other_fh2d._yhist

        for ybin_label in self:
            new_fh2d[ybin_label] += other_fh2d[ybin_label]

        return new_fh2d

    def __iadd__(self, other_fh2d):
        self._yhist += other_fh2d._yhist

        for ybin_label in self:
            self[ybin_label] += other_fh2d[ybin_label]

        return self

    def reset(self):
        self._yhist.reset()

        for ybin_label in self:
            self[ybin_label].reset()

    def fill(self, xarr, yarr, weight=1.):
        # first separate data arrays into y bins
        ybin_edges = self._yhist.axes[0].edges
        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"

            ysel = (yarr >= ylow) & (yarr < yhigh)
            xarr_sel = xarr[ysel]
            warr_sel = weight[ysel] if not isinstance(weight, float) else weight

            self[ybin_label].fill(xarr_sel, weight=warr_sel)

        # also fill y hist
        self._yhist.fill(yarr, weight=weight)

    def nbins(self):
        nbins = 0

        for ybin_label in self:
            nbins += len(self[ybin_label].axes[0].edges) - 1

        return nbins

    def get_bins(self):
        bins_d = {"axis": "x_vs_y"}

        ybin_edges = self._yhist.axes[0].edges

        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"
            bins_d[ybin_label] = {
                "edge": [ylow, yhigh],
                "x_bins": list(self[ybin_label].axes[0].edges)
            }

        return bins_d

    def find_bins(self, xarr, yarr):
        bin_indices = np.zeros(len(xarr), dtype=np.int32)

        ybin_edges = self._yhist.axes[0].edges
        ibin_offset = 0
        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"

            ysel = (yarr >= ylow) & (yarr < yhigh)
            xarr_sel = xarr[ysel]

            xbin_edges = self[ybin_label].axes[0].edges
            bin_indices[ysel] = np.searchsorted(xbin_edges, xarr_sel, side='right') + ibin_offset

            ibin_offset += len(xbin_edges) - 1

        return bin_indices

    def is_underflow_or_overflow(self, xarr, yarr):
        assert(len(xarr)==len(yarr))

        ybin_edges = self._yhist.axes[0].edges
        isflow = yarr < ybin_edges[0] | yarr >= ybin_edges[-1]

        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"
            xbin_edges = self[ybin_label].axes[0].edges

            ysel = (yarr >= ylow) & (yarr < yhigh)
            isflow[ysel] |= (xarr[ysel] < xbin_edges[0] | xarr[ysel] >= xbin_edges[-1])

        return isflow

    def flatten_array(self, xarr, yarr):
        condlist = []
        funclist = []

        ybin_edges = self._yhist.axes[0].edges

        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"
            xbin_edges = self[ybin_label].axes[0].edges

            ysel = (yarr >= ylow) & (yarr < yhigh)
            condlist.append(ysel)

            funclist.append(
                lambda x, xbin_edges=xbin_edges, ylow=ylow, yhigh=yhigh: (x - xbin_edges[0]) / (xbin_edges[-1] - xbin_edges[0]) * (yhigh - ylow) + ylow
            )

        return np.piecewise(xarr, condlist, funclist)

    def flatten(self):
        hflat = None

        if not self._yhist or not self._xhists:
            return hflat

        # bin edges, contents, errors
        flat_bin_edges = []
        flat_bin_values = []
        flat_bin_variances = []

        ybin_edges = self._yhist.axes[0].edges

        for ibin, (ylow, yhigh) in enumerate(zip(ybin_edges[:-1], ybin_edges[1:])):
            ybin_label = f"y_bin{ibin+1}"
            xbin_edges = self[ybin_label].axes[0].edges

            xbin_edges_new = (xbin_edges - xbin_edges[0]) / (xbin_edges[-1] - xbin_edges[0]) * (yhigh - ylow) + ylow

            if ibin == 0:
                # first y bin, include the first lower edge
                flat_bin_edges.append(xbin_edges_new)
            else:
                # include only the upper edges
                flat_bin_edges.append(xbin_edges_new[1:])

            flat_bin_values.append(self[ybin_label].values())
            flat_bin_variances.append(self[ybin_label].variances())

        flat_bin_edges = np.concatenate(flat_bin_edges)
        flat_bin_values = np.concatenate(flat_bin_values)
        flat_bin_variances = np.concatenate(flat_bin_variances)

        # make a 1D histogram
        hflat = hist.Hist(hist.axis.Variable(flat_bin_edges), storage=hist.storage.Weight())
        hflat.view()['value'] = flat_bin_values
        hflat.view()['variance'] = flat_bin_variances

        return hflat

    def fromFlat(self, h_flat):
        flat_bin_values = h_flat.values()
        flat_bin_variances = h_flat.variances()

        y_bin_values = []
        y_bin_variances = []

        bin_offset = 0

        for ybin in range(1, len(self)+1):
            ybin_label = f"y_bin{ybin}"

            xbin_edges = self[ybin_label].axes[0].edges
            nbins_x = len(xbin_edges) - 1

            sub_values = flat_bin_values[bin_offset : bin_offset + nbins_x]
            sub_variances = flat_bin_variances[bin_offset : bin_offset + nbins_x]

            self[ybin_label].view()['value'] = sub_values
            self[ybin_label].view()['variance'] = sub_variances

            y_bin_values.append(np.sum(sub_values))
            y_bin_variances.append(np.sum(sub_variances))

            bin_offset += nbins_x

        self._yhist.view()['value'] = y_bin_values
        self._yhist.view()['variance'] = y_bin_variances

    def scale(self, factor):
        self._yhist *= factor

        for ybin_label in self:
            self[ybin_label] *= factor

        return self

    def multiply(self, other_fh2d):
        self._yhist = myhu.multiply(self._yhist, other_fh2d._yhist)

        for ybin_label in self:
            self[ybin_label] = myhu.multiply(self[ybin_label], other_fh2d[ybin_label])

        return self

    def divide(self, other_fh2d):
        self._yhist = myhu.divide(self._yhist, other_fh2d._yhist)

        for ybin_label in self:
            self[ybin_label] = myhu.divide(self[ybin_label], other_fh2d[ybin_label])

        return self

    def norm(self, flow=True, density=False):
        if flow:
            return myhu.get_hist_norm(self._yhist, density=density, flow=True)
        else:
            # get sum from every y bin slice
            n = 0.
            for ybin_label in self:
                n += myhu.get_hist_norm(self[ybin_label], density=density, flow=False)
            return n

    def renormalize(self, norm=1., flow=True, density=False):
        old_norm = self.norm(flow=flow, density=density)

        self._yhist *= norm / old_norm

        for ybin_label in self:
            self[ybin_label] *= norm / old_norm

        return self

    def make_density(self, outer_bin_width=1.):
        ybin_widths = myhu.get_hist_widths(self._yhist) * outer_bin_width
        self._yhist /= ybin_widths

        for ybin_label, ybin_w in zip(self, ybin_widths):
            xbin_widths = myhu.get_hist_widths(self[ybin_label])
            self[ybin_label] /= (xbin_widths * ybin_w)

    def write(self, f_write, directory):
        f_write[os.path.join(directory, '_yhist')] = self._yhist
        for ybin_label in self:
            f_write[os.path.join(directory, ybin_label)] = self[ybin_label]

    def copy(self):
        return copy.deepcopy(self)

    def set_xlabel(self, label):
        for ybin in self:
            self[ybin].axes[0].label = label

    def get_xlabel(self, ybin='y_bin1'):
        return self[ybin].axes[0].label

    def set_ylabel(self, label):
        if self._yhist is None:
            logger.warn(f"No y hist for setting label")
        else:
            self._yhist.axes[0].label = label

    def get_ylabel(self):
        return self._yhist.axes[0].label

    def plot(
        self,
        figname,
        markers = None,
        colors = None,
        rescales_order_of_magnitude = None,
        **plot_args
        ):

        hists_to_plot = [self[ybin] for ybin in self]

        ybin_edges = self._yhist.axes[0].edges
        yname = self._yhist.axes[0].label
        legends = [f"{ylow}$\leq${yname}$<${yhigh}" for ylow, yhigh in zip(ybin_edges[:-1], ybin_edges[1:])]

        if rescales_order_of_magnitude is not None:
            assert(len(rescales_order_of_magnitude)==len(hists_to_plot))
            for i, oom in enumerate(rescales_order_of_magnitude):
                hists_to_plot[i] = hists_to_plot[i] * 10**oom
                legends[i] = f"($\times 10^{{ {oom} }}$), " + legends[i]

        if colors is None:
            colors = plotter.get_default_colors(len(hists_to_plot))
        else:
            assert(len(colors)>=len(hists_to_plot))
            colors = colors[:len(hists_to_plot)]

        if markers is None:
            markers = ['o'] * len(hists_to_plot)
        else:
            assert(len(markers)>=len(hists_to_plot))
            markers = markers[:len(hists_to_plot)]

        draw_options = []
        for i in range(len(hists_to_plot)):
            draw_options.append(
                {'color': colors[i],
                 'label': legends[i],
                 'marker': markers[i],
                 'markersize': 3,
                 'histtype': 'errorbar',
                 'xerr': True
                }
            )

        plotter.plot_histograms_and_ratios(
            figname,
            hists_numerator = hists_to_plot,
            draw_options_numerator = draw_options,
            xlabel = hists_to_plot[0].axes[0].label,
            stamp_opt = {'fontsize': 9.},
            **plot_args
        )

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
            h_inst.renormalize(norm, flow=True, density=False)

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

    @staticmethod
    def convert_in_dict(hists_dict):
        for k in hists_dict:
            if not isinstance(hists_dict[k], dict):
                continue

            if '_yhist' in hists_dict[k]:
                # convert to an FlattenedHistogram2D object
                hists_dict[k] = FlattenedHistogram2D.from_dict(hists_dict[k])
            else:
                # keep looking
                FlattenedHistogram2D.convert_in_dict(hists_dict[k])

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

    def __len__(self):
        return len(self._xyhists)

    def __iter__(self):
        return iter(self._xyhists)

    def __getitem__(self, bin_label):
        return self._xyhists[bin_label]

    def __setitem__(self, bin_label, newfh2d):
        self._xyhists[bin_label] = newfh2d

    def __add__(self, other_fh3d):
        new_fh3d = self.copy()
        new_fh3d._zhist += other_fh3d._zhist

        for zbin_label in new_fh3d:
            new_fh3d[zbin_label] += other_fh3d[zbin_label]

        return new_fh3d

    def __iadd__(self, other_fh3d):
        self._zhist += other_fh3d._zhist

        for zbin_label in self:
            self[zbin_label] += other_fh3d[zbin_label]

        return self

    def reset(self):
        self._zhist.reset()

        for zbin_label in self:
            self[zbin_label].reset()

    def fill(self, xarr, yarr, zarr, weight=1.):
        # first separate data arrays into z bins
        zbin_edges = self._zhist.axes[0].edges
        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            zsel = (zarr >= zlow) & (zarr < zhigh)

            xarr_sel = xarr[zsel]
            yarr_sel = yarr[zsel]
            warr_sel = weight[zsel] if not isinstance(weight, float) else weight

            self[zbin_label].fill(xarr_sel, yarr_sel, weight=warr_sel)

        # also fill z hist
        self._zhist.fill(zarr, weight=weight)

    def nbins(self):
        nbins = 0

        for zbin_label in self:
            nbins += self[zbin_label].nbins()

        return nbins

    def get_bins(self):
        bins_d = {"axis": "x_vs_y_vs_z"}

        zbin_edges = self._zhist.axes[0].edges

        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            bins_d[zbin_label] = {"edge": [zlow, zhigh]}
            bins_d[zbin_label].update(
                self[zbin_label].get_bins()
            )

        return bins_d

    def find_bins(self, xarr, yarr, zarr):
        bin_indices = np.zeros(len(xarr), dtype=np.int32)

        zbin_edges = self._zhist.axes[0].edges
        ibin_offset = 0
        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            zsel = (zarr >= zlow) & (zarr < zhigh)
            xarr_sel = xarr[zsel]
            yarr_sel = yarr[zsel]

            bin_indices[zsel] = self[zbin_label].find_bins(xarr_sel, yarr_sel) + ibin_offset

            ibin_offset += self[zbin_label].nbins()

        return bin_indices

    def is_underflow_or_overflow(self, xarr, yarr, zarr):
        assert(len(xarr)==len(yarr))
        assert(len(xarr)==len(zarr))

        zbin_edges = self._zhist.axes[0].edges
        isflow = zarr < zbin_edges[0] | zarr >= zbin_edges[-1]

        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            zsel = (zarr >= zlow) & (zarr < zhigh)

            isflow[zsel] |= self[zbin_label].is_underflow_or_overflow(xarr[zsel], yarr[zsel])

        return isflow

    def flatten_array(self, xarr, yarr, zarr):
        condlist = []
        funclist = []

        zbin_edges = self._zhist.axes[0].edges

        xyarr = np.empty_like(xarr)

        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            zsel = (zarr >= zlow) & (zarr < zhigh)
            condlist.append(zsel)

            h_xy_flat = self[zbin_label].flatten()
            xybin_edges = h_xy_flat.axes[0].edges

            funclist.append(
                lambda xy, xybin_edges=xybin_edges, zlow=zlow, zhigh=zhigh: (xy - xybin_edges[0]) / (xybin_edges[-1] - xybin_edges[0]) * (zhigh - zlow) + zlow 
            )

            xyarr[zsel] = self[zbin_label].flatten_array( xarr[zsel], yarr[zsel])

        return np.piecewise(xyarr, condlist, funclist)

    def flatten(self):
        hflat = None

        if not self._zhist or not self._xyhists:
            return hflat

        # bin edges, contents, errors
        flat_bin_edges = []
        flat_bin_values = []
        flat_bin_variances = []

        zbin_edges = self._zhist.axes[0].edges

        for ibin, (zlow, zhigh) in enumerate(zip(zbin_edges[:-1], zbin_edges[1:])):
            zbin_label = f"z_bin{ibin+1}"

            h_xy_flat = self[zbin_label].flatten()

            xybin_edges = h_xy_flat.axes[0].edges

            xybin_edges_new = (xybin_edges - xybin_edges[0]) / (xybin_edges[-1] - xybin_edges[0]) * (zhigh - zlow) + zlow

            if ibin == 0:
                # first z bin, include the first lower edge
                flat_bin_edges.append(xybin_edges_new)
            else:
                # include only the upper edges
                flat_bin_edges.append(xybin_edges_new[1:])

            flat_bin_values.append(h_xy_flat.values())
            flat_bin_variances.append(h_xy_flat.variances())

        flat_bin_edges = np.concatenate(flat_bin_edges)
        flat_bin_values = np.concatenate(flat_bin_values)
        flat_bin_variances = np.concatenate(flat_bin_variances)

        # make a 1D histogram
        hflat = hist.Hist(hist.axis.Variable(flat_bin_edges), storage=hist.storage.Weight())
        hflat.view()['value'] = flat_bin_values
        hflat.view()['variance'] = flat_bin_variances

        return hflat

    def fromFlat(self, h_flat):
        flat_bin_values = h_flat.values()
        flat_bin_variances = h_flat.variances()

        z_bin_values = []
        z_bin_variances = []

        bin_offset = 0

        for zbin in range(1, len(self)+1):
            zbin_label = f"z_bin{zbin}"

            y_bin_values = []
            y_bin_variances = []

            for ybin in range(1, len(self[zbin_label])+1):
                ybin_label = f"y_bin{ybin}"

                xbin_edges = self[zbin_label][ybin_label].axes[0].edges
                nbins_x = len(xbin_edges) - 1

                sub_values = flat_bin_values[bin_offset : bin_offset + nbins_x]
                sub_variances = flat_bin_variances[bin_offset : bin_offset + nbins_x]

                self[zbin_label][ybin_label].view()['value'] = sub_values
                self[zbin_label][ybin_label].view()['variance'] = sub_variances

                y_bin_values.append(np.sum(sub_values))
                y_bin_variances.append(np.sum(sub_variances))

                bin_offset += nbins_x

            self[zbin_label]._yhist.view()['value'] = y_bin_values
            self[zbin_label]._yhist.view()['variance'] = y_bin_variances

            z_bin_values.append(np.sum(y_bin_values))
            z_bin_variances.append(np.sum(y_bin_variances))

        self._zhist.view()['value'] = z_bin_values
        self._zhist.view()['variance'] = z_bin_variances

    def scale(self, factor):
        self._zhist *= factor

        for zbin_label in self:
            self[zbin_label].scale(factor)

        return self

    def multiply(self, other_fh3d):
        self._zhist = myhu.multiply(self._zhist, other_fh3d._zhist)

        for zbin_label in self:
            self[zbin_label] = self[zbin_label].multiply(other_fh3d[zbin_label])

        return self

    def divide(self, other_fh3d):
        self._zhist = myhu.divide(self._zhist, other_fh3d._zhist)

        for zbin_label in self:
            self[zbin_label].divide(other_fh3d[zbin_label])

        return self

    def norm(self, flow=True, density=False):
        if flow:
            return myhu.get_hist_norm(self._zhist, density=density, flow=True)
        else:
            # get sum from every z bin slice
            n = 0.
            for zbin_label in self:
                n += self[zbin_label].norm(flow=False, density=density)
            return n

    def renormalize(self, norm=1., flow=True, density=False):
        old_norm = self.norm(flow=flow, density=density)

        self._zhist *= norm / old_norm

        for zbin_label in self:
            norm_zbin = self[zbin_label].norm(flow=flow, density=density)
            norm_zbin = norm_zbin * norm / old_norm
            self[zbin_label].renormalize(norm_zbin, flow=flow, density=density)

        return self

    def make_density(self, outer_bin_width=1.):
        zbin_widths = myhu.get_hist_widths(self._zhist) * outer_bin_width
        self._zhist /= zbin_widths

        for zbin_label, zbin_w in zip(self, zbin_widths):
            self[zbin_label].make_density(outer_bin_width=zbin_w)

    def set_xlabel(self, label):
        for zbin in self:
            self[zbin].set_xlabel(label)

    def get_xlabel(self, zbin="z_bin1", ybin="y_bin1"):
        return self[zbin].get_xlabel(ybin)

    def set_ylabel(self, label):
        for zbin in self:
            self[zbin].set_ylabel(label)

    def get_ylabel(self, zbin="z_bin1"):
        return self[zbin].get_ylabel()

    def set_zlabel(self, label):
        if self._zhist is None:
            logger.warn(f"No z hist for setting label")
        else:
            self._zhist.axes[0].label = label

    def get_zlabel(self):
        return self._zhist.axes[0].label

    def write(self, f_write, directory):
        f_write[os.path.join(directory, '_zhist')] = self._zhist
        for zbin_label in self:
            self[zbin_label].write(f_write, os.path.join(directory, zbin_label))

    def copy(self):
        return copy.deepcopy(self)

    def plot(
        self,
        figname_prefix,
        markers = None,
        colors = None,
        rescales_order_of_magnitude = None,
        **plot_args
        ):

        # z bin edges
        zbin_edges = self._zhist.axes[0].edges
        zname = self._zhist.axes[0].label

        # stamps
        stamps = [f"{zlow}$\leq${zname}$<${zhigh}" for zlow, zhigh in zip(zbin_edges[:-1], zbin_edges[1:])]

        # rescale
        if rescales_order_of_magnitude is None:
            rescales_order_of_magnitude = [None] * len(self)
        else:
            assert(len(rescales_order_of_magnitude)==len(self))

        for z, zbin in enumerate(self):
            figname = figname_prefix+'_'+zbin

            # update stamp
            plot_args_z = copy.deepcopy(plot_args)
            if 'stamp_texts' in plot_args_z:
                plot_args_z['stamp_texts'].append(stamps[z])
            else:
                plot_args_z['stamp_texts'] = [stamps[z]]

            self[zbin].plot(
                figname,
                markers = markers,
                colors = colors,
                rescales_order_of_magnitude = rescales_order_of_magnitude[z],
                **plot_args_z
            )

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
            h_inst.renormalize(norm, flow=True, density=False)

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

    @staticmethod
    def convert_in_dict(hists_dict):
        for k in hists_dict:
            if not isinstance(hists_dict[k], dict):
                continue

            if '_zhist' in hists_dict[k]:
                # convert to an FlattenedHistogram3D object
                hists_dict[k] = FlattenedHistogram3D.from_dict(hists_dict[k])
            else:
                # keep looking
                FlattenedHistogram3D.convert_in_dict(hists_dict[k])

class FlattenedResponse():
    def __init__(self, flattenedHist_reco, flattenedHist_truth):
        # for binning
        self._fh_reco = flattenedHist_reco.copy()
        self._fh_truth = flattenedHist_truth.copy()

        # get flattened bins
        bins_reco = self._fh_reco.flatten().axes[0].edges
        bins_truth = self._fh_truth.flatten().axes[0].edges

        self._resp = hist.Hist(hist.axis.Variable(bins_reco), hist.axis.Variable(bins_truth), storage=hist.storage.Weight())

    def get(self):
        return self._resp

    def fill(self, arrs_reco, arrs_truth, weight=None):

        arrs_reco_flat = self._fh_reco.flatten_array(*arrs_reco)
        arrs_truth_flat = self._fh_truth.flatten_array(*arrs_truth)

        if isinstance(weight, float):
            weight = [weight] * len(arrs_reco_flat)

        self._resp.fill(arrs_reco_flat, arrs_truth_flat, weight=weight)

        self._fh_reco.fill(*arrs_reco, weight=weight)
        self._fh_truth.fill(*arrs_truth, weight=weight)

    def projectToReco(self, flow=True):
        fh_proj_reco = self._fh_reco.copy()

        fh_proj_reco.fromFlat(myhu.projectToXaxis(self._resp, flow=flow))

        return fh_proj_reco

    def projectToTruth(self, flow=True):
        fh_proj_truth = self._fh_truth.copy()

        fh_proj_truth.fromFlat(myhu.projectToYaxis(self._resp, flow=flow))

        return fh_proj_truth

    def normalize_truth_bins(self):
        # normalize echo truth bin to sum of one
        resp_normed = np.zeros_like(self._resp.values())

        np.divide(self._resp.values(), self._resp.values().sum(axis=0), out=resp_normed, where=self._resp.values().sum(axis=0)!=0)

        self._resp.view()['value'] = resp_normed

    def write(self, f_write, directory):
        f_write[os.path.join(directory, '_resp')] =  self._resp
        self._fh_reco.write(f_write, os.path.join(directory, '_fh_reco'))
        self._fh_truth.write(f_write, os.path.join(directory, '_fh_truth'))

    @classmethod
    def from_dict(cls, hists_d):
        h_inst = cls()

        h_inst._fh_reco = hists_d.get('_fh_reco')
        h_inst._fh_truth = hists_d.get('_fh_truth')
        h_inst._resp = hists_d.get('_resp')

        return h_inst

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