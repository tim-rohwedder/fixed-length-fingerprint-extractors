# -*- coding: utf-8 -*-
#
# This script is based on:
#
# A. Nautsch, D. Meuwly, D. Ramos, J. Lindh, and C. Busch:
# Making Likelihood Ratios digestible for Cross-Application Performance Assessment,
# IEEE Signal Processing Letters, 24(10), pp. 1552-1556, Oct. 2017.
#          see: https://codeocean.com/2017/09/29/verbal-detection-error-tradeoff-lpar-det-rpar/metadata
#          license: HDA-OPEN-RESEARCH
#
# The code (in the source above) is based on the sidekit implementation of the bosaris toolkit
# sidekit: http://www-lium.univ-lemans.fr/sidekit
#          license: LGPL
# bosaris: https://sites.google.com/site/bosaristoolkit
#          license: https://sites.google.com/site/bosaristoolkit/home/License.txt
#
# Copyright 2018 Andreas Nautsch, Hochschule Darmstadt
#           2019 Andreas Nautsch, EURECOM


import numpy
import copy
from scipy.special import erfinv
from collections import namedtuple
import matplotlib.pyplot as mpl
from tikzplotlib import save as tikz_save
import logging


__license__ = "HDA-OPEN-RESEARCH"
__author__ = "Andreas Nautsch"
__copyright__ = "Copyright 2018 Andreas Nautsch, Hochschule Darmstadt"
__maintainer__ = "Andreas Nautsch"
__email__ = "andreas.nautsch@h-da.de"
__status__ = "Production"
__docformat__ = "reStructuredText"
__credits__ = ["Niko Brummer", "Anthony Larcher", "Edward de Villiers"]


# helper functions
# see: sidekit.bosaris.detplot
colorStyle = [
    ((0, 0, 0), "-", 2),  # black
    ((0, 0, 1.0), "--", 2),  # blue
    ((0.8, 0.0, 0.0), "-.", 2),  # red
    ((0, 0.6, 0.0), ":", 2),  # green
    ((0.5, 0.0, 0.5), "-", 2),  # magenta
    ((0.3, 0.3, 0.0), "--", 2),  # orange
    ((0, 0, 0), ":", 2),  # black
    ((0, 0, 1.0), ":", 2),  # blue
    ((0.8, 0.0, 0.0), ":", 2),  # red
    ((0, 0.6, 0.0), "-", 2),  # green
    ((0.5, 0.0, 0.5), "-.", 2),  # magenta
    ((0.3, 0.3, 0.0), "-", 2),  # orange
]

grayStyle = [
    ((0, 0, 0), "-", 2),  # black
    ((0, 0, 0), "--", 2),  # black
    ((0, 0, 0), "-.", 2),  # black
    ((0, 0, 0), ":", 2),  # black
    ((0.3, 0.3, 0.3), "-", 2),  # gray
    ((0.3, 0.3, 0.3), "--", 2),  # gray
    ((0.3, 0.3, 0.3), "-.", 2),  # gray
    ((0.3, 0.3, 0.3), ":", 2),  # gray
    ((0.6, 0.6, 0.6), "-", 2),  # lighter gray
    ((0.6, 0.6, 0.6), "--", 2),  # lighter gray
    ((0.6, 0.6, 0.6), "-.", 2),  # lighter gray
    ((0.6, 0.6, 0.6), ":", 2),  # lighter gray
]

Box = namedtuple("Box", "left right top bottom")


def probit(p):
    # see: sidekit.bosaris.detplot.__probit__
    y = numpy.sqrt(2) * erfinv(2 * p - 1)
    return y


def pavx(y):
    # see: sidekit.bosaris.detplot.pavx; with fixed bugs
    assert y.ndim == 1, "Argument should be a 1-D array"
    assert y.shape[0] > 0, "Input array is empty"
    n = y.shape[0]

    index = numpy.zeros(n, dtype=int)
    length = numpy.zeros(n, dtype=int)

    ghat = numpy.zeros(n)

    ci = 0
    index[ci] = 1
    length[ci] = 1
    ghat[ci] = y[0]

    for j in range(1, n):
        ci += 1
        index[ci] = j + 1
        length[ci] = 1
        ghat[ci] = y[j]
        while (ci >= 1) & (ghat[numpy.max(ci - 1, 0)] >= ghat[ci]):
            nw = length[ci - 1] + length[ci]
            ghat[ci - 1] = ghat[ci - 1] + (length[ci] / nw) * (ghat[ci] - ghat[ci - 1])
            length[ci - 1] = nw
            ci -= 1

    height = copy.deepcopy(ghat[: ci + 1])
    width = copy.deepcopy(length[: ci + 1])

    while n >= 0:
        for j in range(index[ci], n + 1):
            ghat[j - 1] = ghat[ci]
        n = index[ci] - 1
        ci -= 1

    return ghat, width, height


def rocch(tar_scores, nontar_scores, laplace=True):
    # see: sidekit.bosaris.detplot.rocch

    # init PAV isotonic regression
    Nt = tar_scores.shape[0]
    Nn = nontar_scores.shape[0]
    N = Nt + Nn
    scores = numpy.concatenate((tar_scores, nontar_scores))
    Pideal = numpy.concatenate((numpy.ones(Nt), numpy.zeros(Nn)))
    perturb = numpy.argsort(scores, kind="mergesort")
    Pideal = Pideal[perturb]
    if laplace:
        Pideal = numpy.concatenate(([1, 0], Pideal, [1, 0]))

    Popt, width, foo = pavx(Pideal)

    if laplace:
        Popt = Popt[2:-2]

    # ROCCH points
    nbins = width.shape[0]
    pmiss = numpy.zeros(nbins + 1)
    pfa = numpy.zeros(nbins + 1)
    left = 0
    fa = Nn
    miss = 0
    for i in range(nbins):
        pmiss[i] = miss / Nt
        pfa[i] = fa / Nn
        left = int(left + width[i])
        miss = numpy.sum(Pideal[:left])
        fa = N - left - numpy.sum(Pideal[left:])
    pmiss[nbins] = miss / Nt
    pfa[nbins] = fa / Nn

    return pmiss, pfa


def rocch_tradeoff(
    tar, non, pfa_min=5e-4, pfa_max=0.5, pmiss_min=5e-4, pmiss_max=0.5, dps=100
):
    # see: sidekit.bosaris.detplot.rocchdet
    assert (
        (pfa_min > 0) & (pfa_max < 1) & (pmiss_min > 0) & (pmiss_max < 1)
    ), "limits must be strictly inside (0,1)"
    assert (pfa_min < pfa_max) & (
        pmiss_min < pmiss_max
    ), "pfa and pmiss min and max values are not consistent"

    pmiss, pfa = rocch(tar, non)

    x = []
    y = []
    box = Box(left=pfa_min, right=pfa_max, top=pmiss_max, bottom=pmiss_min)
    for i in range(pfa.shape[0] - 1):
        xx = pfa[i : i + 2]
        yy = pmiss[i : i + 2]
        xdots, ydots = plotseg(xx, yy, box, dps)
        x = x + xdots.tolist()
        y = y + ydots.tolist()

    return numpy.array(x), numpy.array(y)


def __DETsort__(x, col=""):
    # see: sidekit.bosaris.detplot.__DETsort__
    assert x.ndim > 1, "x must be a 2D matrix"
    if col == "":
        list(range(1, x.shape[1]))

    ndx = numpy.arange(x.shape[0])

    # sort 2nd column ascending
    ind = numpy.argsort(x[:, 1], kind="mergesort")
    ndx = ndx[ind]

    # reverse to descending order
    ndx = ndx[::-1]

    # now sort first column ascending
    ind = numpy.argsort(x[ndx, 0], kind="mergesort")

    ndx = ndx[ind]
    sort_scores = x[ndx, :]
    return sort_scores


def __compute_roc__(true_scores, false_scores):
    # see: sidekit.bosaris.detplot.__compute_roc__
    num_true = true_scores.shape[0]
    num_false = false_scores.shape[0]
    assert num_true > 0, "Vector of target scores is empty"
    assert num_false > 0, "Vector of nontarget scores is empty"

    total = num_true + num_false

    Pmiss = numpy.zeros((total + 1))
    Pfa = numpy.zeros((total + 1))

    scores = numpy.zeros((total, 2))
    scores[:num_false, 0] = false_scores
    scores[:num_false, 1] = 0
    scores[num_false:, 0] = true_scores
    scores[num_false:, 1] = 1

    scores = __DETsort__(scores)

    sumtrue = numpy.cumsum(scores[:, 1], axis=0)
    sumfalse = num_false - (numpy.arange(1, total + 1) - sumtrue)

    Pmiss[0] = 0
    Pfa[0] = 1
    Pmiss[1:] = sumtrue / num_true
    Pfa[1:] = sumfalse / num_false
    return Pfa, Pmiss


def __filter_roc__(pfa, pm):
    # see: sidekit.bosaris.detplot.__filter_roc__
    out = 0
    new_pm = [pm[0]]
    new_pfa = [pfa[0]]

    for i in range(1, pm.shape[0]):
        if (pm[i] == new_pm[out]) | (pfa[i] == new_pfa[out]):
            pass
        else:
            # save previous point, because it is the last point before the
            # change.  On the next iteration, the current point will be saved.
            out += 1
            new_pm.append(pm[i - 1])
            new_pfa.append(pfa[i - 1])

    out += 1
    new_pm.append(pm[-1])
    new_pfa.append(pfa[-1])
    pm = numpy.array(new_pm)
    pfa = numpy.array(new_pfa)
    return pfa, pm


def plotseg(xx, yy, box, dps):
    # see: sidekit.bosaris.detplot.plotseg
    assert (xx[1] <= xx[0]) & (yy[0] <= yy[1]), "xx and yy should be sorted"

    XY = numpy.column_stack((xx, yy))
    dd = numpy.dot(numpy.array([1, -1]), XY)
    if numpy.min(abs(dd)) != 0:
        seg = numpy.linalg.solve(XY, numpy.array([[1], [1]]))

    # segment completely outside of box
    if (
        (xx[0] < box.left)
        | (xx[1] > box.right)
        | (yy[1] < box.bottom)
        | (yy[0] > box.top)
    ):
        xdots = numpy.array([])
        ydots = numpy.array([])
    else:
        if xx[1] < box.left:
            xx[1] = box.left
            yy[1] = (1 - seg[0] * box.left) / seg[1]

        if xx[0] > box.right:
            xx[0] = box.right
            yy[0] = (1 - seg[0] * box.right) / seg[1]

        if yy[0] < box.bottom:
            yy[0] = box.bottom
            xx[0] = (1 - seg[1] * box.bottom) / seg[0]

        if yy[1] > box.top:
            yy[1] = box.top
            xx[1] = (1 - seg[1] * box.top) / seg[0]

        dx = xx[1] - xx[0]
        xdots = xx[0] + dx * numpy.arange(dps + 1) / dps
        ydots = (1 - seg[0] * xdots) / seg[1]

    return xdots, ydots


def clean_segment(xseg, yseg, minimum_point_distance=0.01):
    # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code
    # motivated by matlab2tikz
    keep_idx = numpy.isfinite(xseg) & numpy.isfinite(yseg)
    xseg = xseg[keep_idx]
    yseg = yseg[keep_idx]

    # remove all points within Euclidean range of minimum_point_distance
    xdist = numpy.diff(xseg)
    ydist = numpy.diff(yseg)
    dist = numpy.sqrt(xdist**2 + ydist**2)
    idx_keep = numpy.concatenate(([True], dist >= minimum_point_distance))

    # find first omitted, and check on which idx is in valid range again from there
    first_pop_idx = numpy.where(idx_keep == False)[0]
    while len(first_pop_idx) > 0:
        xdist = xseg[first_pop_idx[0] - 1] - xseg[first_pop_idx]
        ydist = yseg[first_pop_idx[0] - 1] - yseg[first_pop_idx]
        dist = numpy.sqrt(xdist**2 + ydist**2)
        tmp_keep = numpy.where(dist >= minimum_point_distance)[0]
        if len(tmp_keep) > 0:
            idx_to_keep = first_pop_idx[tmp_keep[0]]
            idx_keep[idx_to_keep] = True
            first_pop_idx = first_pop_idx[numpy.where(first_pop_idx > idx_to_keep)[0]]
        else:
            first_pop_idx = []

    if len(xseg) > 0:
        xseg = xseg[idx_keep]
        yseg = yseg[idx_keep]

    return xseg, yseg


class DET:
    """
    Class for creating DET plots
    see: A. Martin, G. Doddington, T. Kamm, M. Ordowski, M. Przybocki:
         "The DET Curve in Assessment of Detection Task Performance",
         Proc. EUROSPEECH, pp. 1895-1898, 1997
    """

    def __init__(
        self,
        biometric_evaluation_type=None,
        abbreviate_axes=False,
        plot_title=None,
        plot_eer_line=False,
        plot_rule_of_30=False,
        cleanup_segments_distance=0.01,
    ):
        self.num_systems = 0
        self.system_labels = []
        self.axes_transform = probit
        self.plot_eer_line = plot_eer_line
        self.plot_rule_of_30 = plot_rule_of_30
        self.cleanup_segments_distance = cleanup_segments_distance

        self.plot_title = plot_title
        self.x_limits = numpy.array([1e-8, 0.5])
        self.y_limits = numpy.array([1e-8, 0.5])
        self.x_ticks = numpy.array(
            [
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                5e-2,
                20e-2,
                40e-2,
                65e-2,
                85e-2,
                95e-2,
            ]
        )
        self.x_ticklabels = numpy.array(
            [
                "0.00001",
                "0.0001",
                "0.001",
                "0.01",
                "0.1",
                "1",
                "5",
                "20",
                "40",
                "65",
                "85",
                "95",
            ]
        )
        self.y_ticks = numpy.array(
            [
                1e-7,
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                5e-2,
                20e-2,
                40e-2,
                65e-2,
                85e-2,
                95e-2,
            ]
        )
        self.y_ticklabels = numpy.array(
            [
                "0.00001",
                "0.0001",
                "0.001",
                "0.01",
                "0.1",
                "1",
                "5",
                "20",
                "40",
                "65",
                "85",
                "95",
            ]
        )

        if biometric_evaluation_type == "algorithm":
            if abbreviate_axes:
                self.x_label = "FMR (in %)"
                self.y_label = "FNMR (in %)"
            else:
                self.x_label = "False Match Rate (in %)"
                self.y_label = "False Non-Match Rate (in %)"
        elif biometric_evaluation_type == "system":
            if abbreviate_axes:
                self.x_label = "FAR (in %)"
                self.y_label = "FRR (in %)"
            else:
                self.x_label = "False Acceptance Rate (in %)"
                self.y_label = "False Rejection Rate (in %)"
        elif biometric_evaluation_type == "PAD":
            if abbreviate_axes:
                self.x_label = "APCER (in %)"
                self.y_label = "BPCER (in %)"
            else:
                self.x_label = "Attack Presentation Classification Error Rate (in %)"
                self.y_label = "Bona Fide Presentation Classification Error Rate (in %)"
        elif biometric_evaluation_type == "identification":
            if abbreviate_axes:
                self.x_label = "FPIR (in %)"
                self.y_label = "FNIR (in %)"
            else:
                self.x_label = "False Positive Identification Rate (in %)"
                self.y_label = "False Negative Identification Rate (in %)"
        else:
            self.x_label = "Type I Error Rate (in %)"
            self.y_label = "Type II Error Rate (in %)"
            self.x_limits = numpy.array([1e-8, 0.99])
            self.y_limits = numpy.array([1e-8, 0.99])

    def create_figure(self):
        """
        Creates empty DET plot figure
        """
        self.figure = mpl.figure()
        ax = self.figure.add_subplot(111)
        ax.set_aspect("equal")

        mpl.axis(
            [
                self.axes_transform(self.x_limits[0]),
                self.axes_transform(self.x_limits[1]),
                self.axes_transform(self.y_limits[0]),
                self.axes_transform(self.y_limits[1]),
            ]
        )

        ax.set_xticks(self.axes_transform(self.x_ticks))
        ax.set_xticklabels(self.x_ticklabels, size="x-small")

        ax.set_yticks(self.axes_transform(self.y_ticks))
        ax.set_yticklabels(self.y_ticklabels, size="x-small")

        mpl.grid(True)  # grid_color = '#b0b0b0'
        mpl.xlabel(self.x_label)
        mpl.ylabel(self.y_label)
        if self.plot_title is not None:
            mpl.title(self.plot_title)

        mpl.gca().set_xlim(
            left=self.axes_transform(self.x_limits[0]),
            right=self.axes_transform(self.x_limits[1]),
        )
        mpl.gca().set_ylim(
            bottom=self.axes_transform(self.y_limits[0]),
            top=self.axes_transform(self.y_limits[1]),
        )
        mpl.gca().set_aspect("equal")
        if self.plot_eer_line:
            eer_line = numpy.linspace(
                min(
                    self.axes_transform(self.x_limits[0]),
                    self.axes_transform(self.y_limits[0]),
                ),
                max(
                    self.axes_transform(self.x_limits[1]),
                    self.axes_transform(self.y_limits[1]),
                ),
                1000,
            )
            mpl.plot(
                eer_line,
                eer_line,
                color="#b0b0b0",
                linestyle="-",
                linewidth=0.6,
                label=None,
            )

    def plot(
        self,
        tar,
        non,
        label="",
        style="color",
        plot_args="",
        dissimilarity_scores=False,
        plot_rocch=False,
    ):
        if dissimilarity_scores:
            tar = -numpy.array(tar)
            non = -numpy.array(non)
        if not (isinstance(plot_args, tuple) & (len(plot_args) == 3)):
            if style == "gray":
                plot_args = grayStyle[self.num_systems]
            else:
                plot_args = colorStyle[self.num_systems]
        if plot_rocch:
            x, y = rocch_tradeoff(
                tar,
                non,
                self.x_limits[0],
                self.x_limits[1],
                self.y_limits[0],
                self.y_limits[1],
            )
        else:
            # steppy ROC
            x, y = __compute_roc__(tar, non)
            x, y = __filter_roc__(x, y)

        # transform to DET scale
        xseg = self.axes_transform(x)
        yseg = self.axes_transform(y)

        if self.cleanup_segments_distance:
            xseg, yseg = clean_segment(
                xseg, yseg, minimum_point_distance=self.cleanup_segments_distance
            )

        mpl.plot(
            xseg,
            yseg,
            label=label,
            color=plot_args[0],
            linestyle=plot_args[1],
            linewidth=plot_args[2],
        )

        if self.plot_rule_of_30:
            self.__plot_x_rule_of_30__(num_scores=non.shape[0])
            self.__plot_y_rule_of_30__(num_scores=tar.shape[0])

        self.num_systems += 1
        self.system_labels.append(label)

    def show(self):
        """
        onl
        """
        mpl.show()

    def __plot_x_rule_of_30__(
        self, num_scores, plot_args=((0, 0, 0), "--", 1), legend_string=""
    ):
        # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code
        # see: sidekit.bosaris.detplot.DETplot

        pfaval = 30.0 / num_scores

        if not (pfaval < self.y_limits[0]) | (pfaval > self.y_limits[1]):
            drx = self.axes_transform(pfaval)
            mpl.vlines(
                drx,
                ymin=self.axes_transform(self.y_limits[0]),
                ymax=self.axes_transform(self.y_limits[1]),
                color=plot_args[0],
                linestyle=plot_args[1],
                linewidth=plot_args[2],
                label=None,
            )

    def __plot_y_rule_of_30__(
        self, num_scores, plot_args=((0, 0, 0), "--", 1), legend_string=""
    ):
        # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code
        # see: sidekit.bosaris.detplot.DETplot

        pmissval = 30.0 / num_scores

        if not (pmissval < self.x_limits[0]) | (pmissval > self.x_limits[1]):
            dry = self.axes_transform(pmissval)
            mpl.hlines(
                y=dry,
                xmin=self.axes_transform(self.x_limits[0]),
                xmax=self.axes_transform(self.x_limits[1]),
                color=plot_args[0],
                linestyle=plot_args[1],
                linewidth=plot_args[2],
                label=None,
            )

    def legend(self, enable, **kwargs):
        if enable:
            self.legend_on(**kwargs)
        else:
            self.legend_off()

    def legend_on(self, **kwargs):
        kwargs.setdefault("loc", 0)
        mpl.legend(**kwargs)

    def legend_off(self):
        mpl.legend().remove()

    def save(self, filename, type="tikz", dpi=None, width="120pt", height="120pt"):
        if type in ["pdf", "png"]:
            mpl.savefig(filename + "." + type, bbox_inches="tight")
        elif type in ["tikz", "tex", "latex", "LaTeX"]:
            self.__save_as_tikzpgf__(
                outfilename=filename + ".tex", dpi=dpi, width=width, height=height
            )
        else:
            raise ValueError("unknown save format")

    def __save_as_tikzpgf__(
        self,
        outfilename,
        dpi=None,
        width="140pt",
        height="140pt",
        extra_axis_parameters=["xticklabel style={rotate=90}"],
        extra_tikzpicture_parameters=["[font=\\scriptsize]"],
    ):
        # see: https://codeocean.com/algorithm/154591c8-9d3f-47eb-b656-3aff245fd5c1/code

        """
        def replace_tick_label_notation(tick_textpos):
            tick_label = tick_textpos.get_text()
            if 'e' in tick_label:
                tick_label = int(tick_label.replace('1e', '')) - 2
                tick_textpos.set_text('%f' % (10 ** (int(tick_label) - 2)))
        """

        mpl.xlabel(mpl.axes().get_xlabel().replace("%", "\%"))
        mpl.ylabel(mpl.axes().get_ylabel().replace("%", "\%"))

        if dpi is not None:
            mpl.gcf().set_dpi(dpi)

        mpl.gca().set_title("")

        """
        for tick_textpos in mpl.gca().get_xmajorticklabels():
            replace_tick_label_notation(tick_textpos)
        for tick_textpos in mpl.gca().get_ymajorticklabels():
            replace_tick_label_notation(tick_textpos)
        """

        tikz_save(
            outfilename,
            figurewidth=width,
            figureheight=height,
            extra_axis_parameters=extra_axis_parameters,
            extra_tikzpicture_parameters=extra_tikzpicture_parameters,
        )
