"""A figure generator for slicing and displaying data.  Example usage:

.. code-block:: python

    import pandas
    data = pandas.DataFrame()
    data["a"] = None
    data["b"] = None
    data.loc[0] = [ 1, 1 ]
    data.loc[1] = [ 2, 2 ]
    data.loc[2] = [ 3, 2 ]
    maker = FigureMaker(data, "images/", imageExtension = "png")
    maker.bind("a").setup(label = "First column")
    maker.bind("b").setup(label = "Second column", scale = "log")

    with maker.new("a vs b") as fig:
        fig.plotValue("a and b", "a", "b")
"""

import math
import os
import matplotlib
import matplotlib.image
from matplotlib import pyplot
import numpy as np
import pandas
import PIL.Image
import re
# Should come from python-slugify!
from slugify import slugify

_missing = {}


class _Figure(object):
    r"""Produced by :meth:`FigureMaker.new`.  Most plotting functions can take
    the following arguments:

    :kwargs:
        :sort: By default, data is sorted by the data columns that are being
                used for the plot.  However, if there is a hidden column (such
                as time) that should be used for sorting instead, specify it
                here.
    """
    colors = ['b', 'g', 'r', 'c', 'm', 'k']
    styles = ['-', '--', ':', '-.']
    markers = [' ', 'd', '*', 'o', 's', 'v', '<']

    def __init__(self, maker, figureName, axes, legendAnchor):
        self.maker = maker
        self.axes = axes
        self.axes.locator_params(tight=True)
        self._legendAnchor = legendAnchor
        self.width = axes.figure.get_size_inches()[0] * axes.get_position().width
        self.height = axes.figure.get_size_inches()[1] * axes.get_position().height
        self._lineCount = 0
        self._legendHandles = []
        self._figureName = figureName
        self._figureList = None

        self._xBinding = None
        self._yBinding = None
        self._yBinding2 = None
        # Used only when there are two y axes.  Color of first rendered line
        self._yaxis1Color = None
        self._yaxis2 = None

        # For multiple renderings
        self._saved = False


    def __enter__(self):
        return self


    def __exit__(self, type, value, tb):
        if type is None:
            # Draw our legend
            extraArtists = []
            if self._legendAnchor is not None and self._legendHandles:
                legArgs = {
                        'handles': self._legendHandles,
                }
                if isinstance(self._legendAnchor, (tuple, list)):
                    # Anchor!
                    l = self.axes.legend(loc = self._legendAnchor[0],
                            bbox_to_anchor = self._legendAnchor[1:], **legArgs)
                else:
                    l = self.axes.legend(loc = self._legendAnchor, **legArgs)
                extraArtists.append(l)

            # Update second y axis colors
            if self._yaxis2 is not None:
                self.axes.set_ylabel(self.axes.get_ylabel(),
                        color = self._yaxis1Color)
                for t in self.axes.get_yticklabels():
                    t.set_color(self._yaxis1Color)
                for t in self._yaxis2.get_yticklabels():
                    t.set_color(self._yaxis2.yaxis.label.get_color())

            if self._figureList is None:
                self._saveFigure(extraArtists)
            else:
                self._figureList.extraArtists.extend(extraArtists)


    def _saveFigure(self, extraArtists):
        # Finish rendering to file, THEN render to IPython with the title
        if self.maker.imageDestFolder is not None:
            fname = self.maker.getFigurePath(self._figureName)
            if os.path.lexists(fname):
                if not self._saved:
                    raise ValueError("Cannot name two figures {} (slugged {})"
                            .format(self._figureName, slugName))
            matplotlib.pyplot.savefig(fname,
                    dpi = pyplot.rcParams['figure.dpi'],
                    bbox_inches = 'tight',
                    bbox_extra_artists = extraArtists)
            self._saved = True
        self.axes.figure.suptitle(self._figureName, fontsize = 24)

        # Close to save memory
        pyplot.close(self.axes.figure)


    def getDataLimits(self):
        """Returns the currently plotted data limits.  Object has xmin, xmax,
        ymin, and ymax attributes."""
        return self.axes.dataLim


    def plotImage(self, x, img, **kwargs):
        """Plot a column of image _paths_ as staggered images in a plot.
        """
        def innerPlot(axes, data, options):
            xMin = data[x].min()
            xMax = data[x].max()
            def mapX(x):
                """Returns the imAxes.x coordinate for a data (axes.) x coordinate"""
                return (x - xMin) / (xMax - xMin)
            def unmapX(x):
                """Returns the axes.x coordinate for an imAxes.x coordinate"""
                return xMin + x * (xMax - xMin)
            def calcBounds(x, width):
                """Given an imAxes x and width, return the imAxes left and right."""
                partLeft = x
                return (x - width * partLeft, x + width * (1.0 - partLeft))

            # Use the twin axis for our images, and the primary x axis for ticks
            # (and possibly log scaling)
            imAxes = axes.twiny()

            # Original axes must be set AFTER twinning (true for y, anyway)
            axes.set_xlim(xMin, xMax)
            axes.set_ylim(0.0, 1.0)
            axes.set_ymargin(0.0)
            axes.get_yaxis().set_ticks([])
            axes.plot([ xMin, xMax ], [ 0.0, 1.0 ], linestyle = ' ')
            self._legendAnchor = None

            # Now set up our twin axes
            imAxes.set_xscale("linear")
            imAxes.set_xlim(0.0, 1.0)
            imAxes.set_xmargin(0.0)
            imAxes.get_xaxis().set_ticks([])

            # Gutter margin for arrows
            imMargin = 0.01
            imLowestY = imMargin

            if axes.get_xscale() == "log":
                def mapLogX(x):
                    return (math.log(x) - math.log(xMin)) / (math.log(xMax) - math.log(xMin))
                mapX = mapLogX
                def unmapLogX(x):
                    return math.exp(x * (math.log(xMax) - math.log(xMin)) + math.log(xMin))
                unmapX = unmapLogX

            # List of [ imData, imAxesAspect, dataX, imAxesY, imAxesWidth ]
            images = []
            # imAxes are bound on X between [0, 1] and Y between [0, 1].  So,
            # aspect is same as figure width and height.
            imAxesAspect = self.height / self.width
            for coord, imPath in zip(data[x].values, data[img].values):
                # Load and ensure RGBA, flip to correspond to matplotlib's coordinates
                image = PIL.Image.open(imPath).convert("RGBA")
                # nativeAspect is the aspect ratio of the image
                nativeAspect = image.size[1] * 1.0 / image.size[0]
                axesAspect = nativeAspect / imAxesAspect
                images.append([ np.asarray(image), axesAspect, coord, 1, min(1.0, (1.0 - imMargin) / axesAspect) ])

            # Now figure out positions....  this is slow, but we want them all to be the
            # same scale, so...just stack them ideally.
            baseScale = 1.0
            while True:
                ok = True
                for i in range(len(images)):
                    # For each image, detect a "safe y" based on images before it
                    safeY = 1.0
                    myBoundsX = calcBounds(mapX(images[i][2]), images[i][4] * baseScale)
                    for j in range(i):
                        myBoundsY = (safeY - images[i][4] * baseScale * images[i][1], safeY)
                        boundsX = calcBounds(mapX(images[j][2]), images[j][4] * baseScale)
                        boundsY = (images[j][3] - images[j][4] * baseScale * images[j][1], images[j][3])

                        if myBoundsX[0] <= boundsX[1] and myBoundsX[1] >= boundsX[0]:
                            if myBoundsY[0] <= boundsY[1] and myBoundsY[1] >= boundsY[0]:
                                # Overlap!  We can only move down, since we start at the top
                                safeY = boundsY[0] - imMargin
                                if safeY - images[i][4] * baseScale * images[i][1] < imLowestY:
                                    ok = False
                                    break

                    images[i][3] = safeY

                    if not ok:
                        break

                if ok:
                    break
                baseScale *= 0.9

            # Now that we've calculated the display, render everything
            for imData, axesAspect, dataX, imAxesY, imAxesWidth in images:
                imX = mapX(dataX)
                # Apply our new scaling
                imAxesWidth *= baseScale
                imYBottom = imAxesY - axesAspect * imAxesWidth
                bounds = calcBounds(imX, imAxesWidth)
                imAxes.imshow(imData, interpolation = 'nearest', aspect = 'auto',
                        extent = (bounds[0], bounds[1], imYBottom, imAxesY))
                arrowX = (bounds[0] + bounds[1]) * 0.5
                imAxes.annotate("",
                        xy = (imX, 0), xycoords = 'data',
                        xytext = (arrowX, imYBottom), textcoords = 'data',
                        arrowprops = dict(arrowstyle = "->", connectionstyle = "arc3"))

        self._callPlot(x, img, innerPlot, label = None, **kwargs)


    def plotBar(self, label, x, y, y2 = None, style = {}, xInd=0, **kwargs):
        def plotBar_inner(axes, data, options):
            opts = { 'width': 0.35 }
            if y2 is not None:
                opts['yerr'] = data[y2]
            opts.update(options)
            opts.pop('linestyle')
            dx = np.arange(len(data[x]))
            result = axes.bar(dx + xInd * opts['width'], data[y], **opts)
            locs = axes.set_xticks(dx + opts['width']*(xInd + 1) * 0.5)
            axes.set_xticklabels(data[x].tolist())
            return result

        styl = { 'marker': None }
        styl.update(style)
        return self._callPlot(x, y, plotBar_inner, label, styl, **kwargs)


    def plotCustom(self, cb, xAxis, yAxis, style={}, **kwargs):
        """Calls ``cb`` with the data filtered as specified by ``kwargs`` and
        the axes set up for ``xAxis`` and ``yAxis``."""
        def plotCustomInner(axes, data, options):
            cb(axes, data)
        self._callPlot(xAxis, yAxis, plotCustomInner, None, style, **kwargs)


    def plotCustomValue(self, label, xTuple, yTuple, style = {}, **kwargs):
        def plotNoMarkers(axes, data, options):
            options.pop('markevery')
            return axes.plot(xTuple[1], yTuple[1], **options)[0]

        self._callPlot(xTuple[0], yTuple[0], plotNoMarkers, label, style,
                **kwargs)


    def plotError(self, label, x, y, y2, style = {}, **kwargs):
        self._callPlot(
                x, y,
                lambda axes, data, options: axes.errorbar(data[x],
                        data[y], data[y2], capsize = 6, **options),
                label, style, **kwargs)


    def plotFill(self, x, y, y2, style = {}, **kwargs):
        self._callPlot(
                x, y,
                lambda axes, data, options: axes.fill_between(data[x],
                        data[y], data[y2], facecolor = options['color'],
                        alpha = options['alpha']),
                None, style, **kwargs)


    def plotHeat(self, x, y, z, style = {}, **kwargs):
        """Uses pcolor.  Extra style options:

        smooth - If True, smooth the heatmap

        useDataLimits - If True, ignore bounds on z axis, using data limits
                after filtering instead.
        """
        style = style.copy()
        smooth = style.pop('smooth', False)
        useDataLimits = style.pop('useDataLimits', False)

        data = self._getFiltered([], **kwargs)

        # The bindings for the different axes
        xBinding = self.maker.getBinding(x)
        yBinding = self.maker.getBinding(y)
        zBinding = self.maker.getBinding(z)

        nd = data.pivot(y, x, z)
        X, Y = np.meshgrid(nd.columns.values, nd.index.values)

        # Normalize according to zlimits
        limits = zBinding.limits[:]
        if limits[0] is None or useDataLimits:
            limits[0] = nd.values.min()
        if limits[1] is None or useDataLimits:
            limits[1] = nd.values.max()
        if zBinding.scale == "linear":
            normalizer = matplotlib.colors.Normalize(limits[0], limits[1])
        elif zBinding.scale == "log":
            normalizer = matplotlib.colors.LogNorm(limits[0], limits[1])
        else:
            raise NotImplementedError(zBinding.scale)

        defaults = dict(norm = normalizer,
                shading = 'gouraud' if smooth else 'flat')
        defaults.update(style)
        plotted = self.axes.pcolormesh(X, Y, nd.values, **defaults)

        self.axes.set_ylabel(yBinding.label)
        self.axes.set_yscale(yBinding.scale)
        self.axes.set_xlabel(xBinding.label)
        self.axes.set_xscale(xBinding.scale)
        self.axes.figure.colorbar(plotted)


    def plotStem(self, label, x, y, style = {}, **kwargs):
        def doStem(axes, data, options):
            r = axes.stem(data[x], data[y])
            #r = (marker, stem, base)
            markOpts = { k: v for k, v in options.iteritems()
                    if k.startswith('mark') or k in [ 'color', 'label' ] }
            r[0].update(markOpts)
            nopts = options.copy()
            nopts.pop('marker')
            [ c.update(nopts) for c in r[1] ]
            nopts = options.copy()
            nopts.pop('markevery')
            r[2].update(nopts)
            return r[2]
        self._callPlot(x, y, doStem, label, style, **kwargs)


    def plotValue(self, label, x, y, style = {}, **kwargs):
        self._callPlot(x, y,
                # Note that this lambda returns the first object since we only plot
                # one line.
                lambda axes, data, options: axes.plot(data[x], data[y],
                        **options)[0],
                label, style, **kwargs)


    def render(self):
        """For usage without a 'with' block.  Can be called multiple times"""
        self.__exit__(None, None, None)


    def _getFiltered(self, sortFields, **kwargs):
        data = self.maker.data
        for k, v in kwargs.iteritems():
            if k.endswith("_lt"):
                k = k[:-3]
                data = data.loc[data[k] < v]
            elif k.endswith("_lte"):
                k = k[:-4]
                data = data.loc[data[k] <= v]
            elif k.endswith("_gt"):
                k = k[:-3]
                data = data.loc[data[k] > v]
            elif k.endswith("_gte"):
                k = k[:-4]
                data = data.loc[data[k] >= v]
            elif isinstance(v, (tuple, list)):
                data = data.loc[data[k].isin(v)]
            else:
                data = data.loc[data[k] == v]

        if sortFields and self.maker._sort:
            data.sort(sortFields, inplace = True)

        return data


    def _callPlot(self, xDataName, yDataName, plotMethod, label, style,
            sort=None, **kwargs):
        if sort is None:
            sort = [ xDataName, yDataName ]
        data = self._getFiltered(sort, **kwargs)

        if len(data) == 0:
            return

        plotOptions = {
                'label': label,
                'color': self.colors[self._lineCount % len(self.colors)],
                'linestyle': self.styles[self._lineCount % len(self.styles)],
                'marker': self.markers[self._lineCount % len(self.markers)],
                'markeredgecolor': (1.0, 1.0, 1.0, 0.0),
                'markersize': 12,
        }
        if 'marker' in style:
            # We need to overwrite this now so that we'll know if we want to
            # skip marker calculations (done for marker is None)
            plotOptions['marker'] = style['marker']

        if plotOptions['marker'] == '*':
            # stars are small
            plotOptions['markersize'] *= 1.5

        # The bindings for the different axes
        xBinding = self.maker.getBinding(xDataName)
        yBinding = self.maker.getBinding(yDataName)

        if plotOptions['marker']:
            # Calculate markers according to numTicks
            numTicks = 5.0 - 0.5 * self._lineCount
            if numTicks < 3.0:
                numTicks = 9.25 - 0.5 * self._lineCount
            if numTicks < 3.0:
                numTicks = 13.35 - 0.5 * self._lineCount
            markers = []
            xdata = data[xDataName].values
            if xBinding.scale == 'log':
                xdata = [ math.log(x) for x in xdata ]
            # If we want 4 ticks, plus an extra half space at each end, then we
            # need exactly as many spaces as we have ticks.
            markDist = (xdata[-1] - xdata[0]) / numTicks
            # Since we have half ticks, rather than using a half offset, we'll
            # use a quarter for those.
            scalar = 0.5
            scalar = 1.0 - 0.5 * (numTicks - math.floor(numTicks - 1))
            markNext = xdata[0] + markDist * scalar
            for q in range(len(xdata)):
                if xdata[q] >= markNext:
                    markers.append(q)
                    markNext += markDist
            plotOptions['markevery'] = markers

            # End of plot options, update with style
            plotOptions.update(style)
        else:
            # Update with style BEFORE removing marker stuff
            plotOptions.update(style)

            # Take out all marker options
            plotOptions.pop('marker')
            plotOptions.pop('markeredgecolor')
            plotOptions.pop('markersize')

        axes = self.axes

        # Enforce X and Y axis general properties BEFORE render, since e.g.
        # image rendering needs to know if something is a log distribution.
        if self._xBinding is None:
            self._xBinding = xBinding
            axes.set_xlabel(xBinding.label)
            axes.set_xscale(xBinding.scale)
        elif self._xBinding != xBinding:
            raise ValueError("Using multiple x axes in one plot?  Unwise!")

        canUseSecondY = True
        useSecondY = False
        if 'axis' in plotOptions:
            ax = plotOptions.pop('axis')
            if ax == 0:
                canUseSecondY = False
            elif ax == 1:
                useSecondY = True
            else:
                raise ValueError("Got 'axis' from style: {}.  Must be 0 or 1."
                        .format(ax))

        if self._yBinding is None:
            self._yBinding = yBinding
            axes.set_ylabel(yBinding.label)
            axes.set_yscale(yBinding.scale)
            self._yaxis1Color = plotOptions['color']
        elif canUseSecondY and (self._yBinding != yBinding or useSecondY):
            if self._yBinding2 == yBinding:
                # Ok, plot to second
                axes = self._yaxis2
                pass
            elif self._yBinding2 is None:
                # Create a second axis and plot to that
                self._yBinding2 = yBinding
                self._yaxis2 = axes = axes.twinx()
                axes.set_ylabel(yBinding.label, color = plotOptions['color'])
                axes.set_yscale(yBinding.scale)
            else:
                raise ValueError("Using multiple y axes in one plot?  Unwise!")

        plotted = plotMethod(axes, data, plotOptions)
        if plotted is not None:
            self._legendHandles.append(plotted)
            self._lineCount += 1

        # --- Update xlimits based on data
        dataLimits = (axes.dataLim.xmin, axes.dataLim.xmax)
        # Fully specified is OK, but a single None doesn't really work with matplotlib
        limits = xBinding.limits
        if limits[0] is None or limits[1] is None:
            axes.set_xlim(auto = True)
            xMin, xMax = dataLimits
            if limits[0] is not None:
                xMin = limits[0]
            if limits[1] is not None:
                xMax = limits[1]
            axes.set_xlim(xMin, xMax)
        else:
            axes.set_xlim(limits)

        # --- Update ylimits and margin based on data
        dataLimits = (axes.dataLim.ymin, axes.dataLim.ymax)
        # Fully specified is OK, but a single None doesn't really work with matplotlib
        limits = yBinding.limits
        if limits[0] is None or limits[1] is None:
            axes.set_ylim(auto = True)
            yMin, yMax = dataLimits
            if limits[0] is not None:
                yMin = limits[0]
            if limits[1] is not None:
                yMax = limits[1]
            axes.set_ylim(yMin, yMax)
        else:
            axes.set_ylim(limits)

        return plotted


class _FigureList(list):
    def __init__(self, fig):
        super(_FigureList, self).__init__()
        self.fig = fig
        self.extraArtists = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        for e in self:
            e.__exit__(type, value, tb)

        # Fine tuning, remove duplicate x ticks and put the plots next to
        # each other
        pyplot.setp([ a.get_xticklabels() for a in self.fig.axes[:-1] ],
                visible = False)

        # Do this only once
        self[0]._saveFigure(self.extraArtists)


    def append(self, val):
        super(_FigureList, self).append(val)
        val._figureList = self


    def render(self):
        """For usage without a 'with' block.  Can be called multiple times"""
        self.__exit__(None, None, None)



class FigureMaker(object):
    @property
    def sort(self):
        return self._sort


    @sort.setter
    def sort(self, newVal):
        self._sort = newVal


    def __init__(self, data, imageDestFolder = None, imageExtension = 'png',
            imagePrefix = None, defaultHeightAspect = 0.618, sort = True):
        """Creates a rendering platform based on some data.  Note that sort
        applies to individual plots!  E.g., you cannot make a proper I-V curve
        or other plot that has multiple Y-value for a single X-value if sort is
        set to True (default)."""
        self.imageDestFolder = imageDestFolder
        self.imagePrefix = imagePrefix
        self.imageExtension = imageExtension
        if imageExtension not in [ 'eps', 'pdf', 'png' ]:
            raise ValueError("Unrecognized image format: {}".format(
                    imageExtension))

        # Clear out all old png files
        if imageDestFolder is not None:
            try:
                os.makedirs(imageDestFolder)
            except OSError, e:
                # Already exists
                if e.errno != 17:
                    raise
            for fn in os.listdir(imageDestFolder):
                # Only remove images of our type and prefix
                if not re.match(r"^.*\.(eps|pdf|png|mp4)$", fn):
                    continue

                if imagePrefix is None or fn.startswith(imagePrefix):
                    os.remove(os.path.join(imageDestFolder, fn))

        self._defaults = { 'heightAspect': defaultHeightAspect }
        self._marginDefault = 0.00
        self._unnamedCount = 0
        self._sort = sort

        # Sets _bindings and _nameToBinding
        self.reset(data)


    class Binding(object):
        """Represents a group of data sharing the same axis."""
        __slots__ = [ '_maker', '_names', 'label', 'limits', 'scale' ]
        def __init__(self, maker, dataNames):
            self._maker = maker
            self._names = dataNames
            if len(dataNames) == 1:
                self.label = dataNames[0]
            else:
                self.label = '[' + ','.join(dataNames) + ']'
            self.limits = [ None, None ]
            self.scale = "linear"


        def __repr__(self):
            return "FigureMaker.Binding({})".format(self.label)


        def add(self, *args, **kwargs):
            """Same effect as calling maker.bind() with the union of the
            bindings specified by add's arguments and the original bind call.
            """
            nkwargs = kwargs.copy()
            nkwargs['addTo'] = self
            return self._maker.bind(*args, **nkwargs)


        def setup(self, **kwargs):
            self.label = kwargs.pop('label', self.label)
            self.limits = kwargs.pop('limits', self.limits)
            self.scale = kwargs.pop('scale', self.scale)
            if len(kwargs) != 0:
                raise ValueError("Unrecognized kwargs: {}".format(kwargs))
            return self


    def bind(self, dataNames = [], startswith = None, endswith = None,
            addTo = None):
        """Accepts either a single string or a collection of strings.  Binds all
        given names together on the same scale, allowing them to share a label
        and other settings.

        Note: You can also specify bind(startswith = "hey") to bind similarly
        prefixed names.  Suffixes available via endswith.

        addTo - If specified, add detected columns to this binding rather than
                creating a new binding.
        """
        colset = []
        if isinstance(dataNames, basestring):
            colset = [ dataNames ]
        elif dataNames:
            colset = dataNames
        elif startswith is not None:
            colset = [ n for n in self.data.columns.tolist()
                    if n.startswith(startswith) ]
            if endswith is not None:
                colset = [ n for n in colset if n.endswith(endswith) ]
        elif endswith is not None:
            colset = [ n for n in self.data.columns.tolist()
                    if n.endswith(endswith) ]
        else:
            raise ValueError("Unrecognized bind arguments")

        if addTo is None:
            binding = FigureMaker.Binding(self, colset)
            self._bindings.append(binding)
            for name in colset:
                if name in self._nameToBinding:
                    raise ValueError("Column {} is already bound!".format(name))
                self._nameToBinding[name] = binding

            return binding
        else:
            for name in colset:
                if name in self._nameToBinding:
                    if self._nameToBinding[name] != addTo:
                        raise ValueError("Column {} is already bound to a "
                                "different binding!".format(name))
                    # OK if it's in addTo
                else:
                    self._nameToBinding[name] = addTo
                    addTo._names.append(name)

            return addTo


    def getBinding(self, dataName):
        """Returns the Binding for a given dataName"""
        r = self._nameToBinding.get(dataName)
        if r is None:
            r = self.bind(dataName)
        return r


    def getFigurePath(self, figureName, extension=None):
        """Given a figureName, return the path to that figure.  Optionally,
        the extension may be overridden.
        """
        slugName = slugify(figureName)
        prefix = ''
        if self.imagePrefix is not None:
            prefix = self.imagePrefix
        ext = ''
        if extension is None:
            ext = '.' + self.imageExtension
        else:
            ext = '.' + extension
        fname = os.path.join(self.imageDestFolder,
                prefix + slugName + ext)
        return fname


    def new(self, figureName = None, heightAspect = None, scale = 1.0,
            legendLoc = 0, legendAnchor = None, nplotsHigh = 1):
        """legendLoc - The loc argument for a legend in pyplot.  1 is upper right,
        2 is lower right, etc.  Defaults to 0 - best.  Can also be strings 'upper
        right', etc.  Pass legendLoc = None to disable the legend.

        legendAnchor - Overrides legendLoc.  3-tuple of (corner, x, y).
                E.g. ('upper right', 0, 0) will position the legend's upper right
                corner at 0, 0

        nplotsHigh - Number of plots on the vertical scale.

        Returns - if nplotsHigh is not specified, returns a _Figure object for
        plotting.  If it is specified, returns a list of _Figure objects in
        left->right and top->bottom (english reading order).
        """
        baseSize = 10.0 * scale
        if figureName is None:
            self._unnamedCount += 1
            figureName = "Unnamed {}".format(self._unnamedCount)
        if heightAspect is None:
            heightAspect = self._defaults['heightAspect']
        fig, axes = pyplot.subplots(nplotsHigh,
                figsize = (baseSize, baseSize * heightAspect), squeeze = False,
                sharex = True, sharey = True)

        # No space between axes!
        fig.subplots_adjust(hspace = 0)

        if legendAnchor is None:
            legendAnchor = legendLoc
        figs = _FigureList(fig)
        for r in axes:
            for c in r:
                figs.append(_Figure(self, figureName, c, legendAnchor))
        if nplotsHigh != 1:
            return figs
        figs[0]._figureList = None
        return figs[0]


    def reset(self, newData, imagePrefix = _missing):
        """Resets the FigureMaker to operate on a new set of data.  This has the
        advantage of not deleting files, and keeping format / other options.
        However, bindings are erased."""
        self.data = pandas.DataFrame(newData)
        self._bindings = []
        self._nameToBinding = {}

        if imagePrefix is not _missing:
            self.imagePrefix = imagePrefix



# Set all plot default parameters
matplotlib.rc('font', size = 16)
pyplot.rcParams.update({
        'figure.dpi': 300,
        'figure.figsize': (8, 4),
        'font.size': 16,
        'legend.fontsize': 16,
        'text.latex.unicode': True,
        'text.usetex': True,
})
