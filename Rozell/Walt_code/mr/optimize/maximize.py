
from mr.figureMaker import FigureMaker

import inspect
import math
import numpy as np
import os
import pandas
import random
import sklearn
import sys
import time

from job_stream.inline import Work, Multiple

class MetaRunner(object):
    # 1.0 means the capacity to move across a single axis' full range
    VITALITY_START = 0.2
    VITALITY_MIN = 0.01
    VITALITY_STEPS = 20
    # VITALITY_START * VITALITY_LOSS ^ VITALITY_STEPS = VITALITY_MIN
    VITALITY_LOSS = math.exp(math.log(VITALITY_MIN / VITALITY_START)
            / VITALITY_STEPS)

    class AXES_TYPES(object):
        SCALAR = "scalar"
        LINEAR_RANGE = "linear_range"
        LOGARITHMIC_RANGE = "logarithmic_range"

    def __init__(self, id, nRunners, params):
        self._id = id
        self._nRunners = nRunners
        self._axes = {}
        for v in params:
            name = v[0]
            vrange = v[1]
            validator = None
            if len(v) == 2:
                pass
            elif len(v) == 3:
                try:
                    # This bug is because multiprocessing doesn't request a
                    # single process for e.g. frames.
                    import cPickle
                    cPickle.dumps(v[2], cPickle.HIGHEST_PROTOCOL)
                except:
                    raise NotImplementedError()
                validator = v[2]
            else:
                raise ValueError("Unrecognized tuple length: {}".format(len(v)))

            if isinstance(vrange, (list, tuple)):
                if len(vrange) == 2:
                    self._axes[name] = { 'type': self.AXES_TYPES.LINEAR_RANGE,
                            'scale': (vrange[1] - vrange[0]),
                            'min': vrange[0], 'max': vrange[1] }
                elif len(vrange) == 3:
                    if vrange[2] == 'log':
                        self._axes[name] = {
                                'type': self.AXES_TYPES.LOGARITHMIC_RANGE,
                                'scale': math.log(vrange[1])
                                    - math.log(vrange[0]),
                                'min': math.log(vrange[0]),
                                'max': math.log(vrange[1]) }
                    else:
                        raise NotImplementedError()
                else:
                    raise ValueError("Unrecognized length: {}".format(vrange))
            elif isinstance(vrange, (float, int, type(None))):
                # Scalar
                self._axes[name] = { 'type': self.AXES_TYPES.SCALAR,
                        'value': vrange }
            else:
                raise ValueError("Unrecognized param range: {}".format(vrange))

            self._axes[name]['validator'] = validator
        self._state = None


    def getNextTrial(self, lastResult = None, scoreMinStart = None):
        if (self._state is None or lastResult is None
                or self._life <= self.VITALITY_MIN
                or (self._score is None and lastResult[1] <= scoreMinStart)):
            # Initialize!
            self._score = None
            self._life = self.VITALITY_START
            self._state = { 'id': self._id }
            # Every _id should be unique, plus this lets us keep track of
            # how much work we've done
            if self._id % self._nRunners == 0:
                generation = self._id // self._nRunners
                if generation % 1 == 0:
                    sys.stderr.write("  (training generation {})\n".format(
                            generation))
            self._id += self._nRunners
            self._direction = self._randomDirection()
            for k, v in self._axes.iteritems():
                if v['type'] == self.AXES_TYPES.SCALAR:
                    self._state[k] = v['value']
                elif v['type'] == self.AXES_TYPES.LINEAR_RANGE:
                    self._state[k] = random.uniform(v['min'], v['max'])
                elif v['type'] == self.AXES_TYPES.LOGARITHMIC_RANGE:
                    self._state[k] = math.exp(random.uniform(v['min'], v['max']))
                else:
                    raise NotImplementedError()
            self._nextState = self._state
        else:
            # lastResult is tuple: (params, score, score_dev)
            if self._score is None or lastResult[1] > self._score:
                # Keep the new state, and keep heading in the same direction
                self._state = self._nextState
                self._score = lastResult[1]
            else:
                # Head orthogonal
                ndim = len(self._direction)
                if ndim == 1:
                    k, = self._direction.keys()
                    self._direction[k] *= -1.0
                elif ndim == 2:
                    k1, k2 = self._direction.keys()
                    t = self._direction[k1]
                    self._direction[k1] = -self._direction[k2]
                    self._direction[k2] = t
                    if random.random() < 0.5:
                        # Take the other direction (flip it)
                        self._direction[k1] *= -1.0
                        self._direction[k2] *= -1.0
                else:
                    # Generate an arbitrary vector that is orthogonal to our
                    # current self._direction
                    dirLen = 0.0
                    dotLen = 0.0
                    newDir = dict()
                    basis = None
                    for k, v in self._direction.iteritems():
                        if abs(v) >= 1e-300:
                            basis = k
                            break
                    if basis is None:
                        raise ValueError("Non-normalized direction?")

                    for k, v in self._direction.iteritems():
                        if k == basis:
                            continue
                        newDir[k] = p = random.uniform(-1.0, 1.0)
                        dirLen += p * p
                        dotLen += p * self._direction[k]

                    newDir[basis] = p = -dotLen / self._direction[basis]
                    dirLen += p * p
                    dirLen = 1.0 / math.sqrt(dirLen)
                    for k in self._direction.keys():
                        newDir[k] *= dirLen
                    self._direction = newDir

            # Move along!
            self._life *= self.VITALITY_LOSS
            self._nextState = dict(self._state)
            # Only update elements in _direction
            for k, v in self._direction.iteritems():
                ax = self._axes[k]
                if ax['type'] == self.AXES_TYPES.LINEAR_RANGE:
                    self._nextState[k] = (self._nextState[k]
                            + v * self._life * ax['scale'])

                    # Limit result to axis
                    self._nextState[k] = min(ax['max'], max(ax['min'],
                            self._nextState[k]))
                elif self._axes[k]['type'] == self.AXES_TYPES.LOGARITHMIC_RANGE:
                    nlog = math.log(self._nextState[k]) + v * self._life * ax['scale']
                    # Limit in log domain and apply to axis
                    nlog = min(ax['max'], max(ax['min'], nlog))
                    self._nextState[k] = math.exp(nlog)
                elif self._axes[k]['type'] == self.AXES_TYPES.SCALAR:
                    raise ValueError("Scalar ended up in self._direction?  "
                            + k)
                else:
                    raise NotImplementedError()

        # For the actual next state, validate each member
        nextState = dict(self._nextState)
        for k, v in self._axes.iteritems():
            validator = v['validator']
            if validator is not None:
                nargs = len(inspect.getargspec(validator).args)
                if nargs == 2:
                    nextState[k] = validator(nextState[k], self._nextState)
                elif nargs == 1:
                    nextState[k] = validator(nextState[k])
                else:
                    raise ValueError("Validators must take 1 or 2 args")
        return nextState


    def _randomDirection(self):
        """Returns a random, normalized direction vector.

        Note that normalization is without respect to the range and type of
        each variable.  That is, each axis' direction will be on (-1, 1)."""
        dir = dict()
        dirLen = 0.0
        for k, v in self._axes.iteritems():
            if v['type'] == self.AXES_TYPES.SCALAR:
                continue
            elif v['type'] == self.AXES_TYPES.LINEAR_RANGE:
                dir[k] = p = random.uniform(-1.0, 1.0)
                dirLen += p * p
            elif v['type'] == self.AXES_TYPES.LOGARITHMIC_RANGE:
                dir[k] = p = random.uniform(-1.0, 1.0)
                dirLen += p * p
            else:
                raise NotImplementedError()

        if dirLen > 1e-300:
            dirLen = 1.0 / math.sqrt(dirLen)
            for k, v in dir.iteritems():
                dir[k] = v * dirLen

        return dir


class ParamRanges(object):
    class AXES_TYPES(object):
        SCALAR = "scalar"
        LINEAR_RANGE = "linear_range"
        LOGARITHMIC_RANGE = "logarithmic_range"

    @property
    def ndim(self):
        """The dimension of non-scalar components in this ParamRanges"""
        return self._ndim


    def __init__(self, paramRanges):
        self._ranges = paramRanges
        self._axes = {}
        self._ndim = 0
        for v in paramRanges:
            name = v[0]
            vrange = v[1]
            validator = None
            if len(v) == 2:
                pass
            elif len(v) == 3:
                try:
                    # This bug is because multiprocessing doesn't request a
                    # single process for e.g. frames
                    import cPickle
                    cPickle.dumps(v[2], cPickle.HIGHEST_PROTOCOL)
                except:
                    raise NotImplementedError()
                validator = v[2]
            else:
                raise ValueError("Unrecognized tuple length: {}".format(len(v)))

            # Set up this range's parameters
            if isinstance(vrange, (list, tuple)):
                if len(vrange) == 2:
                    self._axes[name] = { 'type': self.AXES_TYPES.LINEAR_RANGE,
                            'scale': (vrange[1] - vrange[0]),
                            'min': vrange[0], 'max': vrange[1] }
                elif len(vrange) == 3:
                    if vrange[2] == 'log':
                        self._axes[name] = {
                                'type': self.AES_TYPES.LOGARITHMIC_RANGE,
                                'scale': math.log(vrange[1])
                                    - math.log(vrange[0]),
                                'min': math.log(vrange[0]),
                                'max': math.log(vrange[1]) }
                    else:
                        raise NotImplementedError()
                else:
                    raise ValueError("Unrecognized length: {}".format(vrange))
            elif isinstance(vrange, (float, int, type(None))):
                # Scalar
                self._axes[name] = { 'type': self.AXES_TYPES.SCALAR,
                        'value': vrange }
            else:
                raise ValueError("Unrecognized param range: {}".format(vrange))
            self._axes[name]['validator'] = validator
            if self._axes[name]['type'] != self.AXES_TYPES.SCALAR:
                self._axes[name]['order'] = self._ndim
                self._ndim += 1


    def mapLocal(self, pt):
        """Given a coordinate dictionary in local space, return the uniform
        point corresponding to it."""
        npt = np.zeros(self.ndim)
        for k, v in self._axes.iteritems():
            if v['type'] == self.AXES_TYPES.SCALAR:
                pass
            elif v['type'] == self.AXES_TYPES.LINEAR_RANGE:
                npt[v['order']] = (pt[k] - v['min']) / v['scale']
            elif v['type'] == self.AXES_TYPES.LOGARITHMIC_RANGE:
                npt[v['order']] = (math.log(pt[k]) - v['min']) / v['scale']
            else:
                raise NotImplementedError()
        return npt


    def mapUniform(self, pt):
        """Given a coordinate of dimension self.ndim, where each component is
        on the [0, 1] range, return a coordinate in the native space of this
        point.

        Note that the input point pt is an array without labels; the result is
        a dictionary."""
        npt = {}
        for k, v in self._axes.iteritems():
            if v['type'] == self.AXES_TYPES.SCALAR:
                npt[k] = v['value']
            elif v['type'] == self.AXES_TYPES.LINEAR_RANGE:
                npt[k] = v['min'] + v['scale'] * pt[v['order']]
            elif v['type'] == self.AXES_TYPES.LOGARITHMIC_RANGE:
                npt[k] = math.exp(v['min'] + v['scale'] * pt[v['order']])
            else:
                raise NotImplementedError()

        # Apply validators
        for k, v in self._axes.iteritems():
            validator = v['validator']
            if validator is not None:
                nargs = len(inpsect.getargspec(validator).args)
                if nargs == 2:
                    npt[k] = validator(npt[k], npt)
                elif nargs == 1:
                    npt[k] = validator(npt[k])
                else:
                    raise ValueError("Validators must take 1 or 2 args")

        return npt


class Maximize(object):
    """Provides a number of ways for solving a maximization problem.
    """

    def __init__(self, nRunners = 1, nRandomSamples = 10):
        """
        :param nRunners: The number of independent samples of the objective
                function to take in parallel.
        :param nRandomSamples: The number of completely random samples of the
                search space before doing anything intelligent.
        """
        self.nRunners = nRunners
        self.nRandomSamples = nRandomSamples


    def maximizeMoe(self, paramRanges, evalJobs, solveTime = 0.0):
        """TEST!!!"""
        raise NotImplementedError("This should work, but broken right now")
        from moe.easy_interface.experiment import Experiment
        from moe.easy_interface.simple_endpoint import gp_next_points
        from moe.optimal_learning.python.data_containers import SamplePoint
        moeExp = Experiment([ [0., 1.] for _ in range(len(paramRanges)) ])
        nRandom = 2
        pts = []
        vals = []
        bval = -10.
        class FakeW(object):
            def job(self, inner):
                self._inner = inner
                return self.call
            def call(self, s):
                return self._inner(s)
        w = FakeW()
        evalJobs(w)

        simEnd = None
        if solveTime > 0.:
            simEnd = time.time() + solveTime*4.
        n = -1
        while simEnd is None or time.time() < simEnd:
            n += 1
            if n < nRandom:
                pt = np.random.uniform(size=(len(paramRanges),))
            else:
                pt = gp_next_points(moeExp)[0]
            dPt = {}
            for i in range(4):
                dPt['d{}'.format(i)] = pt[i]
            val = w.call(dPt)[1]
            pts.append(pt)
            vals.append(val)
            moeExp.historical_data.append_sample_points([SamplePoint(pt, val,
                    0.0)])
            if val > bval:
                bval = val
                print("Best: {} at {}".format(bval, pt))
        print("EVALUATED {}".format(n))


    def maximizeRbf(self, paramRanges, evalJobs, solveTime = 0.0):
        """This maximization routine is almost always superior to maximize...
        It utilizes radial basis function interpolation to predict the next
        best guess for interpolating.

        Runs a job_stream and never returns (unless solveTime is specified).
        Prints out the best params as it finds them.

        paramRanges - List of (name, range[, validator]) parameters.

                range can be any of (a, b) for a standard bound, (a, b, 'log')
                to indicate that the distribution between a and b should be
                treated as logarithmic instead of linear, or a scalar value to
                indicate something that may be parametrized in the future but
                isn't now.

                validator is a lambda with either 1 argument, the evolved
                value, or 2 arguments, the evolved value and all other (indexed
                lower than this) parameters.  The result returned from
                validator will be the actual value used.

        evalJobs(w) - Takes a job_stream.inline.Work instance, and decorates it
        with jobs that take a parameter dict and score the desired dataset,
        returning (params, avg score, std deviation).

        If configured to ever return (solveTime > 0.0), returns a
        pandas.DataFrame representing all results and data.  Otherwise,
        prints out the data as it goes."""
        from scipy.interpolate import Rbf
        # Meta-parameter: Number of random point samples to take to initialize
        # the Rbf.
        nRandom = 10

        w = Work([ None ])
        resultColumns = [ 'id' ] + [ p[0] for p in paramRanges ] + [ 'score',
                'score_dev' ]

        simEnd = None
        if solveTime > 0:
            simEnd = time.time() + solveTime

        pRanges = ParamRanges(paramRanges)

        def getNextRunner(store):
            """Given a store from the frame, return the next point to sample as
            a uniform array.
            """
            n = len(store.pts)

            # First guesses are random
            if n < nRandom:
                return pRanges.mapUniform(np.random.uniform(
                        size=(pRanges.ndim,)))

            # Build the RBF interpolator based on prior data.  Note that
            # building is O(N^3), evaluating is O(N)
            pts = np.asarray(store.pts)
            rbfArgs = [ pts[:, i] for i in range(pRanges.ndim) ]
            rbfArgs.append(np.asarray(store.vals))
            rbfi = Rbf(*rbfArgs, function='thin_plate')

            # Initial guesses
            bpts = []
            for nn in range(10):
                bpts.append(np.random.uniform(size=(pRanges.ndim,)))
            bpts = np.asarray(bpts)

            # Iterate X times following the gradient
            for gradI in range(20):
                # Current values
                bbArgs = [ bpts[:, i] for i in range(pRanges.ndim) ]
                bvals = rbfi(*bbArgs)

                d = 1e-5
                u = 0.01 * (5. / (gradI + 5.) ** 0.5)
                dvals = np.zeros((bpts.shape[0], bpts.shape[1]))
                for j in range(pRanges.ndim):
                    bbArgs = [ bpts[:, i] + (d if i == j else 0.)
                            for i in range(pRanges.ndim) ]
                    dvals[:, j] = rbfi(*bbArgs) - bvals
                dvals /= ((dvals*dvals).sum(axis=1)**0.5).reshape(
                        (dvals.shape[0], 1))
                bpts += dvals * u

            # Final evaluation, and use the best point
            bbArgs = [ bpts[:, i] for i in range(pRanges.ndim) ]
            bvals = rbfi(*bbArgs)
            bvals += np.random.uniform(size=(len(bvals),)) * (10. / (10. + n ** 0.5))
            pt = bpts[bvals.argmax()]

            return pRanges.mapUniform(pt)

        @w.init
        def printHeader():
            print(",".join(resultColumns))

        @w.frame(emit=lambda store: store.frame)
        def frameStart(store, first):
            if not hasattr(store, 'init'):
                store.init = True
                store.best = None
                store.pts = []
                store.vals = []
                store.frame = pandas.DataFrame(columns=resultColumns)

                # Emit initial runners
                runners = []
                for i in range(self.nRunners):
                    runners.append(getNextRunner(store))
                return Multiple(runners)

        evalJobs(w)

        @w.frameEnd
        def frameEnd(store, next):
            # Log the result
            q = dict(next[0])
            if 'score' in q or 'score_dev' in q or 'id' in q:
                raise ValueError("Your dimensions must not include 'score' "
                        "or 'score_dev' or 'id'")
            q['id'] = len(store.pts)
            q['score'] = next[1]
            q['score_dev'] = next[2]
            nextRow = [ q[k] for k in resultColumns ]
            store.frame.loc[len(store.pts)] = nextRow

            if store.best is None or next[1] > store.best:
                store.best = next[1]
                print(store.frame.loc[len(store.pts)])

            # Log this point to aid in the selection of future points
            pt = pRanges.mapLocal(q)
            store.pts.append(pt)
            store.vals.append(q['score'])

            # Emit another?
            if simEnd is not None and time.time() >= simEnd:
                return
            elif simEnd is None:
                # Wipe data, don't want to save it
                store.frame = pandas.DataFrame(columns=resultColumns)

            return getNextRunner(store)

        return w.run()[0]


    def maximizeHillSwarm(self, paramRanges, evalJobs, solveTime = 0.0):
        """Generic maximizer.  Runs a job_stream and never returns.  Prints out
        the best params as it finds them.

        paramRanges - List of (name, range[, validator]) parameters.

                range can be any of (a, b) for a standard bound, (a, b, 'log')
                to indicate that the distribution between a and b should be
                treated as logarithmic instead of linear, or a scalar value to
                indicate something that may be parametrized in the future but
                isn't now.

                validator is a lambda with either 1 argument, the evolved
                value, or 2 arguments, the evolved value and all other (indexed
                lower than this) parameters.  The result returned from
                validator will be the actual value used.

        evalJobs(w) - Takes a job_stream.inline.Work instance, and decorates it
        with jobs that take a parameter dict and score the desired dataset,
        returning (params, avg score, std deviation).

        If configured to ever return (solveTime > 0.0), returns a
        pandas.DataFrame representing all results and data.  Otherwise,
        prints out the data as it goes."""
        w = Work([ None ])

        resultColumns = [ 'id' ] + [ p[0] for p in paramRanges ] + [ 'score',
                'score_dev' ]

        simEnd = None
        if solveTime > 0:
            simEnd = time.time() + solveTime

        @w.init
        def printHeader():
            print(",".join(resultColumns))

        @w.frame(emit = lambda store: store.frame)
        def frameStart(store, first):
            if not hasattr(store, 'init'):
                store.init = True
                store.best = None
                store.n = 0
                store.sampleSum = 0.0
                store.sampleDev = 0.0
                store.sampleCount = 0
                store.runners = []
                store.frame = pandas.DataFrame(columns = resultColumns)
                result = []
                for i in range(self.nRunners):
                    store.runners.append(MetaRunner(i, self.nRunners,
                            paramRanges))
                    result.append(store.runners[-1].getNextTrial())

                return Multiple(result)

        evalJobs(w)

        @w.frameEnd
        def frameEnd(store, next):
            q = dict(next[0])
            q['score'] = next[1]
            q['score_dev'] = next[2]
            nextRow = [ q[k] for k in resultColumns ]

            store.n += 1

            # Integrate into our random sample floor
            if store.sampleCount < self.nRandomSamples * 10:
                if store.sampleCount > 3:
                    store.sampleDev += (next[1]
                            - (store.sampleSum / store.sampleCount)) ** 2
                store.sampleSum += next[1]
                store.sampleCount += 1

            storeAvg = store.sampleSum / store.sampleCount
            storeDev = 0.0
            if store.sampleCount > 3:
                storeDev = math.sqrt(store.sampleDev / (store.sampleCount - 3))

            # Log if we have a new best score
            store.frame.loc[q['id']] = nextRow
            if store.best is None or next[1] > store.best:
                store.best = next[1]
                print(store.frame.loc[q['id']])
                sys.stderr.write("  (random floor: {} / {})\n".format(
                        storeAvg, storeDev))

            # Record score, exit if simulation has gone on long enough
            if simEnd is not None:
                if time.time() > simEnd:
                    return
            else:
                # Wipe data, we don't want to save it
                store.frame = pandas.DataFrame(columns = resultColumns)

            # Determine the next step for this runner
            runner = store.runners[next[0]['id'] % len(store.runners)]
            if store.sampleCount < self.nRandomSamples:
                # Force a restart
                return runner.getNextTrial()

            return runner.getNextTrial(next,
                    # Required performance to start exploring an area
                    storeAvg + 1.5 * storeDev)

        return w.run()[0]


    def maximizeFit(self, learner, paramRanges, trainSet, testSet,
            maxIters = None, nSamples = 3, visualParams = None,
            imageDestFolder = None):
        if not isinstance(trainSet, tuple):
            raise ValueError("trainSet must be a tuple: (inputs, expected), or "
                    "(inputs,) for unsuperved")
        if not isinstance(testSet, tuple):
            raise ValueError("testSet must be a tuple: (inputs, expected), or "
                    "(inputs,) for unsuperved")

        # Clear out images in imageDestFolder
        if visualParams is not None:
            if imageDestFolder is None:
                raise ValueError("If visualParams is set, imageDestFolder must "
                        "be set")

            # Re-use our folder cleaning code
            fm = FigureMaker([], imageDestFolder = imageDestFolder)
        elif imageDestFolder is not None:
            raise ValueError("imageDestFolder requires visualParams")

        unsupervisedInst = sklearn.clone(learner)
        unsupervisedInst.init(len(trainSet[0]),
                len(trainSet[1][0]) if len(trainSet) > 1
                    else unsupervisedInst.nOutputs,
                1)
        def scorePipe(w):
            @w.frame(emit = lambda store: (store.params, store.score,
                    store.score_dev))
            def startParamScore(store, first):
                if not hasattr(store, 'init'):
                    store.init = True
                    store.params = first
                    store.r = []
                    store.trainedEstimators = []
                    return Multiple([first] * nSamples)

                # We're done if we reach here
                store.score = sum(store.r) / len(store.r)
                store.score_dev = 0.0
                if len(store.r) > 1:
                    for v in store.r:
                        store.score_dev += (v - store.score) ** 2
                    # We're sampling from an infinite population of samples, so
                    # use unbiased estimator of deviation
                    store.score_dev = math.sqrt(store.score_dev
                            / (len(store.r) - 1))

                # Visualize the network?
                if visualParams is not None:
                    # Find an average (representative) network
                    closeR = abs(store.r[0] - store.score)
                    closeI = 0
                    for i in range(1, len(store.r)):
                        d = abs(store.r[i] - store.score)
                        if d < closeR:
                            closeI = i
                            closeR = d

                    avgInfo = store.trainedEstimators[closeI]
                    avgTrainer, = avgInfo

                    # Round store.r[closeI] so that we don't save too many
                    # estimators
                    if avgTrainer.UNSUPERVISED:
                        # Bucketized... this is MSE, so determine a piecewise
                        # granularity function
                        v = store.r[closeI]
                        gran = 10.0 ** math.floor(math.log(v) / math.log(10) - 1.0)
                        rounded = int(store.r[closeI] / gran) * gran
                    else:
                        raise NotImplementedError()
                    imgPathBase = os.path.join(imageDestFolder,
                            "{:f}_{}".format(rounded, store.params['id']))
                    found = False
                    for saved in os.listdir(imageDestFolder):
                        if saved.startswith("{:f}_".format(rounded)):
                            found = True
                            break
                    if not found:
                        # Write some images!
                        avgTrainer.visualize(visualParams,
                                path = imgPathBase + ".png")
                        avgTrainer.visualize(visualParams,
                                path = imgPathBase + "_example.png",
                                inputs = testSet[0][0])

                # Since we're maximizing here, negate score if we're
                # reconstructing
                if (store.score is not None
                        and unsupervisedInst.UNSUPERVISED):
                    store.score *= -1.0

            @w.job
            def scoreL(parms):
                e = sklearn.clone(learner)
                pp = dict(parms)
                pp.pop('id')
                e.set_params(**pp)
                e.fit(*trainSet, maxIters = maxIters)
                score = e.score(*testSet)
                if visualParams is None:
                    return (score,)
                return (score, e)

            @w.frameEnd
            def endParamScore(store, next):
                store.r.append(next[0])
                store.trainedEstimators.append(next[1:])

        return self.maximize(paramRanges, scorePipe)
