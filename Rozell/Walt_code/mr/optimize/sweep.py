
from mr.figureMaker import FigureMaker

import collections
from job_stream.inline import Multiple, Work
import math
import os
import pandas
import sklearn
import sys
import traceback

class Sweep(object):
    """Provides facilities for testing a range of parameters uniformly.
    """
    def __init__(self):
        pass


    def sweep(self, paramRanges, nSamples, scoreParams, output = None,
            checkpointFile = None):
        """Generate sweeper.  Runs a job_stream on the given paramRanges,
        sampling scoreParams() nSamples times for each parameter set.

        paramRanges - [ ('name', [ values, ... ]), ... ]

        nSamples - Number of samples for each parameter set

        scoreParams - Function to evaluate a parameter set.  Takes parameter set
                including 'id'.  Returns a dict with keys as attributes, and
                values as a single, floating point number.

                Average and standard deviation are calculated.

        output - Either None to output to stdout, a string to save to the given
                file (default type csv), or a function that takes an array of
                all of the row dicts that would go in a csv.
        """
        # Generate sets of parameters to score
        parmSets = []
        nParms = len(paramRanges)
        stack = [ 0 ] * nParms
        carry = 0
        while carry == 0:
            parms = { 'id': len(parmSets) }
            parmSets.append(parms)
            for i, (name, vals) in enumerate(paramRanges):
                parms[name] = vals[stack[i]]

            # Increment and cascade
            carry = 1
            for i in range(nParms - 1, -1, -1):
                if carry == 0:
                    break
                stack[i] += carry
                if stack[i] >= len(paramRanges[i][1]):
                    stack[i] = 0
                    carry = 1
                else:
                    carry = 0

        with Work(parmSets, checkpointFile = checkpointFile) as w:
            @w.frame(emit = lambda store: store.result)
            def gatherScores(store, first):
                if not hasattr(store, 'init'):
                    store.init = True
                    store.id = first['id']
                    store.first = first
                    store.data = []
                    return Multiple([ first ] * nSamples)

                # We're done!  Calculate averages and such
                avgs = collections.defaultdict(float)
                devs = collections.defaultdict(float)
                for d in store.data:
                    for k, v in d.iteritems():
                        avgs[k] += v
                for k in avgs.keys():
                    avgs[k] /= len(store.data)
                if len(store.data) > 1:
                    for d in store.data:
                        for k, v in d.iteritems():
                            devs[k] += (v - avgs[k]) ** 2
                    for k in devs.keys():
                        devs[k] = (devs[k] / (len(store.data) - 1)) ** 0.5

                store.result = store.first
                for k, v in avgs.iteritems():
                    store.result[k] = v
                    store.result[k + '_dev'] = devs[k]
                sys.stderr.write("...Finished {}\n".format(store.id))

            @w.job
            def scoreSet(parms):
                return scoreParams(parms)

            @w.frameEnd
            def aggScores(store, next):
                store.data.append(next)

            @w.finish
            def saveResults(r):
                resultColumns = [ 'id' ] + [ p[0] for p in paramRanges ]
                for key in sorted(r[0].keys()):
                    if key not in resultColumns:
                        resultColumns.append(key)
                df = pandas.DataFrame(r, columns = resultColumns)
                df.set_index('id', inplace = True)
                df.sort_index(inplace = True)
                print(df.to_string())

                if output is not None:
                    if isinstance(output, str):
                        df.to_csv(output)
                    else:
                        raise NotImplementedError(output)


    def sweepFit(self, learner, paramRanges, trainSet, testSet,
            maxIters = None, nSamples = 3, scoreModel = None, output = None,
            visualParams = None, imageDestFolder = None, checkpointFile = None):
        """Special version of sweep() for the common use case.  Also can
        output visual information, as needed.

        scoreModel - Either None to just return a dict of model.score(*testSet),
                or a function that takes (model, testSet) and returns a dict
                with parameters to track.
        """
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

        # The method to score a param set
        def score(parms):
            e = sklearn.clone(learner)
            pp = dict(parms)
            pp.pop('id')
            try:
                e.set_params(**pp)
                e.fit(*trainSet, maxIters = maxIters)

                if visualParams is not None:
                    imgPathBase = os.path.join(imageDestFolder,
                            "{}".format(store.params['id']))

                    # Write some images!
                    e.visualize(visualParams,
                            path = imgPathBase + ".png")
                    e.visualize(visualParams,
                            path = imgPathBase + "_example.png",
                            inputs = testSet[0][0])

                if scoreModel is None:
                    return dict(score = e.score(*testSet))
                else:
                    return scoreModel(e, testSet)
            except:
                sys.stderr.write("Error for {}:\n{}\n".format(
                        parms, traceback.format_exc()))
                if e.UNSUPERVISED:
                    score = 1.0
                else:
                    score = -1.0
                e = None

        return self.sweep(paramRanges, nSamples, score, output = output,
                checkpointFile = checkpointFile)
