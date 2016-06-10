
from .circuit import Circuit

import job_stream
import numpy as np
import os
import re
import sys
import tempfile

class SpiceNetwork(object):
    """Runs an electrical simulation through a Spice binary, such a XYCE."""

    Measure = Circuit.Measure
    MeasureType = Circuit.MeasureType
    MeasureMethod = Circuit.MeasureMethod

    class SpiceType(object):
        XYCE = "xyce"
        NGSPICE = "ngspice"


    def __init__(self, spicePath, spiceType, uic=False):
        self._spicePath = spicePath
        self._spiceType = spiceType
        self._uic = uic

        # Ngspice version
        if self._spiceType == self.SpiceType.NGSPICE:
            out = job_stream.invoke([ self._spicePath, '--version' ])[0]
            m = re.search("^ngspice compiled.*revision ([0-9]+)$", out, re.M)
            if m is None:
                raise ValueError("Could not find ngspice revision in:\n{}"
                        .format(out))
            self._ngspiceVersion = int(m.group(1))


    def Circuit(self, tsMax, steps=101):
        """Creates a new Circuit object for transient simulation which can be
        modularized a bit.

        tsMax - The time to run transient simulation for.

        steps - The number of datapoints to collect, including time t=0 and
                t=tsMax.
        """
        return Circuit(self, tsMax, steps)


    def simulate(self, measures, tsMax = 1.0, measureSteps = 11, circuit = None):
        """Simulates the given circuit for the given amount of time, measuring
        the given measures.

        measures - Either a list or a dict, where the values are of type
                SpiceNetwork.Measure.
        """
        if circuit is None:
            raise ValueError("Must specify circuit")
        if not isinstance(measureSteps, int):
            raise ValueError("measureSteps must be int: {}".format(
                    measureSteps))

        # Convert measures
        names = None
        meas = measures
        if isinstance(measures, list):
            pass
        elif isinstance(measures, dict):
            names = measures.keys()
            meas = [ measures[k] for k in names ]
        else:
            raise ValueError("Unrecognized measures list/dict: {}".format(
                    measures))

        network = []
        network.append(circuit)
        network.append("\n")

        try:
            if not isinstance(measureSteps, int):
                raise ValueError("measureSteps not int")

            if self._spiceType == self.SpiceType.XYCE:
                result = self._runXyce(network, tsMax, measureSteps, meas)
            elif self._spiceType == self.SpiceType.NGSPICE:
                result = self._runNgspice(network, tsMax, measureSteps, meas)
            else:
                raise ValueError("Unrecognized spice type: {}".format(
                        self._spiceType))

            result = [ r if not isinstance(r, list) else np.asarray(r)
                    for r in result ]

            if names is not None:
                result = { n: result[i] for i, n in enumerate(names) }
                result['time'] = np.linspace(0., tsMax, measureSteps)
            return result
        except:
            # Note!  Network has been modified by _runXyce or other _run
            # function.  Therefore, bad.spice gets the complete circuit.
            with open(os.path.join(tempfile.gettempdir(), 'bad.spice'), 'w') as f:
                f.write(''.join([ str(s) for s in network ]))
                print("Bad spice file at '{}'".format(f.name))
            raise


    def _runSpiceBinary(self, network, extraArgs, extraFiles):
        """Takes extraArgs to call _spicePath with, and returns the
        line-delimited output of the process.  Also available are additional
        transient errors, and files to delete after execution.

        network - list of character snippets (joined by '') comprising the
                netlist to execute.  Will be written to temporary file and
                appended as final argument (after extraArgs).

        extraArgs - Args between self._spicePath and the temporary file
                containing netlist.

        extraFiles - Files potentially created by the simulation.  They
                will be requested to be deleted.  If callable, should take the
                temporary file as an argument and return the path to the
                extraneous temporary to be deleted.
        """
        f = tempfile.NamedTemporaryFile()
        try:
            f.write(''.join([ str(s) for s in network ]))
            f.flush()

            args = [ self._spicePath ] + list(extraArgs) + [ f.name ]
            out = job_stream.invoke(args,
                    [ "Error: Unable to create the sub-directory" ])[0]
            return out.split("\n")
        finally:
            for e in extraFiles:
                if callable(e):
                    self._safeDelete(e(f.name))
                else:
                    self._safeDelete(e)
            # Deletes it
            f.close()


    def _runNgspice(self, network, tsMax, measureSamples, measures):
        ## Request the appropriate, transient simulation
        tsStep = tsMax / (measureSamples - 1)
        network.append(".PRINT TRAN FORMAT=NOINDEX\n")
        # A small bonus time is added to the computed max so that ngspice
        # never complains about a measurement being out of bounds.
        network.extend([ ".TRAN ", tsStep, " ", tsMax + tsStep, " ",
                "UIC" if self._uic else "", "\n" ])

        ## Put measures on network
        measureThings = []
        # Tuples of (name, thing)
        thingsAvg = []
        thingsSample = []
        sampleTimes = np.linspace(0., tsMax, measureSamples)
        for i, m in enumerate(measures):
            # Remember, ngspice returns stuff as lower-case
            name = "m_{}".format(i)
            things = []
            measureThings.append(things)

            if m.valueType == self.MeasureType.VOLTAGE:
                things.append(("{}_v".format(name),
                        "V({})".format(m.node.lower())))
            elif m.valueType == self.MeasureType.POWER:
                n1, n2 = [ n.strip().lower() for n in m.node.split(',') ]
                things.append(("{}_i".format(name),
                        "I({})".format(n1)))
                things.append(("{}_v".format(name),
                        "V({})".format(n2)))
            else:
                raise NotImplementedError(m.valueType)

            if m.method == self.MeasureMethod.AVERAGE:
                thingsAvg.extend(things)
            elif m.method == self.MeasureMethod.SAMPLE:
                thingsSample.extend(things)
            else:
                raise NotImplementedError(m.method)

        for name, thing in thingsAvg:
            network.append(".MEASURE TRAN {} AVG {}\n".format(
                    name, thing))
        for name, thing in thingsSample:
            for sti, st in enumerate(sampleTimes):
                if st == 0.0:
                    st = 1e-30
                network.append(".MEASURE TRAN {} FIND {} AT={}\n".format(
                        "{}_{}".format(name, sti), thing, st))

        ## Run simulation
        out = self._runSpiceBinary(network, [ '-ab' ], [ 'bsim4v4.out' ])

        ## Process output
        outs = {}
        outFinder = re.compile(
                r"([a-zA-Z0-9_]+) += +(-?[0-9]+(\.[0-9]*)?(e[+-][0-9]+)?)")

        # We record between " +Transient Analysis" and
        # "elapsed time since last"
        recording = False
        if self._ngspiceVersion == 24:
            headerFinder = re.compile(r"^ +Transient Analysis$", re.M)
            footerFinder = re.compile(r"^elapsed time since last call", re.M)
        elif self._ngspiceVersion == 26:
            headerFinder = re.compile(r"^  Measurements for Transient Analysis", re.M)
            footerFinder = re.compile(r"^CPU time since last call", re.M)
        else:
            raise NotImplementedError("Ngspice v{}".format(
                    self._ngspiceVersion))
        for iline, line in enumerate(out):
            if not recording:
                if headerFinder.match(line) is not None:
                    recording = True
            else:
                line = line.strip()
                m = outFinder.match(line)
                if m is not None:
                    outs[m.group(1)] = float(m.group(2))
                elif footerFinder.match(line) is not None:
                    break
                elif not line:
                    pass
                else:
                    # Error!
                    raise ValueError("Unrecognized lines in Transient "
                            "Analysis output:\n{}".format(
                                '\n'.join(out[iline:])))
        else:
            raise ValueError("Never found 'Transient Analysis' line? {}"
                    .format('\n'.join(out)))

        # Process outs into appropriate output stream
        results = []
        try:
            for i, m in enumerate(measures):
                things = measureThings[i]
                if m.method == self.MeasureMethod.AVERAGE:
                    sthings = [ [ outs[n] ] for n, _ in things ]
                elif m.method == self.MeasureMethod.SAMPLE:
                    sthings = [ [ outs[n + '_{}'.format(j)]
                            for j in range(len(sampleTimes)) ] for n, _ in things ]
                else:
                    raise NotImplementedError(m.method)

                # Now, sthings is of [ thing [ samples ] ] form.
                if m.valueType == self.MeasureType.VOLTAGE:
                    result = np.asarray(sthings[0])
                elif m.valueType == self.MeasureType.POWER:
                    result = np.asarray(sthings[0]) * np.asarray(sthings[1])
                else:
                    raise NotImplementedError(m.valueType)

                # result = np.asarray([ samples ]) of result.
                if m.method == self.MeasureMethod.AVERAGE:
                    results.append(result.mean())
                elif m.method == self.MeasureMethod.SAMPLE:
                    results.append(list(result))
                else:
                    raise NotImplementedError(m.method)
        except KeyError, e:
            sys.stderr.write("Error: {} not found in output.  Output:\n\n{}"
                    .format(e, '\n'.join(out)))
            raise KeyError("{} (not found in {})".format(e, outs))

        return results


    def _runXyce(self, network, tsMax, measureSamples, measures):
        # Request the appropriate, transient simulation
        tsStep = tsMax / (measureSamples - 1)
        network.append(".PRINT TRAN FORMAT=NOINDEX\n")
        network.extend([ ".TRAN 1ms ", tsMax, " 0 ", tsStep, " ",
                "UIC" if self._uic else "", "\n" ])

        # Put measures on network
        measureNames = []
        for i, m in enumerate(measures):
            name = "M_{}".format(i)
            measureNames.append(name)
            nSamples = 0
            getSampleArgs = lambda i: ""
            if m.method == self.MeasureMethod.AVERAGE:
                smethod = "AVG"
            elif m.method == self.MeasureMethod.SAMPLE:
                # EQN takes value at last time step, rather than average
                smethod = "EQN"
                nSamples = measureSamples
                tsStep = tsMax / (nSamples - 1)
                def getSampleArgs(n):
                    if n == 0:
                        return "FROM=0 TO=0 "
                    elif n == nSamples - 1:
                        return "FROM={} TO={}".format(tsMax, tsMax)
                    return "FROM={} TO={} ".format(
                            tsStep * (n - 1), tsStep * n)
            else:
                raise NotImplementedError("Method: {}".format(m.method))

            for n in range(max(1, nSamples)):
                sname = name
                if nSamples > 0:
                    sname = "{}__{}".format(sname, n)
                if m.valueType == self.MeasureType.VOLTAGE:
                    stype = "V({})".format(m.node)
                elif m.valueType == self.MeasureType.POWER:
                    currentSrc, voltageSrc = m.node.split(',')
                    network.append(".MEASURE TRAN {} {} I({}) {}\n".format(
                            "__tmp__" + sname, smethod, currentSrc, getSampleArgs(n)))
                    stype = "V({})".format(voltageSrc)
                else:
                    raise NotImplementedError("Type: {}".format(m.valueType))

                network.append(".MEASURE TRAN {} {} {} {}\n".format(
                        sname, smethod, stype, getSampleArgs(n)))

        # Get the output from executing the netlist.
        out = self._runSpiceBinary(network, [ '-linsolv', 'klu' ],
                [
                    lambda f: f + ".prn",
                    lambda f: f + ".mt0" ])

        # Take out, the stdout of the spice process, and process our measures
        results = [ None ] * len(measures)
        seenMeasures = False
        seenCount = 0
        oldV = None
        for line in out:
            line = line.strip()
            if not seenMeasures:
                if line == "***** Measure Functions *****":
                    seenMeasures = True
            elif line.startswith("***** Total Simulation Solvers Run Time"):
                break

            delim = line.split(" = ")
            if len(delim) == 1:
                # Not an equivalence line
                continue

            m, v = [ s.strip() for s in delim ]
            for i, name in enumerate(measureNames):
                rName = m.split('__')[0]
                if name == rName:
                    if measures[i].method == self.MeasureMethod.SAMPLE:
                        if results[i] is None:
                            results[i] = []
                            seenCount += 1
                    elif results[i] is not None:
                        raise ValueError("Measure {} seen twice".format(i))

                    if measures[i].valueType == self.MeasureType.VOLTAGE:
                        val = float(v)
                    elif measures[i].valueType == self.MeasureType.POWER:
                        # Current was measurement before this one, always
                        val = float(v) * float(oldV)
                    else:
                        raise NotImplementedError("Measure type: {}".format(
                                measures[i].valueType))

                    if measures[i].method == self.MeasureMethod.AVERAGE:
                        results[i] = val
                        seenCount += 1
                    elif measures[i].method == self.MeasureMethod.SAMPLE:
                        results[i].append(val)
                    else:
                        raise NotImplementedError(measures[i].method)

                    break

            oldV = v

        if seenCount != len(results):
            raise ValueError("Some measures never seen: {}".format(results))
        return results



    def _safeDelete(self, path):
        try:
            os.remove(path)
        except OSError, e:
            # Allow file not found
            if e.errno != 2:
                raise
