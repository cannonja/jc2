
import os
import re
import six
import textwrap

class SpiceLibrary(object):
    """This class manages snippets of SPICE code and lets you write some that
    integrates everything else."""

    def __init__(self, spiceNetwork):
        self._sn = spiceNetwork
        self.Measure = self._sn.Measure
        self.MeasureMethod = self._sn.MeasureMethod
        self.MeasureType = self._sn.MeasureType
        # { 'name': (code, requires) }
        self._subs = {}


    def add(self, sub, reqs, src):
        if sub in self._subs:
            raise ValueError("Already have {}!".format(sub))
        src = "***** SOURCE FOR {} *****\n\n".format(sub) + textwrap.dedent(
                src)
        self._subs[sub] = (src, reqs)
        for r in reqs:
            if r not in self._subs:
                raise ValueError("Requirement {} not added".format(r))


    def addFile(self, sub, reqs, fname):
        src = self._fixMosfet(self._includeRecursive(".include " + fname))
        self.add(sub, reqs = reqs, src = src)


    def copy(self):
        """Returns a separate copy of this library."""
        other = SpiceLibrary(self._sn)
        other._subs = self._subs.copy()
        return other


    def override(self, sub, reqs, src):
        if sub not in self._subs:
            raise ValueError("No existing {}!".format(sub))
        src = "***** SOURCE FOR {} *****\n\n".format(sub) + textwrap.dedent(
                src)
        self._subs[sub] = (src, reqs)
        for r in reqs:
            if r not in self._subs:
                raise ValueError("Requirement {} not added".format(r))


    def simulate(self, measures, tsMax, measureSteps, reqs, cir,
            drawCircuitTo = None):
        """Simulates the given circuit and returns the results for the
        requested measures.  If reqs is None, attempt auto-detect.
        """
        spice = [ textwrap.dedent(cir) ]
        realReqs = []
        reqs = list(reqs)
        while reqs:
            r = reqs.pop()
            if r not in self._subs:
                raise ValueError("Bad requirement: {}".format(r))
            if r in realReqs:
                # Already added
                continue
            # Assume it goes before anything already on the list, then put it
            # after requirements.
            index = 0
            for sub in self._subs[r][1]:
                try:
                    index = max(index, realReqs.index(sub)+1)
                except ValueError:
                    # Sub not already represented
                    reqs.append(sub)
            realReqs.insert(index, r)

        for i, r in enumerate(realReqs):
            spice.insert(i, self._subs[r][0])

        spice = '\n\n'.join(spice)
        if drawCircuitTo is not None:
            try:
                graphSpice(spice, drawCircuitTo)
            except:
                sys.stderr.write("Bad spice file at /tmp/bad.spice\n")
                with open("/tmp/bad.spice", "w") as f:
                    f.write(spice)
                raise
        return self._sn.simulate(measures, tsMax, measureSteps, spice)


    def _fixMosfet(self, cir):
        """Replaces all beta0 = ... since Xyce does not support that parameter.
        """
        ncir = []
        inModel = False
        for l in cir.split('\n'):
            line = l
            if inModel:
                if len(line.strip()) > 0 and line[0] != '+' and line[0] != '*':
                    inModel = False
                else:
                    # Remove beta0 assignment
                    line = re.sub(r'beta0\s*=\s*\d+(.\d+)?\s+', '', line,
                            flags = re.I)

            if not inModel:
                if line.lower().startswith('.model'):
                    inModel = True
            ncir.append(line)
        return '\n'.join(ncir)


    def _includeRecursive(self, cir, relativeTo='.'):
        """Takes cir, a SPICE circuit, and splices in .include directives
        recursively and with proper relative pathing."""
        def doReplace(match):
            path = os.path.abspath(os.path.join(relativeTo, match.group(2)))
            if not os.path.isfile(path):
                raise ValueError("Path not found: {}".format(path))
            return self._includeRecursive(open(path).read(),
                    os.path.dirname(path))
        return re.sub(r"^\.inc(lude)? (.*)", doReplace, cir, flags = re.M)

