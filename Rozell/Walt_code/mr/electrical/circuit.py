import os
import re
import six
import subprocess
import sys
import textwrap

class _SubCircuitModels(object):
    def __init__(self, modelLines):
        self.name = "Models"
        self._src = modelLines


    def getSpice(self, name):
        return self._src


class SubCircuit(object):
    _SHELL_NAME = '__CIRCUIT__'

    @property
    def name(self):
        return self._name


    @property
    def params(self):
        return self._params


    @property
    def ports(self):
        return self._ports


    @classmethod
    def fromFile(cls, fname, *args, **kwargs):
        """Properly creates a subcircuit that encapsulates the content from the
        given file.

        This primarily means taking .MODEL statements at the top level of the
        file and distributing them into the (now) nested .SUBCKTs.
        """
        sc = SubCircuit(*args, **kwargs)
        src = cls._includeRecursive(".include " + fname).splitlines()

        # Since SPICE has really weird hierarchical behavior, .MODEL statements
        # at the global level in the original file

        # First, collapse any multi-line statements
        srcn = [ '* Encapsulated version of {}'.format(fname) ]
        # Index to collapse back to when a '+' line is found
        collapseTo = 0
        for l in src:
            srcn.append(l)
            if l.startswith('*') or not l.strip():
                # Comment / blank line
                pass
            elif l.startswith('+'):
                # Extension line
                srcn[collapseTo] = '\n'.join(srcn[collapseTo:])
                srcn = srcn[:collapseTo+1]
            else:
                collapseTo = len(srcn) - 1

        # Now, find any .MODEL statements that are top level
        models = []
        levels = []
        for l in srcn:
            if len(levels) == 0 and l.lower().startswith(".model "):
                models.append(l)
            elif l.lower().startswith(".subckt "):
                m = re.match(r".SUBCKT (\S+)", l, flags = re.I)
                if not m:
                    raise ValueError("SUBCKT invalid? {}".format(l))
                levels.append(m.group(1))
            elif l.lower().startswith(".ends"):
                m = re.match(r".ENDS (\S+)", l, flags = re.I)
                if not m:
                    raise ValueError(".ENDS MUST have subcircuit name: {}"
                            .format(l))
                if m.group(1).lower() != levels[-1].lower():
                    raise ValueError(
                            ".ENDS does not match last level? {} != {}.  {}"
                                .format(m.group(1).lower(), levels[-1].lower(),
                                    l))
                levels.pop()

        # Add global models after any .SUBCKT
        models = '\n'.join(models)
        for i in reversed(range(len(srcn))):
            if srcn[i].lower().startswith('.subckt '):
                srcn.insert(i+1, models)

        # Insert the updated source
        sc.add('\n'.join(srcn))

        return sc


    @classmethod
    def modelsFromFile(self, fname):
        """Imports the given file (fname) as a set of MODEL statements, meaning
        they will be added to any SubCircuit with this SubCircuit in its
        depends list."""
        src = self._includeRecursive(".include " + fname)
        return _SubCircuitModels(self._fixMosfet(src))


    def __init__(self, name, ports, params={}, depends=[]):
        """Defines a .SUBCKT block, with dependencies and parameters.

        name - (string) Name of subcircuit.  Same as in spice, but may be
                suffixed to get rid of ambiguity.

        ports - ([string,]) - The names of ports that this SubCircuit interacts
                with.

        params - { name: default } - The PARAMS elements of the SUBCKT block.

        depends - ([object,]) - SubCircuits or Models that this SubCircuit
                depends on IN TEXT FORM.  These will be embedded in this
                SubCircuit specifically, and cannot be shared (since we don't
                parse the names out).
        """
        self._name = name
        self._ports = ports
        self._params = params
        self._depends = depends
        self._source = []
        self._localNameCounts = {}


    def __call__(self, ports, **params):
        return SubCircuitInstance(self, ports, params)


    def add(self, obj):
        """Adds to this SUBCKT's netlist.  Behavior depends on type of obj:

        string - SPICE code added directly to netlist

        SubCircuit - Sets a dependency for SPICE code specified by string; that
                is, the name of this SubCircuit within this context must
                exactly match.

        SubCircuitInstance - Adds the SPICE code for the given instance of the
                given SubCircuit.
        """
        if isinstance(obj, six.string_types):
            self._source.append(textwrap.dedent(obj))
            self._source.append("\n")
        elif isinstance(obj, SubCircuit):
            self._depends.append(obj)
        elif isinstance(obj, SubCircuitInstance):
            self._source.append(obj)
            self._source.append("\n")
            obj.name = self._getLocalName('X{}_inst'.format(
                    obj.subcircuit.name))
        elif isinstance(obj, _SubCircuitModels):
            self._depends.append(obj)
        else:
            raise NotImplementedError(repr(obj))


    def getSpice(self, name, recursive=None):
        """Gets the SPICE code for this subcircuit with the given name."""
        # Resolve body
        if recursive is not None:
            subs, names = recursive
            parentSubs = subs
            subs = subs.copy()
            names = names.copy()
        else:
            parentSubs = set()
            subs = set()
            names = {}
        recursive = (subs, names)

        body = []
        body.append('*' * 79)
        body.append('\n')
        body.append('* Circuit')
        body.append('\n')
        body.append('*' * 79)
        body.append('\n')
        for s in self._source:
            if isinstance(s, six.string_types):
                body.append(s)
            elif isinstance(s, SubCircuitInstance):
                if s.subcircuit not in subs:
                    # Not already available at this level.
                    if s.subcircuit in names:
                        raise ValueError("{} in names?".format(s.subcircuit))
                    subs.add(s.subcircuit)
                    names[s.subcircuit] = scName = s.subcircuit.name
                else:
                    scName = names[s.subcircuit]
                body.append(s.getSpice(scName))
            else:
                raise NotImplementedError(s)

        body.append('*' * 79)
        body.append('\n')
        body.append('* Dependencies')
        body.append('\n')
        body.append('*' * 79)
        body.append('\n')
        for s in subs:
            if s in parentSubs:
                continue
            body.append(s.getSpice(names[s], recursive=recursive))

        body.append('*' * 79)
        body.append('\n')
        body.append('* Hard-coded dependencies')
        body.append('\n')
        body.append('*' * 79)
        body.append('\n')
        for s in self._depends:
            body.append(s.getSpice(s.name))
        body.append('\n')

        if name == SubCircuit._SHELL_NAME:
            return ''.join(body)

        parms = ''
        if self._params:
            parms = 'PARAMS: ' + ' '.join('{}={}'.format(k, v)
                    for k, v in self._params.iteritems())
        return ".SUBCKT {} {} {}\n{}\n.ENDS {}\n\n".format(name,
                ' '.join(self.ports), parms, ''.join(body), name)


    @classmethod
    def _fixMosfet(cls, cir):
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


    def _getLocalName(self, baseName):
        """Returns a unique local name based on the given baseName."""
        i = 1
        if baseName not in self._localNameCounts:
            self._localNameCounts[baseName] = i
        else:
            self._localNameCounts[baseName] += 1
            i = self._localNameCounts[baseName]
        return '{}_{}'.format(baseName, i) if i > 1 else baseName


    @classmethod
    def _includeRecursive(cls, cir, relativeTo='.'):
        """Takes cir, a SPICE circuit, and splices in .include directives
        recursively and with proper relative pathing."""
        def doReplace(match):
            path = os.path.abspath(os.path.join(relativeTo, match.group(2)))
            if not os.path.isfile(path):
                raise ValueError("Path not found: {}".format(path))
            return cls._includeRecursive(open(path).read(),
                    os.path.dirname(path))
        return re.sub(r"^\.inc(lude)? (.*)", doReplace, cir, flags = re.M)




class SubCircuitInstance(object):
    """A particular realization of a subcircuit.  When added to a SubCircuit,
    is given a locally-unique name (SUBCKT name + _inst)."""

    @property
    def name(self):
        if self._name is None:
            raise ValueError("SubCircuitInstance has not yet been added to "
                    "a circuit!")
        return self._name


    @name.setter
    def name(self, value):
        if self._name is not None:
            raise ValueError("SubCircuitInstance cannot have name changed.  "
                    "Already added to another circuit?")
        self._name = value


    class NameConcat(object):
        """Class that generates hierarchical names for parts of a Circuit."""
        def __init__(self, name):
            self._name = name
        def __getitem__(self, name):
            return '{}.{}'.format(self._name, name)
    @property
    def internal(self):
        """internal is used to generate hierarchical names for internal parts
        of the circuit."""
        return self.NameConcat(self.name)


    def __init__(self, subcircuit, ports, params):
        self.subcircuit = subcircuit
        self.ports = ports
        self.params = params.copy()
        self._name = None

        if len(self.ports) != len(self.subcircuit.ports):
            raise ValueError("Bad number of ports: {} != {}".format(
                    len(self.ports), len(self.subcircuit.ports)))
        for k in self.params.iterkeys():
            if k not in self.subcircuit.params:
                raise ValueError("Unrecognized parameter: {}".format(k))


    def getSpice(self, subcircuitName):
        """Returns the SPICE for this instance."""
        parms = ''
        if self.params:
            parms = 'PARAMS: ' + ' '.join(
                    '{}={}'.format(k, v) for k, v in self.params.iteritems())
        return "{} {} {} {}".format(self.name, ' '.join(self.ports),
                subcircuitName, parms)



class Circuit(object):
    class Measure(object):
        """A value that will be sampled from the SPICE simulation."""

        def __init__(self, node, valueType = "voltage",
                method = "sample"):
            self.method = method
            self.valueType = valueType
            self.node = node
    class MeasureMethod(object):
        """The method used to process the sampled values for a
        :class:`Measure`.
        """
        AVERAGE = "avg"
        """Returns a single value, which is the average across the whole
        simulation.
        """
        SAMPLE = "sample"
        """Returns a list of values, evenly spaced throughout the simulation.
        The number of samples is determined by the value of tsStep relative
        to tsMax.
        """
    class MeasureType(object):
        VOLTAGE = "voltage"
        """Default :class:`MeasureType`, measures the voltage of a single node.
        """
        POWER = "power"
        """If POWER is being measured, then node must be comma-separated:
        "voltage-source,positiveTerminal".  Note that the measured power is
        negative for using power, positive for absorbing it.
        """


    def __init__(self, spiceNetwork, tsMax, steps):
        self._circuit = [ '* Circuit from mr.electrical\n' ]
        # Supporting subcircuits / etc.  Put in file after the non-dependent
        # stuff.
        self._spice = spiceNetwork
        self._tsMax = tsMax
        self._steps = steps

        self._subcircuit = SubCircuit(SubCircuit._SHELL_NAME, [], [])
        self.add = self._subcircuit.add


    def draw(self, fname):
        """Draws (using dot) a graph that represents this circuit."""
        spice = "Circuit\n{}".format(self._subcircuit.getSpice(
                self._subcircuit.name))
        try:
            _graphSpice(spice, fname)
        except:
            sys.stderr.write("Bad spice file at /tmp/bad.spice\n")
            with open("/tmp/bad.spice", "w") as f:
                f.write(spice)
            raise


    def run(self, measArgs):
        meas = {}
        if isinstance(measArgs, dict):
            meas = measArgs
        elif isinstance(measArgs, list):
            for a in measArgs:
                if isinstance(a, six.string_types):
                    if ',' in a:
                        device, node = a.split(',', 1)
                        meas['pwr_{}'.format(device)] = self._spice.Measure(
                                a, self._spice.MeasureType.POWER)
                    else:
                        meas[a] = self._spice.Measure(a)
                else:
                    raise NotImplementedError(a)
        else:
            raise NotImplementedError(measArgs)

        return self._spice.simulate(meas, self._tsMax, self._steps,
                'Circuit\n' + self._subcircuit.getSpice(self._subcircuit.name))



def _graphSpice(spice, fname):
    """Utility function to draw a picture of the given spice source."""
    if not fname.endswith(".pdf"):
        raise ValueError("Filename must end with .pdf: {}".format(fname))
    dot = [ "graph blah {\n",
            "rankdir=LR;\n",
            "splines=true;\n",
            'sep="1.0";\n',
            'esep="0.3";\n',
            'overlap="200:compress";\n',
            #"maxiter=100000;\n",
            "concentrate=true;\n",
            ]

    seenDev = set()
    subs = [ '' ]
    subToNodes = {}
    def n(v):
        return subs[-1] + v
    def subStart(name, nodes):
        subs.append(name + '_')
        dot.append('subgraph cluster{} {{\n'.format(name))
        dot.append('label="{}"\n'.format(name))
        dot.append('style="filled"\ncolor=lightgray;\n')
        for i, n in enumerate(nodes):
            if n.lower() == 'params:':
                nodes = nodes[:i]
                break
        subToNodes[name.lower()] = nodes
    def subEnd():
        subs.pop()
        dot.append('}\n')
    def addDevice(name, **spec):
        name = n(name)
        if name in seenDev:
            raise ValueError("{} already defined".format(name))
        seenDev.add(name)
        specArgs = ' '.join([ '{}="{}"'.format(k, v)
                for k, v in spec.iteritems() ])
        dot.append('{} [{}];\n'.format(name, specArgs))
    def addEdge(a, b, **spec):
        a = n(a)
        b = n(b)
        specArgs = ' '.join([ '{}="{}"'.format(k, v)
                for k, v in spec.iteritems() ])
        dot.append('{} -- {} [{}];\n'.format(a, b, specArgs))

    thisSkipped = False
    for i, l in enumerate(spice.splitlines()):
        # Keep track of if next line should be logged if it starts with '+'
        lastSkipped = thisSkipped
        thisSkipped = False
        if i == 0:
            # First line is comment line
            thisSkipped = True
            continue
        elif not l.strip() or l.strip().startswith("*"):
            # Comment line
            # Does not play well with "+" lines (meaning a "+" line resumes the
            # previous non-comment line)
            thisSkipped = lastSkipped
            continue

        l = l.split()
        if l[0][0].lower() == 'r':
            assert len(l) == 4
            addDevice(l[0], label="{} ({})".format(n(l[0]), l[3]),
                    shape="rect")
            addEdge(l[1], l[0])
            addEdge(l[2], l[0])
        elif l[0][0].lower() == 'v':
            assert len(l) >= 4
            addDevice(l[0], label="{{<p>+|{} ({})|<m>-}}".format(n(l[0]),
                    l[3]), shape="Mrecord")
            addEdge(l[1], l[0] + ":p")
            addEdge(l[2], l[0] + ":m")
        elif l[0][0].lower() == 'c':
            assert len(l) >= 4
            addDevice(l[0], label="{} ({})".format(n(l[0]), l[3]),
                    shape="rect")
            addEdge(l[1], l[0])
            addEdge(l[2], l[0])
        elif l[0][0].lower() == 'm':
            # MOSFET
            assert len(l) >= 6
            addDevice(l[0], label="{{<d>D|{{<g>G|{} {}|<b>B}}|<s>S}}".format(
                    n(l[0]), ' '.join(l[5:])), shape="Mrecord")
            addEdge(l[1], l[0] + ':d:w')
            addEdge(l[2], l[0] + ':g:n')
            addEdge(l[3], l[0] + ':s:e')
            addEdge(l[4], l[0] + ':b:s')
        elif l[0][0].lower() == 'x':
            # X - arbitrary subcircuit / device
            for li in range(len(l)):
                if l[li].lower() == 'params:':
                    l = l[:li]
                    break
            subName = l[-1]
            if subName.lower() not in subToNodes:
                raise ValueError("Unknown SUBCKT: {}".format(subName))
            connNames = subToNodes[subName.lower()]
            conns = l[1:-1]
            if len(conns) != len(connNames):
                raise ValueError("Bad SUBCKT? {} requires {}, found {}".format(
                        subName, connNames, conns))
            addDevice(l[0], label="{}|{{{}}}".format(l[0], '|'.join([
                    "<n{}>{}".format(ci, cn) for ci, cn in enumerate(connNames)
                    ])), shape="Mrecord")
            for ci, c in enumerate(conns):
                addEdge(c, l[0] + ':n{}:s'.format(ci))
        elif l[0].lower() == '.subckt':
            subStart(l[1], l[2:])
        elif l[0].lower() == '.ends':
            subEnd()
        elif l[0][0] == '+' and lastSkipped:
            # Keep on skipping
            thisSkipped = True
        else:
            print("**** Skipping: {}".format(l))
            thisSkipped = True

    dot.append("}")
    dot = ''.join(dot)
    # print("Graphing via dot:\n{}\n\n->\n\n{}".format(spice, dot))

    # fdp
    p = subprocess.Popen(['dot', '-Kfdp', '-Tpdf', '-o{}'.format(fname)],
            stdin = subprocess.PIPE, stdout = subprocess.PIPE,
            stderr = subprocess.PIPE)
    out, err = p.communicate(dot)
    r = p.wait()
    if r != 0:
        print("------ dot output ------\n{}\n------ dot error ------\n{}"
                .format(out, err))
        raise ValueError("dot returned {}: {}".format(r, err))


