import datetime
import os
import random

import binutil  # required to import from dreamcoder modules

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.domains.list.listPrimitives import RecursionDepthExceeded
from dreamcoder.grammar import Grammar
from dreamcoder.program import Primitive, Program
from dreamcoder.task import Task
from dreamcoder.type import arrow, tint, tboolean, tlist, t0, t1, baseType, TypeConstructor
from dreamcoder.utilities import numberOfCPUs

# Primitives
tunit = baseType("unit")
def tstatem(t): return TypeConstructor("statem", [t])

def _if(c): return lambda t: lambda f: t if c else f
def _and(x): return lambda y: x and y
def _or(x): return lambda y: x or y
def _eq0(x): return x == 0
def _gt0(x): return x > 0
def _lt0(x): return x < 0
def _abs(x): return abs(x)

_knight0 = lambda xy: ((xy[0]+2, xy[1]+1), tuple())
_knight1 = lambda xy: ((xy[0]+1, xy[1]+2), tuple())
_knight2 = lambda xy: ((xy[0]-1, xy[1]+2), tuple())
_knight3 = lambda xy: ((xy[0]-2, xy[1]+1), tuple())
_knight4 = lambda xy: ((xy[0]-2, xy[1]-1), tuple())
_knight5 = lambda xy: ((xy[0]-1, xy[1]-2), tuple())
_knight6 = lambda xy: ((xy[0]+1, xy[1]-2), tuple())
_knight7 = lambda xy: ((xy[0]+2, xy[1]-1), tuple())

_mnop = lambda xy: (xy, tuple())
_readx = lambda xy: (xy, xy[0])
_ready = lambda xy: (xy, xy[1])
_decrx = lambda xy: ((xy[0]-1, xy[1]), tuple())
_incrx = lambda xy: ((xy[0]+1, xy[1]), tuple())
_decry = lambda xy: ((xy[0], xy[1]-1), tuple())
_incry = lambda xy: ((xy[0], xy[1]+1), tuple())
def _mbind(m1): 
    def inner(m2): 
        def monad(state0):
            (state1, result1) = m1(state0)
            return m2(result1)(state1)
        return monad
    return inner
def _mrepeat(i):
    def inner(m): 
        def monad(s):
            p = (s, tuple())
            for i in range(i):
                p = m(p[0])
            return p
        return monad
    return inner
def _mrun(m):
    def inner(i):
        return m(i)[0]
    return inner

if __name__ == "__main__":

    # Options more or less copied from list.py

    args = commandlineArguments(
        enumerationTimeout=10, activation='tanh',
        iterations=10, recognitionTimeout=3600,
        a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
        helmholtzRatio=0.5, structurePenalty=1.,
        CPUs=numberOfCPUs())

    timestamp = datetime.datetime.now().isoformat()
    outdir = 'experimentOutputs/demo/'
    os.makedirs(outdir, exist_ok=True)
    outprefix = outdir + timestamp
    args.update({"outputPrefix": outprefix})

    # Create list of primitives
    primitives = [
        Primitive("if", arrow(tboolean, t0, t0, t0), _if),
        Primitive("and", arrow(tboolean, tboolean, tboolean), _and),
        Primitive("or", arrow(tboolean, tboolean, tboolean), _or),
        Primitive("eq0", arrow(tint, tboolean), _eq0),
        Primitive("gt0", arrow(tint, tboolean), _gt0),
        Primitive("lt0", arrow(tint, tboolean), _lt0),

        Primitive("abs", arrow(tint, tint), _abs),
        
        Primitive("readx", tstatem(tint), _readx),
        Primitive("ready", tstatem(tint), _ready),
        #Primitive("decrx", tstatem(tunit), _decrx),
        #Primitive("incrx", tstatem(tunit), _incrx),
        #Primitive("decry", tstatem(tunit), _decry),
        #Primitive("incry", tstatem(tunit), _incry),
        Primitive("knight0", tstatem(tunit), _knight0),
        Primitive("knight1", tstatem(tunit), _knight1),
        Primitive("knight2", tstatem(tunit), _knight2),
        Primitive("knight3", tstatem(tunit), _knight3),
        Primitive("knight4", tstatem(tunit), _knight4),
        Primitive("knight5", tstatem(tunit), _knight5),
        Primitive("knight6", tstatem(tunit), _knight6),
        Primitive("knight7", tstatem(tunit), _knight7),
        Primitive("mnop", tstatem(tunit), _mnop),
        Primitive("mbind", arrow(tstatem(t0), arrow(t0, tstatem(t1)), tstatem(t1)), _mbind),
        Primitive("mrepeat", arrow(tint, tstatem(tunit), tstatem(tunit)), _mrepeat),
        Primitive("mrun", arrow(tstatem(t0), tlist(tint), tlist(tint)), _mrun),
    ]

    # Create grammar

    grammar = Grammar.uniform(primitives)

    # Tests
    #p = Program.parse("(lambda ((mrun decrx) $0))")
    #assert p.evaluate([])(1) == 0
    
    # Training data
    
    training = []
    for n in range(5): # distance from the goal
        for m in range(4): # number of examples
            starts = []
            for example in range(m+1):
                (x,y) = (0,0)
                for step in range(1):  # Temporarily made this 1.  I need a better curriculum
                    a = random.choice([1, 2])
                    b = 3 - a
                    sa = random.choice([-1, 1])
                    sb = random.choice([-1, 1])
                    (x,y) = (x+a*sa,y+b*sb)
                starts.append([x, y])
            training.append(Task(
                f"ex_{n+1}_{m+1}",
                arrow(tlist(tint), tlist(tint)),
                [((s,), [0,0]) for s in starts],
            ))

    testing = []
    for i in range(20,23):
        for j in range(20,23):
            testing.append(Task(
                f"ex_{i}_{j}",
                arrow(tlist(tint), tlist(tint)),
                [(([i,j],), [0,0])],
            ))

    # EC iterate

    generator = ecIterator(grammar,
                           training,
                           testingTasks=testing,
                           **args)
    for i, _ in enumerate(generator):
        print('ecIterator count {}'.format(i))