try:
    import binutil  # required to import from dreamcoder modules
except ModuleNotFoundError:
    import bin.binutil  # alt import if called as module

from dreamcoder.domains.sketch.main import *
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs


if __name__ == '__main__':
    arguments = commandlineArguments(
        featureExtractor=SketchCNN,
        CPUs=numberOfCPUs(),
        helmholtzRatio=0.5,
        recognitionTimeout=3600,
        iterations=6,
        a=3,
        structurePenalty=1.5,
        pseudoCounts=30,
        topK=2,
        maximumFrontier=5,
        solver="python")
    main(arguments)
