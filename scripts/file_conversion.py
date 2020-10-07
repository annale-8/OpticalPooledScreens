import sys
from ops.imports import *
import ops.filenames
from ops.nd2_to_tif import nd2_to_tif
from joblib import Parallel, delayed

# input_directory = '20200815_6W-S23/20200815*/*/*.nd2'
input_directory = sys.argv[1]
nd2_files = natsorted(glob(input_directory))

# parallelize file conversion with joblib
r = Parallel(n_jobs=2, verbose=10)(delayed(nd2_to_tif)(file) for file in nd2_files)