"""Compare a prepared reference with native inference output, including file formats."""
import argparse
import json
from pathlib import Path
import numpy as np
import scipy.io
import pyexr
p = argparse.ArgumentParser()
p.add_argument('folder', type=Path)
a = p.parse_args()
n = np.load(a.folder/'expected.npy')
actual = np.fromfile(a.folder/'native/normal.f32', dtype='<f4').reshape(3,*n.shape[:2]).transpose(1,2,0)
mat = scipy.io.loadmat(a.folder/'native/normal.mat')['Normal_est']
exr = pyexr.read(str(a.folder/'native/normal.exr'))
assert np.array_equal(actual, mat), 'MAT layout/value mismatch'
assert np.array_equal(actual, exr), 'EXR layout/value mismatch'
error = np.abs(actual-n)
metrics = {'max_abs':float(error.max()), 'mean_abs':float(error.mean()), 'shape':list(actual.shape), 'mat_exact':True, 'exr_exact':True}
(a.folder/'metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
assert np.isfinite(actual).all() and error.max() < 0.003, metrics
