"""Independent end-to-end check: upstream preprocessing/process vs native CLI, including MAT/EXR."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import cv2
import numpy as np
import scipy.io
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--model', type=Path, required=True)
p.add_argument('--exe', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--cpu', action='store_true')
p.add_argument('--large', action='store_true', help='Exercise 512px tiled refinement')
p.add_argument('--reference-cuda', action='store_true')
p.add_argument('--prepare-only', action='store_true')
a = p.parse_args()
torch.set_num_threads(4)
sys.path.insert(0, str(a.source.resolve()))
from utils import load_imgs_mask, depadding
from Transformer_multi_res_7 import Transformer_multi_res_7

data = a.out/'images'
data.mkdir(parents=True, exist_ok=True)
height, width = (517, 389) if a.large else (69, 53)
y, x = np.mgrid[:height, :width]
mask = ((y-height//2)**2/(height//2-2)**2+(x-width//2)**2/(width//2-2)**2 < 1)
cv2.imwrite(str(data/'mask.png'), mask.astype(np.uint8)*255)
for i in range(3):
    bgr = np.stack([(x*317+y*113+i*500+j*2000)%60000+1000 for j in range(3)], -1).astype(np.uint16)
    cv2.imwrite(str(data/f'{i+1:02}.png'), bgr)
imgs, mask, padding, crop, original = load_imgs_mask(str(data), -1, filenames=['01.png','02.png','03.png'])
model = Transformer_multi_res_7().eval()
model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True))
if a.reference_cuda:
    model.to('cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
with torch.inference_mode():
    result = model.process({'imgs':imgs, 'mask':mask}, int(np.log2(mask.shape[-1]/32))+1)['n']
n = depadding(result[0].permute(1,2,0).cpu().float().numpy(), padding)
n = torch.nn.functional.normalize(torch.from_numpy(n), dim=-1).numpy()
n = np.pad(n, ((crop[0],crop[1]),(crop[2],crop[3]),(0,0)))
np.save(a.out/'expected.npy', n)
if a.prepare_only:
    print(f'Prepared reference in {a.out}')
    sys.exit(0)
command = [str(a.exe.resolve())] + (['--cpu'] if a.cpu else []) + ['infer','--model',str(a.model.resolve()),'--data',str(data.resolve()),'--out',str((a.out/'native').resolve())]
subprocess.run(command, check=True)
actual = np.fromfile(a.out/'native/normal.f32', dtype='<f4').reshape(3,*original[:2]).transpose(1,2,0)
mat = scipy.io.loadmat(a.out/'native/normal.mat')['Normal_est']
assert np.array_equal(actual, mat), 'MAT dimensions/layout mismatch'
error = np.abs(actual-n)
metrics = {'max_abs':float(error.max()), 'mean_abs':float(error.mean()), 'shape':list(actual.shape), 'mat_exact':True}
(a.out/'metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
assert np.isfinite(actual).all() and error.max() < 0.003, metrics
