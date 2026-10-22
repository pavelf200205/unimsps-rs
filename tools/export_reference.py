"""Export independent FP32 network fixtures from the untouched upstream implementation."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--source', type=Path, required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
p.add_argument('--size', type=int, default=32)
p.add_argument('--lights', type=int, default=3)
p.add_argument('--refinement', action='store_true')
a = p.parse_args()
sys.path.insert(0, str(a.source.resolve()))
from Transformer_multi_res_7 import Transformer_multi_res_7
torch.set_num_threads(4)
torch.manual_seed(17)
model = Transformer_multi_res_7().eval()
model.load_state_dict(torch.load(a.checkpoint, map_location='cpu', weights_only=True), strict=True)
net = model.Net_stage if a.refinement else model.Net_first
channels = 6 if a.refinement else 3
x = torch.rand(a.lights, channels, a.size, a.size)
if a.refinement:
    x[:, 3:] = torch.nn.functional.normalize(x[:, 3:]*2-1, dim=1)
with torch.inference_mode():
    out = net([x.unsqueeze(0), torch.zeros(1, 1, a.size, a.size, dtype=torch.bool)])['n']
a.out.mkdir(parents=True, exist_ok=True)
x.numpy().astype('<f4').tofile(a.out/'input.f32')
out.numpy().astype('<f4').tofile(a.out/'normal.f32')
(a.out/'fixture.json').write_text(json.dumps({'shape':list(x.shape), 'refinement':a.refinement, 'torch':torch.__version__}, indent=2))
print(f'Exported {a.out}')
