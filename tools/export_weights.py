"""One-time trusted PyTorch checkpoint conversion; Python is not a runtime dependency."""
import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
state = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
assert isinstance(state, dict) and len(state) == 1292, 'Unexpected checkpoint architecture'
state = {k: v.detach().float().contiguous() for k, v in state.items()}
assert state['Net_first.patch_embed1.proj.weight'].shape == (64, 3, 3, 3), 'Requires uncalibrated weights'
assert all(torch.isfinite(v).all() for v in state.values()), 'Non-finite weights'
a.out.parent.mkdir(parents=True, exist_ok=True)
save_file(state, str(a.out), metadata={'architecture': 'Uni-MS-PS/Transformer_multi_res_7', 'source': a.checkpoint.name})
print(f'Exported {len(state)} tensors, {sum(v.numel() for v in state.values()):,} parameters to {a.out}')
