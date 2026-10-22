"""Optional dependency comparison against a user-specified Cargo.lock."""
import argparse
import re
from pathlib import Path

root = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument('reference', type=Path, help='Reference Cargo.lock to compare against')
args = p.parse_args()
def packages(file):
    result = {}
    for name, version in re.findall(r'name = "([^"]+)"\nversion = "([^"]+)"', file.read_text()):
        result.setdefault(name, set()).add(version)
    return result
a = packages(args.reference)
b = packages(root/'Cargo.lock')
conflicts = {k: (sorted(a[k]), sorted(b[k])) for k in a.keys() & b.keys() if not b[k] <= a[k]}
assert not conflicts, conflicts
print(f'All {len(a.keys() & b.keys())} shared dependency names use versions present in the reference lockfile.')
print('Additional dependencies:', sorted(b.keys()-a.keys()))
