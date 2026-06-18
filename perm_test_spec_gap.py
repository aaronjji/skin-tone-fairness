"""
Permutation test: light-vs-dark Δspec for the full model.
Loads test_predictions.csv (score, label, ita), isolates light (ITA>41)
and dark (ITA<=10) groups, computes observed spec gap, then runs a
5000-iteration permutation test with seed=42.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

SEED       = 42
N_ITER     = 5000
THRESHOLD  = 0.50
ITA_LIGHT  = 55.0   # HC-ITA subset: light ITA>55, dark ITA<0
ITA_DARK   = 0.0

df = pd.read_csv("results/test_predictions.csv")
df.columns = [c.strip() for c in df.columns]

# Build masks
light_mask = df["ita"] > ITA_LIGHT
dark_mask  = df["ita"] < ITA_DARK

n_light = int(light_mask.sum())
n_dark  = int(dark_mask.sum())

# Pool light + dark (drop medium)
pool = df[light_mask | dark_mask].copy().reset_index(drop=True)
is_dark_pool = (pool["ita"] <= ITA_DARK).values  # True=dark, False=light

scores = pool["score"].values
labels = pool["label"].values


def spec_at(s, l, t=THRESHOLD):
    preds = (s >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(l, preds, labels=[0, 1]).ravel()
    return tn / (tn + fp) if (tn + fp) > 0 else float("nan")


def delta_spec(dark_flag, s, l):
    sd, ld = s[dark_flag],  l[dark_flag]
    sl, ll = s[~dark_flag], l[~dark_flag]
    return spec_at(sd, ld) - spec_at(sl, ll)


# Observed gap
obs_gap = delta_spec(is_dark_pool, scores, labels)

# Permutation null
rng   = np.random.default_rng(SEED)
nulls = []
idx   = np.arange(len(pool))
for _ in range(N_ITER):
    perm        = rng.permutation(idx)
    fake_dark   = np.zeros(len(pool), dtype=bool)
    fake_dark[perm[:n_dark]] = True
    nulls.append(delta_spec(fake_dark, scores, labels))

nulls  = np.array(nulls)
p_val  = float(np.mean(np.abs(nulls) >= abs(obs_gap)))

# Cohen's h
spec_dark  = spec_at(scores[is_dark_pool],  labels[is_dark_pool])
spec_light = spec_at(scores[~is_dark_pool], labels[~is_dark_pool])
h = 2*np.arcsin(np.sqrt(np.clip(spec_dark,  1e-9, 1-1e-9))) \
  - 2*np.arcsin(np.sqrt(np.clip(spec_light, 1e-9, 1-1e-9)))

print(f"\nLight-vs-Dark specificity permutation test  ({N_ITER} iter, seed={SEED})")
print(f"  ITA thresholds: light > {ITA_LIGHT}, dark <= {ITA_DARK}")
print(f"  n_light = {n_light}, n_dark = {n_dark}")
print(f"  spec_light = {spec_light:.4f}")
print(f"  spec_dark  = {spec_dark:.4f}")
print(f"  Dspec (dark-light) = {obs_gap:+.4f}")
print(f"  Cohen's h          = {h:.4f}")
print(f"  p-value            = {p_val:.4f}")
if p_val == 0:
    print(f"  (p < {1/N_ITER:.4f}  -- no permutation equalled or exceeded observed gap)")
